"""
rate_limiter.py - Client-side token bucket, admission controller & retry guard.
Implements:
  - Header-informed token bucket tracking (RPM, TPM, RPD, TPD)
  - Proactive queue shedding under 4500ms user patience budget (drop if queued > 3200ms)
  - Automatic retrieval downshifting when remaining_tpm < 1500 or remaining_rpm < 6
  - Zero-token extractive fallback when daily budget is exhausted
  - Bounded jittered retry policy (max 1 retry, retry_after <= 2.0s, queue <= 8)
"""

import asyncio
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class AdmissionController:
    @classmethod
    def from_config(cls, config_path: Optional[Path] = None, naive_retry_mode: bool = False, **overrides) -> "AdmissionController":
        path = config_path or (Path(__file__).resolve().parent.parent / "config.yaml")
        kwargs: Dict[str, Any] = {"naive_retry_mode": naive_retry_mode}
        if path.exists():
            try:
                import yaml
                with open(path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                ac = cfg.get("admission_control", {})
                if "user_patience_ms" in ac:
                    kwargs["user_patience_ms"] = float(ac["user_patience_ms"])
                if "queue_timeout_shed_ms" in ac:
                    kwargs["queue_timeout_ms"] = float(ac["queue_timeout_shed_ms"])
                if "synthesis_budget_ms" in ac:
                    kwargs["synthesis_budget_ms"] = float(ac["synthesis_budget_ms"])
                if "downshift_tpm_threshold" in ac:
                    kwargs["downshift_tpm_threshold"] = int(ac["downshift_tpm_threshold"])
                if "downshift_rpm_threshold" in ac:
                    kwargs["downshift_rpm_threshold"] = int(ac["downshift_rpm_threshold"])
                if "max_retries" in ac:
                    kwargs["max_retries"] = int(ac["max_retries"])
                if "max_queue_depth" in ac:
                    kwargs["max_queue_depth"] = int(ac["max_queue_depth"])
            except Exception as e:
                logger.warning(f"Could not load admission config from {path}: {e}")
        kwargs.update(overrides)
        return cls(**kwargs)

    def __init__(
        self,
        user_patience_ms: float = 4500.0,
        queue_timeout_ms: float = 3200.0,
        synthesis_budget_ms: float = 1300.0,
        downshift_tpm_threshold: int = 4200,
        downshift_rpm_threshold: int = 6,
        max_retries: int = 1,
        max_queue_depth: int = 8,
        naive_retry_mode: bool = False,
    ):
        self.user_patience_ms = user_patience_ms
        self.queue_timeout_ms = queue_timeout_ms
        self.synthesis_budget_ms = synthesis_budget_ms
        self.downshift_tpm_threshold = downshift_tpm_threshold
        self.downshift_rpm_threshold = downshift_rpm_threshold
        self.max_retries = 3 if naive_retry_mode else max_retries
        self.max_queue_depth = max_queue_depth
        self.naive_retry_mode = naive_retry_mode

        # Real-time provider quota state
        self.remaining_rpm = 30
        self.remaining_tpm = 6000
        self.remaining_rpd = 14400
        self.remaining_tpd = 500000

        # Concurrency & queue metrics
        self.current_queue_depth = 0
        self.peak_queue_length = 0
        self.in_flight_requests = 0
        self.lock = asyncio.Lock()

    def update_from_headers(self, headers: Dict[str, str]):
        """Updates remaining quotas from real Groq / Stand-in response headers."""
        for k, v in headers.items():
            k_lower = k.lower()
            try:
                if k_lower in ("x-ratelimit-remaining-requests-daily", "x-ratelimit-remaining-day-requests"):
                    self.remaining_rpd = int(v)
                elif k_lower in ("x-ratelimit-remaining-tokens-daily", "x-ratelimit-remaining-day-tokens"):
                    self.remaining_tpd = int(v)
                elif k_lower == "x-ratelimit-remaining-requests":
                    val = int(v)
                    self.remaining_rpm = min(30, val)
                elif k_lower == "x-ratelimit-remaining-tokens":
                    self.remaining_tpm = int(v)
            except Exception:
                pass

    def should_downshift(self) -> bool:
        """Determines if retrieval should downshift to small_256_k3_norerank."""
        return (self.remaining_tpm < self.downshift_tpm_threshold) or (self.remaining_rpm < self.downshift_rpm_threshold)

    def is_daily_budget_exhausted(self) -> bool:
        """Checks if daily RPD/TPD budget is exhausted."""
        return (self.remaining_rpd <= 0) or (self.remaining_tpd < 200)

    def should_proactively_shed(self, queue_wait_ms: float) -> bool:
        """
        Proactive cancellation: if queued > 3200ms, less than 1300ms remains before
        the 4500ms user patience deadline; cancel to save provider tokens.
        """
        if self.naive_retry_mode:
            # Naive mode doesn't shed; it queues indefinitely and retries aggressively
            return False
        return queue_wait_ms > self.queue_timeout_ms

    def can_retry(
        self,
        attempt: int,
        status_code: int,
        retry_after_sec: float,
        elapsed_ms: float,
    ) -> Tuple[bool, float]:
        """
        Bounded retry policy:
          - Maximum 1 retry on 429 or 503
          - ONLY if retry-after <= 2.0s
          - ONLY if elapsed_ms + retry_after_ms < 3500ms
          - ONLY if queue_depth <= 8
        """
        if status_code not in (429, 503):
            return False, 0.0

        if self.naive_retry_mode:
            if attempt < self.max_retries:
                # Naive mode retries up to 3 times aggressively with immediate retry (0.15s)
                return True, 0.15
            return False, 0.0

        if attempt >= self.max_retries:
            return False, 0.0

        if retry_after_sec > 2.0:
            return False, 0.0

        retry_ms = retry_after_sec * 1000.0
        if elapsed_ms + retry_ms >= 3500.0:
            return False, 0.0

        if self.current_queue_depth > self.max_queue_depth:
            return False, 0.0

        # Full jitter backoff: min(0.35, retry_after) + Uniform(50, 150) ms
        jitter_sec = random.uniform(0.05, 0.15)
        sleep_sec = min(0.35, retry_after_sec) + jitter_sec
        return True, sleep_sec
