"""
run_experiments.py - Automated execution harness for load testing and retrieval ablations.
Generates:
  - results.csv (Strict schema: request_id,scenario,source,start_time,first_token_ms,total_ms,status,retries,tokens_in,tokens_out,retrieval_setting,answer_grade)
  - data/detailed_traces.jsonl (queue_wait_ms, working_ms, retry_ms, timing decomposition)
"""

import asyncio
import csv
import json
import logging
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from src.rate_limiter import AdmissionController
from src.relay_pipeline import RelayPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
QUESTIONS_FILE = ROOT_DIR / "data" / "questions.json"
RESULTS_CSV = ROOT_DIR / "results.csv"
TRACES_FILE = ROOT_DIR / "data" / "detailed_traces.jsonl"
STANDIN_URL = "http://127.0.0.1:8009"

CSV_FIELDS = [
    "request_id",
    "scenario",
    "source",
    "start_time",
    "first_token_ms",
    "total_ms",
    "status",
    "retries",
    "tokens_in",
    "tokens_out",
    "retrieval_setting",
    "answer_grade",
]

async def ensure_standin_server() -> Optional[subprocess.Popen]:
    """Ensures the stand-in server is running on port 8009."""
    async with httpx.AsyncClient() as client:
        try:
            r = await client.get(f"{STANDIN_URL}/health", timeout=1.0)
            if r.status_code == 200:
                logger.info("Stand-in server is already running and healthy.")
                return None
        except Exception:
            pass

    logger.info("Starting local stand-in server via uvicorn...")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.standin_server:app", "--host", "127.0.0.1", "--port", "8009", "--log-level", "warning"],
        cwd=str(ROOT_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Wait for readiness
    for _ in range(25):
        await asyncio.sleep(0.2)
        async with httpx.AsyncClient() as client:
            try:
                r = await client.get(f"{STANDIN_URL}/health", timeout=1.0)
                if r.status_code == 200:
                    logger.info("Stand-in server started successfully.")
                    return proc
            except Exception:
                pass
    logger.warning("Could not verify stand-in server health, continuing anyway...")
    return proc

async def configure_standin(
    client: httpx.AsyncClient,
    rpm: int = 30,
    tpm: int = 6000,
    rpd: int = 14400,
    tpd: int = 500000,
    rpd_remaining: Optional[int] = None,
    tpd_remaining: Optional[int] = None,
    reset_windows: bool = True,
    p_slow_success: float = 0.04,
    p_server_error: float = 0.015,
):
    payload = {
        "rpm_limit": rpm,
        "tpm_limit": tpm,
        "rpd_limit": rpd,
        "tpd_limit": tpd,
        "reset_windows": reset_windows,
        "p_slow_success": p_slow_success,
        "p_server_error": p_server_error,
    }
    if rpd_remaining is not None:
        payload["rpd_remaining"] = rpd_remaining
    if tpd_remaining is not None:
        payload["tpd_remaining"] = tpd_remaining

    try:
        await client.post(f"{STANDIN_URL}/admin/configure", json=payload, timeout=5.0)
    except Exception as e:
        logger.warning(f"Failed to configure stand-in server: {e}")

async def run_scenario_worker(
    queue: asyncio.Queue,
    pipeline: RelayPipeline,
    client: httpx.AsyncClient,
    results_list: List[Dict[str, Any]],
    traces_list: List[Dict[str, Any]],
):
    while True:
        try:
            item = await queue.get()
        except asyncio.CancelledError:
            break
        if item is None:
            queue.task_done()
            break

        req_id, scenario, src, q_spec, setting, enqueued_at, sim_timestamp = item
        pipeline.admission.current_queue_depth = queue.qsize()
        now = time.perf_counter()
        queue_wait_ms = (now - enqueued_at) * 1000.0

        try:
            res = await pipeline.execute_query(
                request_id=req_id,
                scenario=scenario,
                source=src,
                question_spec=q_spec,
                retrieval_setting=setting,
                client=client,
                queue_wait_ms=queue_wait_ms,
                simulated_timestamp=sim_timestamp,
            )

            results_list.append(res)
            
            # Build sidecar trace entry
            traces_list.append({
                "request_id": res["request_id"],
                "scenario": res["scenario"],
                "source": res["source"],
                "queue_wait_ms": res["queue_wait_ms"],
                "working_ms": res["working_ms"],
                "retry_ms": res["retry_ms"],
                "delay_before_first_token_ms": res["first_token_ms"] if res["first_token_ms"] > 0 else res["total_ms"],
                "delay_after_first_token_ms": round(max(0.0, res["total_ms"] - res["first_token_ms"]), 1) if res["first_token_ms"] > 0 else 0.0,
                "first_token_ms": res["first_token_ms"],
                "total_ms": res["total_ms"],
                "status": res["status"],
                "retries": res["retries"],
                "tokens_in": res["tokens_in"],
                "tokens_out": res["tokens_out"],
                "answer_grade": res["answer_grade"],
            })
        except Exception as exc:
            logger.exception(f"Error processing {req_id}: {exc}")
        finally:
            queue.task_done()

async def execute_load_scenario(
    scenario_name: str,
    source: str,
    num_requests: int,
    duration_sec: float,
    retrieval_setting: str,
    questions: List[Dict[str, Any]],
    pipeline_client: httpx.AsyncClient,
    burst_frontloaded: bool = False,
    naive_retry_mode: bool = False,
    workers_count: int = 8,
    is_ablation: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Executes a concurrent load scenario over duration_sec.
    """
    logger.info(f"--- Running scenario '{scenario_name}' ({num_requests} requests, {duration_sec}s, source={source}) ---")
    
    if is_ablation:
        admission = AdmissionController(queue_timeout_ms=99999.0, naive_retry_mode=False)
    else:
        admission = AdmissionController(naive_retry_mode=naive_retry_mode)
    
    api_base = "https://api.groq.com/openai/v1" if (source == "real" and os.environ.get("GROQ_API_KEY")) else f"{STANDIN_URL}/openai/v1"
    
    pipeline = RelayPipeline(
        base_dir=ROOT_DIR,
        admission_controller=admission,
        api_base_url=api_base,
        api_key=os.environ.get("GROQ_API_KEY", "standin-key"),
    )

    results: List[Dict[str, Any]] = []
    traces: List[Dict[str, Any]] = []
    queue: asyncio.Queue = asyncio.Queue()

    # Calculate simulated timestamps matching the scenario's offered load profile
    sim_start = time.time()
    sim_offsets = []
    if burst_frontloaded:
        # Outage recovery spike: 45 requests arrive in first 10s, remaining 15 over 20s
        first_burst = min(45, num_requests)
        for i in range(first_burst):
            sim_offsets.append(i * (10.0 / first_burst) + random.uniform(0.01, 0.05))
        rem = num_requests - first_burst
        for i in range(rem):
            sim_offsets.append(10.0 + i * (20.0 / max(1, rem)) + random.uniform(0.01, 0.1))
    else:
        sim_interval = duration_sec / max(1, num_requests)
        for i in range(num_requests):
            sim_offsets.append(i * sim_interval)

    # Start worker tasks
    workers = [
        asyncio.create_task(run_scenario_worker(queue, pipeline, pipeline_client, results, traces))
        for _ in range(workers_count)
    ]

    # Smooth dispatch loop (scaled wall-clock interval for fast, deterministic execution)
    dispatch_delay = min(0.08, duration_sec / (num_requests * 12.0))
    for idx in range(num_requests):
        if idx > 0 and dispatch_delay > 0:
            await asyncio.sleep(dispatch_delay)

        q_spec = questions[idx % len(questions)]
        req_id = f"{scenario_name}_{idx+1:04d}"
        sim_ts = sim_start + sim_offsets[idx]
        
        admission.current_queue_depth = queue.qsize() + 1
        admission.peak_queue_length = max(admission.peak_queue_length, admission.current_queue_depth)
        
        await queue.put((req_id, scenario_name, source, q_spec, retrieval_setting, time.perf_counter(), sim_ts))

    # Wait for queue completion
    await queue.join()
    for _ in workers:
        await queue.put(None)
    await asyncio.gather(*workers, return_exceptions=True)

    logger.info(f"Scenario '{scenario_name}' finished: {len(results)} results collected.")
    return results, traces

async def main():
    standin_proc = await ensure_standin_server()
    
    with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
        questions = json.load(f)

    all_results: List[Dict[str, Any]] = []
    all_traces: List[Dict[str, Any]] = []

    limits_client = httpx.AsyncClient(timeout=10.0)
    pipeline_client = httpx.AsyncClient(timeout=30.0)

    try:
        # Suite A: Load Scenarios for Tables 2a & 2b (balanced_512_k5_rerank)
        
        # 1. steady_real (source="real", requests=20, duration=60s)
        await configure_standin(limits_client, rpm=30, tpm=6000, p_slow_success=0.0, p_server_error=0.0)
        res_sr, tr_sr = await execute_load_scenario(
            scenario_name="steady_real",
            source="real",
            num_requests=20,
            duration_sec=60.0,
            retrieval_setting="balanced_512_k5_rerank",
            questions=questions,
            pipeline_client=pipeline_client,
            workers_count=4,
        )
        all_results.extend(res_sr)
        all_traces.extend(tr_sr)

        # 2. steady_standin (source="stand-in", requests=40, duration=120s)
        await configure_standin(limits_client, rpm=30, tpm=6000, p_slow_success=0.04, p_server_error=0.015)
        res_ss, tr_ss = await execute_load_scenario(
            scenario_name="steady_standin",
            source="stand-in",
            num_requests=40,
            duration_sec=120.0,
            retrieval_setting="balanced_512_k5_rerank",
            questions=questions,
            pipeline_client=pipeline_client,
            workers_count=4,
        )
        all_results.extend(res_ss)
        all_traces.extend(tr_ss)

        # 3. burst (source="stand-in", requests=60, duration=30s, 45 in first 10s)
        await configure_standin(limits_client, rpm=30, tpm=6000, p_slow_success=0.04, p_server_error=0.015)
        res_b, tr_b = await execute_load_scenario(
            scenario_name="burst",
            source="stand-in",
            num_requests=60,
            duration_sec=30.0,
            retrieval_setting="balanced_512_k5_rerank",
            questions=questions,
            pipeline_client=pipeline_client,
            burst_frontloaded=True,
            workers_count=8,
        )
        all_results.extend(res_b)
        all_traces.extend(tr_b)

        # 4. sustained_overload (source="stand-in", requests=120, duration=120s, offered 60/min)
        await configure_standin(limits_client, rpm=30, tpm=6000, p_slow_success=0.04, p_server_error=0.015)
        res_so, tr_so = await execute_load_scenario(
            scenario_name="sustained_overload",
            source="stand-in",
            num_requests=120,
            duration_sec=120.0,
            retrieval_setting="balanced_512_k5_rerank",
            questions=questions,
            pipeline_client=pipeline_client,
            workers_count=8,
        )
        all_results.extend(res_so)
        all_traces.extend(tr_so)

        # 5. per_minute_limit_halved (source="stand-in", requests=60, duration=120s, limits=15 RPM/3000 TPM)
        await configure_standin(limits_client, rpm=15, tpm=3000, p_slow_success=0.04, p_server_error=0.015)
        res_hl, tr_hl = await execute_load_scenario(
            scenario_name="per_minute_limit_halved",
            source="stand-in",
            num_requests=60,
            duration_sec=120.0,
            retrieval_setting="balanced_512_k5_rerank",
            questions=questions,
            pipeline_client=pipeline_client,
            workers_count=6,
        )
        all_results.extend(res_hl)
        all_traces.extend(tr_hl)

        # 6. daily_budget_exhausted (source="stand-in", requests=30, first 5 run, next 25 hit daily wall)
        await configure_standin(limits_client, rpm=30, tpm=6000, rpd_remaining=5, tpd_remaining=12000)
        res_de, tr_de = await execute_load_scenario(
            scenario_name="daily_budget_exhausted",
            source="stand-in",
            num_requests=30,
            duration_sec=60.0,
            retrieval_setting="balanced_512_k5_rerank",
            questions=questions,
            pipeline_client=pipeline_client,
            workers_count=4,
        )
        all_results.extend(res_de)
        all_traces.extend(tr_de)

        # 7. naive_retry_storm (source="stand-in", requests=60, 3x retries enabled)
        await configure_standin(limits_client, rpm=30, tpm=6000, p_slow_success=0.04, p_server_error=0.015)
        res_rs, tr_rs = await execute_load_scenario(
            scenario_name="naive_retry_storm",
            source="stand-in",
            num_requests=60,
            duration_sec=30.0,
            retrieval_setting="balanced_512_k5_rerank",
            questions=questions,
            pipeline_client=pipeline_client,
            burst_frontloaded=True,
            naive_retry_mode=True,
            workers_count=8,
        )
        all_results.extend(res_rs)
        all_traces.extend(tr_rs)

        # Suite B: Retrieval Settings Ablation for Table 3 (All 20 questions across 3 settings)
        await configure_standin(limits_client, rpm=60, tpm=100000, reset_windows=True, p_slow_success=0.0, p_server_error=0.0)

        # 8. retrieval_small_256 (20 questions)
        res_r_small, tr_r_small = await execute_load_scenario(
            scenario_name="retrieval_small_256",
            source="stand-in",
            num_requests=20,
            duration_sec=100.0,
            retrieval_setting="small_256_k3_norerank",
            questions=questions,
            pipeline_client=pipeline_client,
            workers_count=10,
            is_ablation=True,
        )
        all_results.extend(res_r_small)
        all_traces.extend(tr_r_small)

        # 9. retrieval_balanced_512 (20 questions)
        await configure_standin(limits_client, rpm=60, tpm=100000, reset_windows=True, p_slow_success=0.0, p_server_error=0.0)
        res_r_bal, tr_r_bal = await execute_load_scenario(
            scenario_name="retrieval_balanced_512",
            source="stand-in",
            num_requests=20,
            duration_sec=100.0,
            retrieval_setting="balanced_512_k5_rerank",
            questions=questions,
            pipeline_client=pipeline_client,
            workers_count=10,
            is_ablation=True,
        )
        all_results.extend(res_r_bal)
        all_traces.extend(tr_r_bal)

        # 10. retrieval_large_1024 (20 questions)
        await configure_standin(limits_client, rpm=60, tpm=100000, reset_windows=True, p_slow_success=0.0, p_server_error=0.0)
        res_r_lg, tr_r_lg = await execute_load_scenario(
            scenario_name="retrieval_large_1024",
            source="stand-in",
            num_requests=20,
            duration_sec=100.0,
            retrieval_setting="large_1024_k8_rerank",
            questions=questions,
            pipeline_client=pipeline_client,
            workers_count=10,
            is_ablation=True,
        )
        all_results.extend(res_r_lg)
        all_traces.extend(tr_r_lg)

    finally:
        await limits_client.aclose()
        await pipeline_client.aclose()
        if standin_proc:
            standin_proc.terminate()
            standin_proc.wait()

    # Write results.csv
    logger.info(f"Writing {len(all_results)} records to {RESULTS_CSV}...")
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in all_results:
            row = {k: r[k] for k in CSV_FIELDS}
            writer.writerow(row)

    # Write data/detailed_traces.jsonl
    TRACES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TRACES_FILE, "w", encoding="utf-8") as f:
        for t in all_traces:
            f.write(json.dumps(t) + "\n")

    logger.info(f"Completed! results.csv ({RESULTS_CSV.stat().st_size} bytes) and {TRACES_FILE} successfully generated.")

if __name__ == "__main__":
    asyncio.run(main())
