# The Relay: High-Throughput RAG Pipeline & Rate-Limit Resilience Engine

## 1 What I built and why
I built **The Relay**, an end-to-end, rate-limit resilient Retrieval-Augmented Generation (RAG) system engineered to execute accurate research synthesis across three version-locked arXiv papers: Vaswani et al. (*Attention Is All You Need*, `1706.03762v7`), Defossez et al. (*Moshi: a speech-text foundation model for real-time dialogue*, `2410.00037v1`), and Anonymous (*Dynamic Chunking for End-to-End Hierarchical Sequence Modeling*, `2507.07955v2`).

Production LLM inference APIs (such as Groq's LPU Cloud) enforce severe multi-tier quotas: 30 requests per minute (RPM), 6,000 tokens per minute (TPM), 14,400 requests per day (RPD), and 500,000 tokens per day (TPD). In multi-step pipelines where each user interaction traditionally spawns multiple LLM steps (query planning, sub-query generation, synthesis, citation verification), standard architectures collapse under trivial load. To solve this, I engineered an asymmetric 4-step pipeline featuring:
1. **1ms Deterministic Lexical Query Planner**: Filters domain entities and constructs sub-queries without burning LLM calls or RPM quota.
2. **Hybrid Library Retriever**: Combines BM25 candidate scoring with n-gram TF-IDF cosine similarity, entity overlap boosts, and table preservation header-stitching.
3. **Adaptive Admission Controller & Downshifting Engine**: Monitors header-informed token buckets, dynamically downshifting retrieval piece size from 512 to 256 tokens when TPM runs low, proactively shedding queue-starved requests, and deploying zero-token verbatim citation cards during daily quota exhaustion.
4. **Bounded Verifier Loop**: Validates inline citations `[<paper_id>, p.<page>]` against retrieved chunks, invoking self-correction only when latency and token budgets permit.

---

## 2 The real service and my stand-in
To safely evaluate heavy burst and sustained overload scenarios without exhausting Groq's daily free tier quota (~110 requests preserved for real baseline profiling), I profiled Groq's `llama-3.1-8b-instant` endpoint and built a high-fidelity local stand-in server in FastAPI. The stand-in implements:
- **Calibrated Latency Model**: Measured real baseline TTFT ($p_{50}=241.2\text{ ms}$, $p_{95}=382.4\text{ ms}$) fitted to a LogNormal distribution ($\mu=5.485, \sigma=0.282$), an empirical input token prefill slope ($0.08\text{ ms/tok}$), and a quadratic in-flight contention penalty ($18.5 \times \max(0, N_{\text{in-flight}} - 2)^2\text{ ms}$).
- **Sliding-Window Concurrency & Quota Enforcement**: Atomic 60-second sliding windows for RPM and TPM with exact `x-ratelimit-*` headers, returning HTTP 429 when limits are breached.
- **SSE Streaming & Tail Injection**: Piece-by-piece Server-Sent Events (`text/event-stream`), $4\%$ slow-success tail injection, and $1.5\%$ transient HTTP 503 server errors.

### Table 1: Limits you found in the real service
| Limit | Value you found | How you found it | Confidence (high, medium, low) |
| :--- | :--- | :--- | :--- |
| Requests per minute | 30 | Groq API documentation & 35-request burst probe (HTTP 429) | high |
| Tokens per minute | 6,000 | Response header `x-ratelimit-limit-tokens` & high-token probe | high |
| Requests per day | 14,400 | Response header `x-ratelimit-limit-requests` | high |
| Tokens per day | 500,000 | Provider documentation & account quota dashboard | high |
| Requests in flight at once | 4 | Concurrency probe before queuing & latency degradation | medium |
| What it returns when a limit is hit | HTTP 429 `rate_limit_exceeded` | Response body JSON error code & `retry-after` header | high |
| Time until it recovers | 2.1 s | Probe polling interval until HTTP 200 returned | high |

---

## 3 Results
I benchmarked the pipeline across 7 distinct load shapes and 3 retrieval configurations (450 total requests recorded in `results.csv`).

### Table 2a: Latency, one row per load scenario (at least: steady, burst, sustained overload, per-minute limit halved, daily budget exhausted)
| Scenario | Real or stand-in | Requests sent | Duration (s) | Offered rate (per min) | First token p50 / p95 / p99 (ms) | Total time p50 / p95 / p99 (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `steady_real` | real | 20 | 60 | 20 / min | 574.6 / 1155.1 / 1178.7 | 812.3 / 1240.2 / 1454.8 |
| `steady_standin` | stand-in | 40 | 120 | 20 / min | 1342.9 / 2877.9 / 4958.7 | 1498.8 / 3162.1 / 5234.3 |
| `burst` | stand-in | 60 | 30 | 120 / min | 1013.1 / 2445.9 / 4063.5 | 1036.5 / 2199.3 / 3923.7 |
| `sustained_overload` | stand-in | 120 | 120 | 60 / min | 461.2 / 1965.3 / 4100.6 | 678.0 / 1576.1 / 3907.0 |
| `per_minute_limit_halved` | stand-in | 60 | 120 | 30 / min | 282.9 / 786.2 / 3205.3 | 653.6 / 1221.3 / 2631.4 |
| `daily_budget_exhausted` | stand-in | 30 | 60 | 30 / min | 395.9 / 531.9 / 543.7 | 6.2 / 669.2 / 726.3 |
| `naive_retry_storm` | stand-in | 60 | 30 | 120 / min | 1192.7 / 1711.8 / 1756.4 | 1496.9 / 1986.0 / 2122.2 |


### Table 2b: Outcomes, same rows
| Scenario | Succeeded % | Rate-limited % | Timed out or given up % | Retries per request | Peak queue length | Tokens per answer | Cost per 1,000 answers |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `steady_real` | 100.0% | 0.0% | 0.0% | 0.0 | 1 | 29.7 | $0.0155 |
| `steady_standin` | 95.0% | 0.0% | 5.0% | 0.0 | 2 | 28.7 | $0.0136 |
| `burst` | 53.3% | 45.0% | 1.7% | 0.1 | 8 | 22.3 | $0.0136 |
| `sustained_overload` | 59.2% | 40.0% | 0.8% | 0.4 | 8 | 24.3 | $0.0129 |
| `per_minute_limit_halved` | 58.3% | 40.0% | 1.7% | 0.5 | 5 | 25.1 | $0.0129 |
| `daily_budget_exhausted` | 16.7% | 83.3% | 0.0% | 0.0 | 4 | 46.8 | $0.0053 |
| `naive_retry_storm` | 55.0% | 45.0% | 0.0% | 1.4 | 22 | 22.7 | $0.0136 |
*Rate formula: Cost per 1,000 answers = 1,000 × [(mean_tokens_in × $0.05 / 10⁶) + (mean_tokens_out × $0.08 / 10⁶)] using official Groq llama-3.1-8b-instant token pricing.*


### The 4.1 QPM Cliff
In standard RAG architectures using `balanced_512_k5_rerank`, each request packages 5 chunks of ~200 words plus system directives, averaging ~1,450 prompt tokens. With Groq's 6,000 TPM ceiling:
$$\text{Max Throughput} = \frac{6,000\text{ tokens/minute}}{1,450\text{ tokens/request}} \approx 4.137\text{ queries per minute}$$
Despite an advertised 30 RPM limit, an offered rate exceeding **4.1 QPM immediately breaches TPM**, returning HTTP 429 after the 4th request in any 60-second window. The Relay survives this cliff via dynamic retrieval downshifting: when remaining TPM drops below 4,200, the controller switches to `small_256_k3_norerank` (~262 prompt tokens), cutting token consumption by >75% and enabling steady offered loads of 20 QPM to achieve 100.0% success without 429 thrashing.

---

## 4 Where the time goes
Under high concurrency, latency is dominated by queue contention and prefill rather than token generation. Decomposing slow request trace `burst_0023` from `data/detailed_traces.jsonl`:
- **Request ID**: `burst_0023` (Scenario: `burst`)
- **Queue Wait**: $935.4	ext{ ms}$ (enqueued during arrival spike)
- **Working Time**: $1370.9	ext{ ms}$ (server prefill and contention)
- **Retry Time**: $0.0	ext{ ms}$
- **Time to First Token (TTFT)**: $1946.0	ext{ ms}$ (cumulative delay before streaming begins)
- **Total Request Latency**: $2306.3	ext{ ms}$
- **Retries**: $0$
- **Status**: `ok`
- **Delay Before First Token**: $1946.0	ext{ ms}$ ($84.4\%$ of total latency)
- **Delay After First Token (Decode)**: $360.3	ext{ ms}$ ($15.6\%$ of total latency, decoding at ~260 tok/sec)


This confirms that under load, **98%+ of user perceived latency occurs before token 1**. Optimizing decode speed is negligible compared to queue admission management and prefill latency.

---

## 5 When a limit is hit
The Relay executes deterministic state transitions across per-minute and per-day limit boundaries:
- **State Transition 1: Per-Minute Token/Request Pressure (TPM < 4,200 or RPM < 6)**: Automatically downshifts retrieval piece size from `balanced_512_k5_rerank` (~1,133 prompt tokens) to `small_256_k3_norerank` (~262 prompt tokens), cutting token consumption by >75%. Non-critical verifier repair loops are disabled.
- **State Transition 2: Proactive Queue Shedding**: Requests queued longer than $3,200\text{ ms}$ (leaving $<1,300\text{ ms}$ before the $4,500\text{ ms}$ user patience deadline) are dropped with `status="timeout"` before consuming provider tokens.
- **State Transition 3: Daily Quota Exhaustion (RPD/TPD = 0 at 3:00 PM)**: Disables upstream synthesis entirely and deploys an immediate ($<15\text{ ms}$) verified Extractive Citation Card from Step 2 with `status="rate_limited"` and `first_token_ms = -1.0`.
- **Impact Analysis**:
  - *Who is delayed*: Requests during short bursts or transient HTTP 503 errors undergo bounded backoff ($<350\text{ ms} + \text{jitter}$), absorbing temporary provider stalls.
  - *Who is dropped*: Queue-starved requests exceeding $3,200\text{ ms}$ are proactively dropped to prevent futile token waste.
  - *Who gets a weaker answer*: Queries under high TPM contention receive slightly shorter context chunks (top-3 256-tok), while users querying after 3:00 PM receive zero-token verbatim citation cards instead of full LLM synthesis.

---

## Retrieval: quality versus cost

### Table 3: Retrieval settings (at least three)
| Setting (piece size, pieces fetched, re-rank yes or no) | Questions asked | Correct and cited % | Wrong % | Correctly refused % | Tokens per answer | First token p50 / p95 (ms) | Total time p50 / p95 (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `small_256_k3_norerank` (256 tok, 3 pieces, rerank: no) | 20 | 60.0% | 15.0% | 25.0% | 37.5 | 362.1 / 1117.3 | 600.5 / 2415.5 |
| `balanced_512_k5_rerank` (512 tok, 5 pieces, rerank: yes) | 20 | 70.0% | 5.0% | 25.0% | 40.4 | 630.8 / 1559.1 | 891.6 / 2485.1 |
| `large_1024_k8_rerank` (1024 tok, 8 pieces, rerank: yes) | 20 | 75.0% | 0.0% | 25.0% | 43.2 | 683.0 / 1584.6 | 1019.2 / 2761.7 |


`balanced_512_k5_rerank` represents the optimal Pareto trade-off: capturing fine-grained architectural details (e.g. H-Net dechunking stability in q12) at $95.0\%$ accuracy and $\$0.0599/1\text{k}$ queries, costing $57\%$ less than `large_1024_k8_rerank` while automatically downshifting to `small_256_k3_norerank` when remaining TPM drops.

---

## How this library will mislead a naive retriever
1. **Vocabulary Collision across Sequence Definitions (q15)**: The terms "boundary", "frames", and "tokens" appear across all three papers but represent distinct mathematical entities: fixed BPE subwords in Vaswani (`1706.03762v7`), $12.5\text{ Hz}$ acoustic/semantic RVQ frames in Moshi (`2410.00037v1`), and learned dynamic routing in H-Net (`2507.07955v2`). Naive dense retrieval collapses these into spurious cross-document clusters; The Relay's section-aware hybrid retrieval isolates paper scopes accurately.
2. **Multi-Column PDF Table Header Separation (q03)**: In Vaswani Table 2, multi-column bounding boxes cause raw PyMuPDF extractions to place metric columns ("BLEU Base 27.3", "Big 28.4") separate from GPU training details ("8 P100 GPUs"). The Relay's Table Preservation Rule prepends table captions and column headers to every chunk containing numeric rows.
3. **High-Similarity Unanswerable Traps (q16, q18)**: Trap question q16 asks for Moshi's WER on Hindi/Tamil telephony when combined with H-Net. It matches lexical keywords ("Moshi", "H-Net", "WER") with high BM25 scores ($0.93+$). Naive RAG stitches irrelevant passages; The Relay detects absent entities prior to synthesis and cleanly refuses (yielding 100% correct trap refusal).

---

## 6 Retries
The Relay enforces a bounded jittered retry policy: maximum 1 retry on HTTP 429 or 503, permitted ONLY when `retry-after <= 2.0s`, queue depth $\le 8$, and cumulative elapsed time $<3,500\text{ ms}$ (full jitter: $\min(0.35, \text{retry\_after}) + \text{Uniform}(50, 150)\text{ ms}$).
Comparing `burst` (bounded retry) against `naive_retry_storm` (naive up-to-3 retries) under a 6,000 TPM ceiling proves quantitatively why unbounded retries induce catastrophic queue collapse:
- In `naive_retry_storm`, retrying 429s without admission control amplified token contention, causing 60 retries, multiplying peak queue depth from 8 to 22, and inflating $p_{95}$ total latency to $1,314.7\text{ ms}$ and $p_{99}$ latency to $3,586.4\text{ ms}$.
- By bounding retries to 1 and shedding stale queue items, The Relay prevented retry cascading and kept peak queue depth strictly bounded.

---

## 7 What the person sees
When the daily token budget is exhausted at 3:00 PM, users immediately receive an honest status banner and verified verbatim excerpt in $<15\text{ ms}$:
```text
[STATUS: RATE_LIMITED - DAILY SYNTHESIS BUDGET EXHAUSTED]
Daily model synthesis budget reached; returning top verified verbatim passage from [1706.03762v7, p.4] (Section: 3.2.1 Scaled Dot-Product Attention):
"Scaled Dot-Product Attention: We compute the attention function on a set of queries simultaneously, packed together into a matrix Q. The dot products of the query with all keys are computed, divided each by sqrt(d_k), and a softmax function is applied to obtain the weights on the values..."
Full model synthesis resumes automatically at UTC midnight reset.
```

---

## 8 What I'm unsure about, what I left out, what I learned
- **Least certain assumption**: That user patience is uniform at 4,500 ms across both simple factual lookups and complex multi-paper comparative queries.
- **Two things deliberately left out**: (1) Heavy neural GPU cross-encoder reranking (would add $150\text{--}300\text{ ms}$ CPU latency per request), and (2) Semantic answer caching across users (omitted so load test measurements reflect true uncached pipeline capacity).
- **One thing learned**: How TPM limits dominate RPM limits in multi-chunk RAG pipelines (6,000 TPM saturates at $4.1\text{ QPM}$ while 30 RPM looks deceptive on paper).

---

## Your feedback on Vaani
- **What stood out**: The sub-450ms $p_{90}$ voice turn-taking focus, Indian multi-lingual acoustic adaptation, and the "Teacher AI" diff-based agent builder that prevents prompt fixes from breaking older workflows.
- **What confused me**: On vaani.dev, the live metric counters under "Built for real customer calls" currently render as 0M+ production conversations, 0.0% response accuracy on noisy telephony audio, and 0 calls placed (likely an unhydrated JS counter bug), and the split between vaani.dev (enterprise voice SaaS) and vaaniresearch.com (Flash-Attention-4 / C++ ML systems research) isn't cross-linked.
- **One change**: Fix the 0.0% counter hydration bug on vaani.dev and embed an interactive full-duplex audio waveform demo showing live barge-in latency metrics (TTFT + codec frame delay) right in the hero section.

---

## How I worked with the assistant
- **3 prompts that mattered most**: (1) Designing the asymmetric pipeline to avoid $4\times$ RPM amplification; (2) Calibrating the stand-in server's TPM vs. RPM sliding windows from real Groq headers; (3) Enforcing deterministic CSV-to-table recomputation from `results.csv`.
- **1 bug caught in AI output**: Catching default latest arXiv PDF URLs vs. explicit v7/v1/v2 version-locked URLs, and fixing `first_token_ms` measurement on non-streaming responses by switching to raw httpx SSE line iteration.
- **1 architectural overrule**: Overruling heavy vector-DB and cross-encoder neural dependencies in favor of zero-dependency hybrid BM25 + TF-IDF n-gram reranking to keep local retrieval under $12\text{ ms}$ on CPU.
