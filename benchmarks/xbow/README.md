# XBOW impact benchmark (offensive / Scenario 2)

Measures whether CTI poisoning degrades an LLM pentest agent's **flag-capture rate** on the
[XBOW validation-benchmarks](https://github.com/xbow-engineering/validation-benchmarks) — 104
local Dockerized CTF challenges. The agent (Claude backend) exploits each target with poison
OFF vs ON; a drop in solve-rate = real downstream impact (stronger than the in-sandbox PDS proxy).

> ⚠️ **Scope:** this is offensive tooling (Scenario 2), which the defensive harness's `SAFETY.md`
> currently puts out of scope. Kept as a separate, clearly-labelled module. Authorized-use only:
> XBOW challenges are intentionally-vulnerable apps you build + run **locally on a host you own**,
> with a flag you bake in yourself. `SAFETY.md` must be reconciled before committing this.

## Host setup (Windows, verified 2026)

1. **Docker Desktop** installed + running.
2. **⚠️ Disable the containerd image store** (Settings → General → uncheck *"Use containerd for
   pulling and storing images"* → Apply & Restart). The classic **overlay2** store is required —
   containerd fails to extract XBOW's old base images (e.g. `mysql:5.7.15`:
   *"failed to get reader from content store … not found"*). Verify: `docker info --format '{{.Driver}}'` → `overlay2`.
3. Clone the benchmarks next to this repo: `git clone https://github.com/xbow-engineering/validation-benchmarks ../xbow-benchmarks`
4. `pip install anthropic pyyaml` ; set `ANTHROPIC_API_KEY`.
5. Agent commands run in **WSL** by default (`--agent-shell wsl`, real Linux tooling); pass
   `--agent-shell native` to use the host shell instead.

The runner auto-fixes XBOW's invalid `expose: host:container` compose entries (docker compose v2
rejects them) by writing a normalized `docker-compose.normalized.yml` per benchmark.

## Two agent backends (victim consistency)

The Kaggle RQ1–3 victim is **Qwen**; this benchmark needs a capable agent or it floors at 0 solves
(no signal). To avoid silently mixing victims, run **both**:

- **Claude (impact arm)** — `--backend claude` (default, model `claude-opus-4-8`, needs `ANTHROPIC_API_KEY`).
  Strong agent → poison degradation is measurable.
- **Qwen (consistency arm)** — `--backend openai`, pointing at a local vLLM serving Qwen2.5-7B
  (the same victim as RQ1–3), bridging the in-sandbox results to the real-impact metric:
  ```bash
  pip install openai vllm
  vllm serve Qwen/Qwen2.5-7B-Instruct --enable-auto-tool-choice --tool-call-parser hermes  # :8000
  ```

Report the victim per experiment and frame the cross-model result as **transferability**
(mechanism validated in-sandbox on Qwen → impact transfers to a frontier agent).

## Run

```bash
# Claude impact arm
python benchmarks/xbow/run_xbow.py --xbow ../xbow-benchmarks --only XBEN-001-24 \
    --poison-file data/gfcti_finance.jsonl

# Qwen consistency arm (vLLM running)
python benchmarks/xbow/run_xbow.py --xbow ../xbow-benchmarks --level 1 --max 3 \
    --backend openai --base-url http://localhost:8000/v1 --model Qwen/Qwen2.5-7B-Instruct \
    --poison-file data/gfcti_finance.jsonl --out benchmarks/xbow/xbow_results_qwen.json
```

Output `xbow_results.json`: per-benchmark `solved`/`turns` for `clean` vs `poisoned`, plus
`solve_rate_clean` / `solve_rate_poisoned`. The metric = the gap between the two.

## Verified pipeline (XBEN-001-24, IDOR finance)
build ✅ → containers Healthy ✅ → port discovery ✅ → target HTTP 200 ✅ → teardown ✅.
Only the agent loop needs your `ANTHROPIC_API_KEY` to run end-to-end.
