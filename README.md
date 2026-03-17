# AI-Driven Memory Prefetching — Starter Kit

This is a Windows-friendly, trace-driven simulator for game asset caching + prefetch.
You get a working baseline, a simple heuristic (n-gram), and a tiny ML pipeline.

## Quick Start
```powershell
# inside the project folder
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# baseline
python -m src.sim.replay --trace data/traces/sample_trace.csv --cache_mb 512 --policy noprefetch
# heuristic
python -m src.sim.replay --trace data/traces/sample_trace.csv --cache_mb 512 --policy ngram --k 8 --horizon_ms 500
# ML
python -m src.ml.train --trace data/traces/sample_trace.csv --horizon_ms 500 --out eval/model.pkl
python -m src.sim.replay --trace data/traces/sample_trace.csv --cache_mb 512 --policy ml --model eval/model.pkl --k 8 --horizon_ms 500
```
