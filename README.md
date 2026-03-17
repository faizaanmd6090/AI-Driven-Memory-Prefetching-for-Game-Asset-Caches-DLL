🚀 AI-Driven Memory Prefetching for Game Asset Caches

A lightweight, trace-driven simulation framework for studying intelligent memory prefetching in game engines.
This project explores how machine learning can improve asset streaming, reduce stutters, and optimize memory usage under constraints.

🎯 Overview

Modern games dynamically load assets such as textures, models, and audio while players move through the environment.
If assets are not loaded in time, it results in frame drops, stutters, and poor user experience.

This project simulates a game asset caching system and compares three approaches:

❌ No Prefetch (Baseline) – reactive loading

🔁 N-Gram Prefetch (Heuristic) – sequence-based prediction

🧠 ML Prefetch (Adaptive) – learned prediction using logistic regression

🧩 System Architecture
Trace Generator → ML Training → Simulation Engine → Metrics Output
Components:
Module	Description
synth_trace.py	Generates realistic asset access traces
ml/train.py	Trains ML model on access patterns
sim/replay.py	Runs simulation with cache + prefetch logic
sim/cache.py	LRU cache implementation
sim/policies.py	Prefetch strategies (baseline, n-gram, ML)
⚙️ Quick Start
# setup environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
📊 Run Experiments
1️⃣ Generate Synthetic Trace
python -m src.synth_trace --out data/traces/sample_trace.csv --events 1000 --assets 300
2️⃣ Baseline (No Prefetch)
python -m src.sim.replay --trace data/traces/sample_trace.csv --cache_mb 512 --policy noprefetch
3️⃣ Heuristic (N-Gram Prefetch)
python -m src.sim.replay --trace data/traces/sample_trace.csv --cache_mb 512 --policy ngram --k 8 --horizon_ms 500
4️⃣ Machine Learning Prefetch
# train model
python -m src.ml.train --trace data/traces/sample_trace.csv --horizon_ms 500 --out eval/model.pkl

# run simulation
python -m src.sim.replay --trace data/traces/sample_trace.csv --cache_mb 512 --policy ml --model eval/model.pkl --k 8 --horizon_ms 500
🧠 How It Works
🔹 Trace Generator (synth_trace.py)

Simulates player movement across game sectors:

Each sector contains a cluster of assets

80% of accesses come from nearby assets

20% are random noise

Produces realistic access patterns at scale (10K–100K events)

🔹 Cache Model (LRUCache)

Byte-level LRU eviction

Tracks:

cache hit rate

bytes loaded

evictions

memory pressure

🔹 Prefetch Strategies
❌ No Prefetch

Loads assets only when requested
→ lowest performance baseline

🔁 N-Gram Prefetch

Learns short sequences (Markov-style)

Predicts next asset based on last N accesses

Works well for repetitive patterns

🧠 ML Prefetch

Logistic regression classifier

Predicts probability of future asset usage

Uses features like:

recent access history

hashed asset IDs

time patterns

Controlled by:

--admit_threshold (aggressiveness)

memory-aware prefetch budgeting

📈 Metrics Collected
Metric	Description
cache_hit_rate	% of requests served from cache
late_loads	assets loaded too late (stalls)
bytes_loaded	total I/O usage
bytes_evicted	cache eviction volume
wasted_io_bytes	unused prefetched data
🔬 Example Results
Policy	Hit Rate	Late Loads	Waste (%)
No Prefetch	~35%	High	0%
N-Gram	~95%	Very Low	3–4%
ML Prefetch	~88–90%	Low	<3%
⚙️ Advanced Features
🔹 Memory-Aware Prefetching

Prefetch budget adapts to cache pressure:

budget ∝ (1 - memory_pressure)

High pressure → fewer prefetches

Low pressure → more aggressive

🔹 Threshold Tuning
--admit_threshold 0.2 → aggressive (high hit rate)
--admit_threshold 0.6 → conservative (low waste)
🔹 Batch Experiments

Run multiple configurations:

Cache sizes: 32 / 64 / 128 MB

Thresholds: 0.2 / 0.4 / 0.6

Policies: noprefetch / ngram / ml

🧪 Reproducibility

Deterministic trace generation (seeded)

Stable hashing for ML features

Repeatable runs across environments

Structured output logs

🚧 Limitations

Synthetic traces (no real Unity/Unreal data yet)

Simple ML model (logistic regression)

No real I/O latency modeling

Single-threaded simulation
