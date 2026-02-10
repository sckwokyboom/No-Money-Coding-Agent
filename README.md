# No-Money-Coding-Agent

Simple tools to:
- download `nuprl/AgentPack` from Hugging Face into `claude.jsonl`
- filter Spring-related samples from that file
- compute statistics for filtered samples

## Requirements

- Python 3.10+
- `requests`, `pyyaml`, `datasets`

Install:

```bash
python -m pip install -U requests pyyaml datasets
```

## Input format for filtering

`filter_spring.py` expects a JSONL file (for example `data/claude.jsonl`) with fields:
- `event_id`
- `agent`
- `repo`
- `sha`
- `description`
- `patch`

One JSON object per line.

## 1) Download AgentPack -> claude.jsonl

```bash
python scripts/download_agentpack.py \
  --output data/claude.jsonl \
  --split train \
  --data-dir train
```

Optional quick test with limit:

```bash
python scripts/download_agentpack.py \
  --output data/claude.jsonl \
  --limit 5000
```

## 2) Filter Spring samples

Using config + explicit certificate:

```bash
python -m nmc_agent.benchmark.filter_spring \
  --config configs/benchmark/spring_filter.yaml \
  --ca-bundle /path/to/corp-ca.pem
```

Direct flags (explicit input/output):

```bash
python -m nmc_agent.benchmark.filter_spring \
  --input data/claude.jsonl \
  --output data/spring_only.jsonl \
  --out-all data/scored_all.jsonl \
  --progress \
  --ca-bundle /path/to/corp-ca.pem
```

`--output` is where filtered JSONL with Spring-only samples is written.

## 3) Compute stats

Using config + explicit certificate:

```bash
python -m nmc_agent.benchmark.stats_spring \
  --config configs/benchmark/spring_stats.yaml \
  --ca-bundle /path/to/corp-ca.pem
```

Direct flags:

```bash
python -m nmc_agent.benchmark.stats_spring \
  --input data/spring_only.jsonl \
  --progress \
  --ca-bundle /path/to/corp-ca.pem
```

## Progress display

Both scripts print processing progress with:
- processed/total
- percent
- rows per second
- ETA

Progress is controlled by:
- CLI: `--progress --progress-every 250`
- config: `progress.enabled`, `progress.every`
