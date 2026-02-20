# forge-world

Generic benchmark-modify-benchmark harness for pipeline optimization. Define six protocols, get regression detection, caching, diff reporting, and an autonomous evolution loop for free.

## Install

```bash
pip install -e ".[dev]"
```

Requires Python 3.11+.

## Quick Start

1. Create a module that implements the forge-world protocols (see [Protocols](#protocols))
2. Add a `forge-world.toml`:
   ```toml
   [forge-world]
   module = "your_project.forge_world"
   ```
3. Run:
   ```bash
   forge bench --single-seed 42          # benchmark
   forge lock --single-seed 42           # save baseline
   # ... make changes ...
   forge check --single-seed 42          # detect regressions
   forge diff --single-seed 42           # compact delta
   ```

## Protocols

Your project module must export these factory functions:

| Factory | Returns Protocol | Purpose |
|---------|-----------------|---------|
| `create_pipeline()` | `Pipeline` | Analyzes items, produces `Finding`s |
| `create_aggregator()` | `Aggregator` | Combines findings into `AggregatedResult` |
| `create_dataset()` | `LabeledDataset` | Provides labeled test items |
| `create_rules()` | `PassFailRuleSet` | Determines pass/fail per expected label |

Optional:

| Factory | Returns Protocol | Purpose |
|---------|-----------------|---------|
| `create_fitness()` | `FitnessFunction` | Evaluates overall benchmark quality |
| `create_diagnostics()` | `Diagnostics` | Domain-specific failure clustering |

Optional protocol on dataset:

| Method | Protocol | Purpose |
|--------|----------|---------|
| `tiers()` | `TieredDataset` | Maps tier names to category lists for fast subsets |

## CLI Commands

### `forge bench`

Run the benchmark pipeline against the dataset.

```bash
forge bench                              # multi-seed (default)
forge bench --single-seed 42             # single seed
forge bench --tier smoke                 # only smoke-tier categories
forge bench --only-failures              # re-run only baseline failures
forge bench --only-failures --then-full  # failures first, then full suite
forge bench --sample-size 50             # limit seeded-random items per seed
forge bench --json-output                # machine-readable output
```

### `forge lock`

Lock current outputs as a baseline snapshot (saved to `.forge-world/snapshots/`, git-committable).

```bash
forge lock                      # lock as "baseline"
forge lock --name v2            # lock with custom name
```

### `forge check`

Run benchmark and compare against a locked baseline. Detects regressions, improvements, false positive changes, and multi-seed instability. Exits non-zero on new false positives.

```bash
forge check
forge check --baseline v2
forge check --only-failures     # check just previously-failed items
```

### `forge diff`

Compact delta report between current run and baseline. Shows new failures/passes, risk level shifts (without pass/fail flip), boundary proximity changes, and unstable items.

```bash
forge diff
forge diff --json-output
```

### `forge analyze`

Print structured context optimized for AI agent consumption. Includes failure analysis, method effectiveness, near-misses, config schema, and regression data.

```bash
forge analyze --single-seed 42
forge analyze --baseline baseline --json-output
```

### `forge cache clear`

Manage the persistent disk cache (`.forge-world/cache/`). The cache stores `pipeline.analyze()` results keyed by `config_hash + item_id`, avoiding re-analysis of unchanged items.

```bash
forge cache clear                        # clear all
forge cache clear --config-hash abc123   # clear specific config
```

### `forge evolve`

Autonomous bench-modify-bench loop. Iteratively: benchmarks, builds context, invokes an AI agent, validates constraints, and accepts/rejects changes via git.

```bash
forge evolve --agent-command "claude -p 'Read {context_file} and improve the pipeline'"
forge evolve --max-iterations 10 --patience 3
forge evolve --tier smoke --single-seed 42
```

Configure defaults in `forge-world.toml`:

```toml
[forge-world.evolve]
agent_command = "claude -p 'Read {context_file} and improve the pipeline'"
max_iterations = 10
hard_constraints = [{metric = "fpr", op = "<=", value = 0}]
```

The loop enforces hard constraints (e.g. FPR=0) and optimizes a target metric (default: sensitivity). Changes that violate constraints or don't improve the target are rolled back via `git checkout`.

## Global Options

```bash
forge --module your.module ...   # override module discovery
forge --no-cache ...             # disable disk cache
forge --cache-dir /tmp/cache ... # custom cache location
forge --quiet ...                # suppress per-item output
```

## Multi-Seed Evaluation

By default, commands run multiple seeds to detect overfitting:

```bash
forge bench --seeds 42,137 --exploration-seeds 2
```

- **Stable seeds**: fixed seeds for reproducible regression detection
- **Exploration seeds**: random seeds to test generalization

Use `--single-seed N` for fast single-seed runs during iteration.

## Architecture

```
forge_world/
  cli/main.py          CLI entry point (Click)
  core/
    protocols.py        Protocol definitions + data types
    runner.py           BenchmarkRunner (single + multi-seed)
    metrics.py          Confusion matrix, near-misses, failure clusters
    snapshots.py        Snapshot locking + regression detection
    agent_interface.py  AI agent context builder
    cache.py            Persistent disk cache
    diff.py             Compact delta reporting
    evolve.py           Autonomous evolution loop
```

## Configuration

`forge-world.toml` in the project root:

```toml
[forge-world]
module = "your_project.forge_world"
sample_size = 50                          # default seeded-random items per seed

[forge-world.evolve]
agent_command = "claude -p 'Read {context_file} and improve'"
max_iterations = 10
hard_constraints = [{metric = "fpr", op = "<=", value = 0}]
optimization_target = {metric = "sensitivity", direction = "max"}
```

## File Layout

| Path | Committed? | Purpose |
|------|-----------|---------|
| `.forge-world/snapshots/` | Yes | Locked baselines for regression detection |
| `.forge-world/cache/` | No | Disk cache for analysis results |
| `.forge-world/evolution-context.md` | No | Transient context file for `forge evolve` |
| `forge-world.toml` | Yes | Project configuration |
