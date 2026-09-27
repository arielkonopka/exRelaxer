# Minimum architectures (nl_static)

Solved: test MSE ≤ the runs' target_mse in ≥ 80% of seeds. Learning rate: per model and architecture, lowest median validation MSE.

## Minimum size per task

| Task | Model | Solved archs | Min depth | Min width | Min neurons (arch) | Min params (arch) |
|------|-------|--------------|-----------|-----------|--------------------|-------------------|
| l0 | er | 0/24 | – | – | – | – |
| l1 | er | 0/24 | – | – | – | – |
| l2 | er | 0/24 | – | – | – | – |
| l3 | er | 0/24 | – | – | – | – |
| l4 K=1 | er | 0/24 | – | – | – | – |
| l4 K=2 | er | 0/24 | – | – | – | – |
| l4 K=4 | er | 0/24 | – | – | – | – |
| l4 K=8 | er | 0/24 | – | – | – | – |
| l4 K=16 | er | 0/24 | – | – | – | – |

## At the smallest solving architecture (fewest neurons)

| Task | Model | Arch | lr | Test MSE | Spikes/sample | Active/tick | Event synops/sample | Inference µs |
|------|-------|------|----|----------|---------------|-------------|---------------------|--------------|

## Best test MSE per task (any architecture, median over seeds)

| Task | er |
|------|---|
| l0 | 0.0116 (1×8) |
| l1 | 0.0286 (1×128) |
| l2 | 0.0219 (1×128) |
| l3 | 0.0469 (1×128) |
| l4 K=1 | 0.113 (1×32) |
| l4 K=2 | 0.175 (1×128) |
| l4 K=4 | 0.0935 (1×64) |
| l4 K=8 | 0.0517 (1×128) |
| l4 K=16 | 0.0716 (1×128) |
