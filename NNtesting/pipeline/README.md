# pipeline

Neuron models x learning methods on one architecture, playing Doom, resumable
and profiled: `pipeline.py` (the runner), `runner-setup.sh` (venv and build of
a commit), `suites/` (what to compare). The GitHub Actions workflow
`.github/workflows/compare.yml` runs it on a self-hosted runner. Everything is
in [doc/pipeline.md](../../doc/pipeline.md).

```bash
eval "$(NNtesting/pipeline/runner-setup.sh --native | sed 's/^/export /')"
$PYTHON NNtesting/pipeline/pipeline.py run --suite smoke     # every stage in a few minutes
$PYTHON NNtesting/pipeline/pipeline.py run --suite doom      # the comparison; Ctrl-C and run again to continue
$PYTHON NNtesting/pipeline/pipeline.py report --suite doom   # ~/exr-pipeline/runs/doom/report/report.md
```
