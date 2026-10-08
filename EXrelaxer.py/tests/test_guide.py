"""Runs every Python block of doc/snake_growth_guide.md, in order, as one script, so the guide
(also the Python API manual) stays correct."""
import re
from pathlib import Path

GUIDE = Path(__file__).resolve().parents[2] / "doc" / "snake_growth_guide.md"


def test_guide_runs(monkeypatch):
    monkeypatch.chdir(GUIDE.parents[1])  # the guide runs from the repository root
    blocks = re.findall(r"```python\n(.*?)```", GUIDE.read_text(), re.S)
    assert len(blocks) > 10
    scope = {"__name__": "guide"}
    for i, block in enumerate(blocks):
        try:
            exec(compile(block, f"{GUIDE.name} block {i + 1}", "exec"), scope)
        except Exception as e:
            raise AssertionError(f"guide block {i + 1} failed:\n{block}") from e
