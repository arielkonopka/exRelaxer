#!/usr/bin/env bash
# Prepares a machine (the self-hosted GitHub runner, or any Linux PC) for the comparison pipeline,
# once per commit and cheaply after that:
#   - a Python virtual environment in $EXR_PIPELINE_HOME/venv with numpy, nanobind and ViZDoom
#     (installed once; again only when this list changes);
#   - the library and its Python package built for this commit in $EXR_PIPELINE_HOME/build/<sha>
#     (skipped when that build is complete; the three newest builds are kept).
# Progress goes to stderr; it prints the variables the pipeline needs (PYTHON, PYTHONPATH) as KEY=VALUE lines, and appends
# them to $GITHUB_ENV inside GitHub Actions.
#
#   NNtesting/pipeline/runner-setup.sh [--native]      # --native: optimise for this CPU
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
home="${EXR_PIPELINE_HOME:-$HOME/exr-pipeline}"
native=OFF
[[ "${1:-}" == "--native" ]] && native=ON
mkdir -p "$home/build"

missing=()
for tool in cmake c++ python3 git; do
    command -v "$tool" > /dev/null || missing+=("$tool")
done
python3 -c "import venv, ensurepip" 2> /dev/null || missing+=("python3-venv")
python3 -c "import sysconfig, os, sys; sys.exit(0 if os.path.exists(os.path.join(sysconfig.get_paths()['include'], 'Python.h')) else 1)" \
    || missing+=("python3-dev")
if (( ${#missing[@]} )); then
    echo "runner-setup: missing ${missing[*]}; on Ubuntu: sudo apt-get install build-essential cmake python3-venv python3-dev" >&2
    exit 1
fi
ldconfig -p 2> /dev/null | grep -q libopenal.so.1 || \
    echo "runner-setup: warning: libopenal1 not found, Doom's sound will be silent (sudo apt-get install libopenal1)" >&2

# The virtual environment.
venv="$home/venv"
requirements="numpy nanobind>=2.0 vizdoom"
if [[ ! -x "$venv/bin/python" ]]; then
    python3 -m venv "$venv" >&2
fi
if [[ "$(cat "$venv/.requirements" 2> /dev/null)" != "$requirements" ]]; then
    "$venv/bin/python" -m pip install --upgrade pip > /dev/null
    # shellcheck disable=SC2086
    "$venv/bin/python" -m pip install --upgrade $requirements >&2
    echo "$requirements" > "$venv/.requirements"
fi

# The build of this commit.
sha="$(git -C "$root" rev-parse HEAD)"
dirty=""
[[ -n "$(git -C "$root" status --porcelain --untracked-files=no)" ]] && dirty="-dirty"
build="$home/build/${sha:0:12}$dirty-native$native"
if [[ -n "$dirty" || ! -f "$build/.complete" ]]; then
    cmake -S "$root" -B "$build" -DCMAKE_BUILD_TYPE=Release \
        -DEXRELAXER_BUILD_TESTS=OFF -DEXRELAXER_BUILD_NNTESTING=OFF -DEXRELAXER_INSTALL=OFF \
        -DEXRELAXER_NATIVE="$native" -DEXRELAXER_BUILD_PYTHON=ON \
        -DPython_EXECUTABLE="$venv/bin/python" -DPython3_EXECUTABLE="$venv/bin/python" > "$home/build/last-configure.log"
    cmake --build "$build" -j "$(getconf _NPROCESSORS_ONLN)" >&2
    touch "$build/.complete"
fi
touch "$build"  # in use: newest
# Keep the three newest builds.
ls -1dt "$home"/build/*/ 2> /dev/null | tail -n +4 | xargs -r rm -rf

package="$build/EXrelaxer.py/package"
PYTHONPATH="$package" "$venv/bin/python" -c "import exrelaxer, vizdoom" \
    || { echo "runner-setup: the Python package does not import" >&2; exit 1; }

vars=("EXR_PIPELINE_HOME=$home" "PYTHON=$venv/bin/python" "PYTHONPATH=$package")
printf '%s\n' "${vars[@]}"
if [[ -n "${GITHUB_ENV:-}" ]]; then
    printf '%s\n' "${vars[@]}" >> "$GITHUB_ENV"
fi
