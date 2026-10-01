#!/usr/bin/env bash
# One command for everything: configures and builds the library, the unit
# tests and the nntest harness (and optionally the Python package), runs the
# tests, and installs the library as a CMake package so other C++ programs
# can use it.
#
#   ./build.sh                    build, test, install into ./install
#   ./build.sh --python           ... and the Python package (needs nanobind, pytest)
#   ./build.sh --vizdoom          ... and ViZDoom from PyPI, for the doom_rl experiments
#   ./build.sh --goe              ... and Gardens of Eris from its repository, for goe_rl
#   ./build.sh --all              everything: --python --vizdoom --goe
#   ./build.sh --help             all options
#
# After it:
#   build/exrelaxer_tests                     the unit tests (already run by ctest)
#   build/NNtesting/nntest list               experiments (also install/bin/nntest)
#   NNtesting/nntest.py list                  Python experiments (with --python)
#   find_package(exrelaxer) with -DCMAKE_PREFIX_PATH=<prefix>, see examples/consumer
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="$root/build"
prefix="$root/install"
build_type=Release
jobs=""
python=OFF
vizdoom=0
goe=0
goe_dir="$root/third_party/Gardens-of-Eris"
goe_repo="${GOE_REPO:-https://github.com/arielkonopka/Gardens-of-Eris.git}"
native=OFF
run_tests=1
install=1
clean=0
extra=()

usage() {
    cat <<EOF
Usage: ./build.sh [options] [-- extra cmake configure arguments]

  --prefix DIR     install the library, headers, CMake package and nntest into DIR
                   (default: ./install)
  --build-dir DIR  build tree (default: ./build)
  --debug          Debug build (default: Release)
  --native         optimise for this machine's CPU (EXRELAXER_NATIVE)
  --python         also build the Python extension and run its tests
  --vizdoom        pip-install ViZDoom (official Farama release from PyPI) and
                   the Python build requirements (nanobind, pytest, numpy) into
                   the Python interpreter, for the doom_rl experiments; implies
                   --python. Use a virtual environment: Debian's system Python
                   refuses pip installs
  --goe            clone Gardens of Eris (GOE_REPO, default its GitHub
                   repository) into ./third_party/Gardens-of-Eris, or update
                   that clone, and pip-install its Python package goe (which
                   builds the game; needs liballegro5-dev, libopenal-dev,
                   libsndfile1-dev), for the goe_rl experiments; implies --python
  --goe-dir DIR    the Gardens of Eris checkout to use or clone into
  --all            everything: --python --vizdoom --goe
  --no-tests       build only, do not run the tests
  --no-install     do not install
  --clean          delete the build tree first
  -j, --jobs N     parallel build jobs (default: all cores)
  -h, --help       this help

Environment: PYTHON selects the Python interpreter for --python (default:
the python3 on PATH). Anything after -- goes to cmake, e.g.
  ./build.sh -- -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=/path/to/googletest
EOF
}

absolute() { [[ "$1" = /* ]] && echo "$1" || echo "$PWD/$1"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix) prefix="$(absolute "$2")"; shift 2 ;;
        --build-dir) build_dir="$(absolute "$2")"; shift 2 ;;
        --debug) build_type=Debug; shift ;;
        --native) native=ON; shift ;;
        --python) python=ON; shift ;;
        --vizdoom) vizdoom=1; python=ON; shift ;;
        --goe) goe=1; python=ON; shift ;;
        --goe-dir) goe_dir="$(absolute "$2")"; shift 2 ;;
        --all) vizdoom=1; goe=1; python=ON; shift ;;
        --no-tests) run_tests=0; shift ;;
        --no-install) install=0; shift ;;
        --clean) clean=1; shift ;;
        -j|--jobs) jobs="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        --) shift; extra=("$@"); break ;;
        *) echo "build.sh: unknown option $1 (see --help)" >&2; exit 2 ;;
    esac
done

if [[ -z "$jobs" ]]; then
    jobs="$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"
fi

step() { printf '\n==> %s\n' "$*"; }

if [[ $clean == 1 ]]; then
    step "Removing $build_dir"
    rm -rf "$build_dir"
fi

configure=(-S "$root" -B "$build_dir"
           -DCMAKE_BUILD_TYPE="$build_type"
           -DCMAKE_INSTALL_PREFIX="$prefix"
           -DEXRELAXER_BUILD_TESTS=ON
           -DEXRELAXER_BUILD_NNTESTING=ON
           -DEXRELAXER_INSTALL=ON
           -DEXRELAXER_NATIVE="$native"
           -DEXRELAXER_BUILD_PYTHON="$python")
if [[ $python == ON ]]; then
    py="${PYTHON:-$(command -v python3 || command -v python)}"
    configure+=(-DPython_EXECUTABLE="$py" -DPython3_EXECUTABLE="$py")
fi

if [[ $vizdoom == 1 ]]; then
    step "Installing ViZDoom and the Python build requirements with $py"
    "$py" -m pip install --upgrade vizdoom nanobind pytest numpy
    "$py" -c "import vizdoom; print('  vizdoom', vizdoom.__version__)"
    if ! { ldconfig -p 2>/dev/null || /sbin/ldconfig -p 2>/dev/null; } | grep -q libopenal.so.1; then
        echo "  warning: libopenal1 not found; Doom's audio buffer will be silent" >&2
        echo "           (sudo apt-get install libopenal1)" >&2
    fi
    if ! command -v ffmpeg >/dev/null; then
        echo "  warning: ffmpeg not found; watch.py needs it to write replays" >&2
        echo "           (sudo apt-get install ffmpeg)" >&2
    fi
fi

if [[ $goe == 1 ]]; then
    if [[ -d "$goe_dir/.git" ]]; then
        step "Updating Gardens of Eris in $goe_dir"
        git -C "$goe_dir" pull --ff-only || echo "  warning: could not update $goe_dir; building it as it is" >&2
    else
        step "Cloning Gardens of Eris from $goe_repo into $goe_dir"
        mkdir -p "$(dirname "$goe_dir")"
        git clone --depth 1 "$goe_repo" "$goe_dir"
    fi
    missing=()
    for lib in allegro-5 openal sndfile; do
        pkg-config --exists "$lib" 2>/dev/null || missing+=("$lib")
    done
    if [[ ${#missing[@]} -gt 0 ]]; then
        echo "  warning: pkg-config does not find ${missing[*]}; the game may not build" >&2
        echo "           (sudo apt-get install liballegro5-dev libopenal-dev libsndfile1-dev)" >&2
    fi
    step "Installing the goe package (builds the game) with $py"
    "$py" -m pip install --upgrade nanobind pytest numpy
    "$py" -m pip install "$goe_dir/agent/python"
    "$py" -c "import goe; print('  goe from', goe.__file__)"
fi

step "Configuring ($build_type) in $build_dir"
cmake "${configure[@]}" "${extra[@]}"

step "Building with $jobs jobs"
cmake --build "$build_dir" --config "$build_type" -j "$jobs"

if [[ $run_tests == 1 ]]; then
    # Unit tests, the quick experiments' checks, and (with --python) pytest.
    step "Running the tests"
    ctest --test-dir "$build_dir" -C "$build_type" -j "$jobs" --output-on-failure
fi

if [[ $install == 1 ]]; then
    step "Installing into $prefix"
    cmake --install "$build_dir" --config "$build_type" --prefix "$prefix"
fi

step "Done"
echo "  unit tests:        $build_dir/exrelaxer_tests"
echo "  experiments:       $build_dir/NNtesting/nntest list"
[[ $python == ON ]] && echo "  Python package:    PYTHONPATH=$build_dir/EXrelaxer.py/package  (NNtesting/nntest.py list)"
if [[ $vizdoom == 1 ]]; then
    echo "  Doom:              NNtesting/nntest.py run doom_rl  (doc/doom.md)"
    echo "  watch the agent:   PYTHONPATH=$build_dir/EXrelaxer.py/package $py NNtesting/experiments/doom_rl/watch.py"
    echo "                     (writes replay.mp4, .srt and .html; --live shows the game window instead)"
fi
if [[ $goe == 1 ]]; then
    echo "  Gardens of Eris:   NNtesting/nntest.py run goe_rl  (doc/goe.md; checkout in $goe_dir)"
    echo "  evolve a model:    PYTHONPATH=$build_dir/EXrelaxer.py/package $py NNtesting/experiments/goe_rl/es.py \\"
    echo "                     --out results/goe-es/er_reservoir --config NNtesting/experiments/goe_rl/models/er_reservoir.json"
fi
if [[ $install == 1 ]]; then
    echo "  library package:   $prefix  (find_package(exrelaxer) with -DCMAKE_PREFIX_PATH=$prefix;"
    echo "                     link exrelaxer::core, see examples/consumer)"
fi
