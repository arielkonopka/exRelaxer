#!/usr/bin/env bash
# Builds the Debian packages from a finished CMake build (configured with
# -DCMAKE_INSTALL_PREFIX=/usr and -DEXRELAXER_BUILD_PYTHON=ON):
#
#   libexrelaxer-dev   static library, headers, CMake package (find_package(exrelaxer)), nntest
#   python3-exrelaxer  the exrelaxer Python package, for the system python3, and exr-nntest
#
#   packaging/build-debs.sh BUILD_DIR OUT_DIR [VERSION]
#
# VERSION defaults to the CMake project version plus the date and commit,
# e.g. 1.0.0+git20261008.223db87. Needs dpkg-deb and dpkg-shlibdeps (dpkg-dev).
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build="$(cd "${1:?build directory}" && pwd)"
mkdir -p "${2:?output directory}"
out="$(cd "$2" && pwd)"
arch="$(dpkg --print-architecture)"
project_version="$(sed -n 's/^ *VERSION \([0-9.]*\).*/\1/p' "$root/CMakeLists.txt" | head -1)"
version="${3:-${project_version}+git$(git -C "$root" log -1 --format=%cd --date=format:%Y%m%d).$(git -C "$root" rev-parse --short HEAD)}"
maintainer="Jardemirał Bantropan <neurocyp@users.noreply.github.com>"
homepage="https://github.com/arielkonopka/exRelaxer"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

# Runtime dependencies of the ELF files under $1, as a Depends value.
shlibs() {
    local files
    mapfile -t files < <(find "$1" -type f \( -name '*.so' -o -perm -u+x \) -exec sh -c 'file -b "$1" | grep -q ELF' _ {} \; -print)
    ((${#files[@]})) || return 0
    mkdir -p "$work/shlibs/debian"
    touch "$work/shlibs/debian/control"
    (cd "$work/shlibs" && dpkg-shlibdeps -O "${files[@]}" 2>/dev/null) | sed -n 's/^shlibs:Depends=//p'
}

control() {  # package dir name, description, depends
    local dir=$1 name=$2 summary=$3 depends=$4
    mkdir -p "$dir/DEBIAN"
    cat > "$dir/DEBIAN/control" <<CONTROL
Package: $name
Version: $version
Architecture: $arch
Maintainer: $maintainer
Installed-Size: $(du -sk "$dir" | cut -f1)
Depends: $depends
Section: science
Priority: optional
Homepage: $homepage
Description: $summary
 exRelaxer: biologically inspired neurons with excitation-relaxation (E-R),
 habituation and reward-modulated learning, in layers and networks.
CONTROL
    dpkg-deb --root-owner-group --build "$dir" "$out/${name}_${version}_${arch}.deb"
}

# --- libexrelaxer-dev -------------------------------------------------------
lib="$work/libexrelaxer-dev"
DESTDIR="$lib" cmake --install "$build" --prefix /usr >/dev/null
lib_deps="$(shlibs "$lib")"
control "$lib" libexrelaxer-dev "exRelaxer C++ library (static), headers, CMake package and nntest" \
        "${lib_deps:-libgomp1}, g++ (>= 4:13)"

# --- python3-exrelaxer ------------------------------------------------------
py="$work/python3-exrelaxer"
stage="$build/EXrelaxer.py/package/exrelaxer"
[[ -d "$stage" ]] || { echo "no Python package in $stage: configure with -DEXRELAXER_BUILD_PYTHON=ON" >&2; exit 1; }
python="$(grep -m1 -E '^_?Python_EXECUTABLE:' "$build/CMakeCache.txt" | cut -d= -f2)"
pyver="$("$python" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
next="$("$python" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1] + 1}")')"
mkdir -p "$py/usr/lib/python3/dist-packages" "$py/usr/bin"
cp -r "$stage" "$py/usr/lib/python3/dist-packages/"
find "$py" -name __pycache__ -prune -exec rm -rf {} +
cat > "$py/usr/bin/exr-nntest" <<'SCRIPT'
#!/usr/bin/python3
import sys
from exrelaxer.harness import main
sys.exit(main())
SCRIPT
chmod 755 "$py/usr/bin/exr-nntest"
py_deps="$(shlibs "$py")"
control "$py" python3-exrelaxer "exRelaxer Python bindings (import exrelaxer)" \
        "${py_deps:+$py_deps, }python3 (>= $pyver), python3 (<< $next), python3-numpy"

ls -l "$out"/*.deb
