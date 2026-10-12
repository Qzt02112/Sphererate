#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE="${ROOT}/third_party/GBVS360-BMS360-ProSal"
PREFIX="${ROOT}/third_party/.envs/bms360"

if [[ ! -d "${PREFIX}/include/opencv4" ]]; then
  echo "Missing isolated BMS360 build environment: ${PREFIX}" >&2
  exit 1
fi

# The upstream makefiles hard-code Homebrew paths. Build each required static
# library explicitly so command-line include paths do not erase its local ones.
COMMON="-fPIC -O2 -W -Wall -std=c++14 -I${PREFIX}/include/opencv4 -I${PREFIX}/include"
LDFLAGS="-L${PREFIX}/lib -Wl,-rpath,${PREFIX}/lib \
-lopencv_core -lopencv_imgproc -lopencv_objdetect -lopencv_highgui \
-lopencv_imgcodecs -lopencv_videoio -lboost_program_options \
-lboost_exception -lboost_thread -lboost_system -lboost_regex -lfftw3"

make -C "${SOURCE}/lib/libgnomonic" -j"$(nproc)"
make -C "${SOURCE}/lib/libbms" -j"$(nproc)" bin/libbms.a \
  CC=g++ CFLAGS="${COMMON} -Isrc"
make -C "${SOURCE}/lib/liblinper" -j"$(nproc)" bin/liblinper.a \
  CC=g++ CFLAGS="${COMMON} -fpermissive -Ijlinkage/include -Ilsd/src"
make -C "${SOURCE}/lib/libmeanshift" -j"$(nproc)" bin/libmean_shift.a \
  CC=g++ CFLAGS="${COMMON}"
make -C "${SOURCE}/lib/libhmd" -j"$(nproc)" bin/libhmd.a \
  CC=g++ CFLAGS="${COMMON}"
make -C "${SOURCE}/lib/libgbvs" -j"$(nproc)" bin/libgbvs.a \
  CC=g++ CFLAGS="${COMMON} -I../liblinper/src -I../libmeanshift/src"

make -C "${SOURCE}" -j"$(nproc)" bin/salient \
  CC=g++ \
  CFLAGS="${COMMON} -Ilib/libgnomonic/src -Ilib/libgnomonic/lib/libinter/src \
  -Ilib/libbms/src -Ilib/libgbvs/src -Ilib/libhmd/src -Ilib/liblinper/src \
  -Ilib/libmeanshift/src -Ilibgbvs360" \
  LDFLAGS="${LDFLAGS} -Llib/libgnomonic/bin -Llib/libgnomonic/lib/libinter/bin \
  -Llib/libbms/bin -Llib/libgbvs/bin -Llib/liblinper/bin -Llib/libhmd/bin \
  -Llib/libmeanshift/bin -lgnomonic -linter -lbms -lgbvs -llinper \
  -lmean_shift -lhmd"

HELP="$(${SOURCE}/bin/salient --help 2>&1 || true)"
grep -q -- "--general-model" <<<"${HELP}"
echo "Built ${SOURCE}/bin/salient"
