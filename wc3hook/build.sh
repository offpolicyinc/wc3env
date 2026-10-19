#!/usr/bin/env bash
# build.bat on Linux: the same MSVC and flags, through msvc-wine's wrappers (https://github.com/mstorsjo/msvc-wine:
# vsdownload.py --accept-license --dest ~/msvc --architecture x86 x64, then install.sh ~/msvc).
#     MSVC_BIN=~/msvc/bin/x86 wc3hook/build.sh
set -euo pipefail
cd "$(dirname "$0")"
bin=${MSVC_BIN:-$HOME/msvc/bin/x86}
export WINEDEBUG=-all
mkdir -p out
json="/DYYJSON_DISABLE_WRITER=1 /DYYJSON_DISABLE_UTILS=1 /DYYJSON_DISABLE_FILE=1 /DYYJSON_DISABLE_INCR_READER=1 /DYYJSON_DISABLE_NON_STANDARD=1 /DYYJSON_READER_DEPTH_LIMIT=64"
"$bin/cl" /nologo /O2 /W3 /WX $json /LD /I minhook/include main.c allocator.c transport.c clock.c step.c observe.c \
    difficulty.c overlay.c chat.c metadata.c events.c act.c placement.c players.c setup.c instance.c diag.c rpc.c stage.c \
    features.c masks.c cuts.c yyjson/yyjson.c minhook/src/buffer.c minhook/src/hook.c minhook/src/trampoline.c \
    minhook/src/hde/hde32.c /Fo:out/ /Fe:out/wc3hook.dll /link ole32.lib user32.lib winmm.lib
"$bin/cl" /nologo /O2 /W3 /WX $json ../tests/native/test_core.c yyjson/yyjson.c /Fo:out/ /Fe:out/test_core.exe
wine out/test_core.exe
mkdir -p ../src/wc3env/native
cp out/wc3hook.dll ../src/wc3env/native/wc3hook.dll
echo "Built and tested wc3hook.dll; copied into the Python package."
