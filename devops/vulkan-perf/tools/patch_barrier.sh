#!/bin/bash
# A/B helper: rewrite the memory semantics of every OpControlBarrier in the
# clspv-generated SPIR-V of an app's AdaptiveCpp JIT cache (keeps a .orig backup).
# usage: patch_barrier.sh <app jit-cache dir> <old semantics> <new semantics>
#        patch_barrier.sh <dir> restore
# e.g.   patch_barrier.sh $ACPP_APPDB_DIR/apps/pathfinder-gpu-*/jit-cache 328 264
#        (0x148 AcqRel|Uniform|Workgroup -> 0x108 AcqRel|Workgroup)
# Run the app once first (with a private ACPP_APPDB_DIR) so the cache exists.
# Needs spirv-dis/spirv-as/spirv-val (Vulkan SDK).
dir=$1
tmp=$(mktemp --suffix=.spvasm)
trap 'rm -f "$tmp"' EXIT
cd "$dir" || exit 1
if [ "$2" = restore ]; then
  for f in *.orig; do [ -e "$f" ] && mv "$f" "${f%.orig}" && echo "restored ${f%.orig}"; done
  exit 0
fi
old=$2 new=$3
for f in *.jit; do
  [ "$(xxd -l4 -p "$f")" = 03022307 ] || continue
  strings "$f" | grep -q Clspv || continue
  grep -q "OpControlBarrier.*%uint_$old\$" <(spirv-dis "$f") || continue
  [ -e "$f.orig" ] || cp "$f" "$f.orig"
  spirv-dis --raw-id "$f.orig" -o "$tmp"
  # find the id of the uint type and append a new constant after the old one
  python3 - "$old" "$new" "$tmp" <<'EOF'
import re, sys
old, new, p = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(p).read()
m = re.search(r'^\s*(%\d+) = OpConstant (%\d+) ' + old + r'\s*$', s, re.M)
cid, tid = m.group(1), m.group(2)
bound = int(re.search(r'; Bound: (\d+)', s).group(1))
nid = '%' + str(bound)
s = s.replace(m.group(0), m.group(0) + '\n' + f'{nid} = OpConstant {tid} {new}', 1)
n = 0
def rep(mm):
    global n; n += 1
    return mm.group(1) + nid
s = re.sub(r'(OpControlBarrier %\d+ %\d+ )' + re.escape(cid) + r'\b', rep, s)
open(p, 'w').write(s)
print(f'patched {n} barriers')
EOF
  spirv-as --target-env spv1.5 "$tmp" -o "$f" && \
    spirv-val --target-env vulkan1.2 "$f" && echo "patched $f"
done
