#!/usr/bin/env python3
# Summarise run_tool.sh outputs: median over the measured reps of every
# "label: <number>" line, one column per device tag.
# usage: TAGS="ocl-mtl vk-mtl ..." tool_median.py <tool> [dir]
import collections, glob, os, re, statistics, sys
tool = sys.argv[1]
d = sys.argv[2] if len(sys.argv) > 2 else '/tmp/vk-perf-runs'
tags = os.environ.get('TAGS', 'ocl-mtl vk-mtl cuda-ada vk-ada ocl-cpu lvp omp').split()
rows = collections.OrderedDict()
for tag in tags:
    for f in sorted(glob.glob(f'{d}/{tool}.{tag}.[0-9]*')):
        for line in open(f):
            if 'kernel_cache' in line:
                print(f'NOTE: JIT in measured run {f}')
            if 'Warning' in line or line.startswith('device'):
                continue
            m2 = re.match(r'(.+?): submit ([0-9.]+) us/launch, end-to-end ([0-9.]+)', line)
            if m2:
                rows.setdefault(m2.group(1) + ' submit', {}).setdefault(tag, []).append(float(m2.group(2)))
                rows.setdefault(m2.group(1) + ' end-to-end', {}).setdefault(tag, []).append(float(m2.group(3)))
                continue
            m = re.match(r'(.+?):\s*([0-9.]+)', line)
            if m:
                rows.setdefault(m.group(1).strip(), {}).setdefault(tag, []).append(float(m.group(2)))
print('| metric | ' + ' | '.join(tags) + ' |')
for k, v in rows.items():
    cells = ['%.1f' % statistics.median(v[t]) if t in v else '-' for t in tags]
    print(f'| {k} | ' + ' | '.join(cells) + ' |')
