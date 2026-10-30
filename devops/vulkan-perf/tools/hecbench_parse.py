#!/usr/bin/env python3
# Extract the key metric(s) from HeCBench logs written by hecbench_suite.sh /
# hecbench_run.sh and print a markdown table with Vulkan/native time ratios
# (>1 means Vulkan slower). Column layout is M2's: three (native, Vulkan)
# pairs plus OpenMP; rename the tags with TAGS if your device tags differ.
# usage: [TAGS="ocl-mtl vk-mtl cuda-ada vk-ada ocl-cpu lvp omp"] hecbench_parse.py [dir]
import re, sys, os
D = sys.argv[1] if len(sys.argv) > 1 else '/tmp/hecbench-runs'
TAGS = os.environ.get('TAGS', 'ocl-mtl vk-mtl cuda-ada vk-ada ocl-cpu lvp omp').split()

def read(path):
    try:
        return open(path, errors='replace').read()
    except OSError:
        return None

def num(rx, txt, idx=0, scale=1.0, agg=None):
    m = re.findall(rx, txt)
    if not m:
        return None
    vals = [float(x if isinstance(x, str) else x[0]) for x in m]
    if agg == 'max':
        return max(vals) * scale
    if agg == 'sum':
        return sum(vals) * scale
    return vals[idx] * scale

# name -> (file prefix, [(metric label, extractor)])
B = [
 ('bfs (random 1M nodes, ~6 edges/node)', 'bfs/bfs', [('kernel time (ms)', lambda t: num(r'Total kernel execution time : ([0-9.]+)', t, scale=1e-3))]),
 ('bfs (1000x1000 grid, 1999 levels)', 'bfsgrid/bfs', [('kernel time (ms)', lambda t: num(r'Total kernel execution time : ([0-9.]+)', t, scale=1e-3)),
                                                         ('process wall (s)', lambda t: num(r'wall=([0-9.]+)', t))]),
 ('floydwarshall (1024 100 16)', 'floydwarshall', [('avg time/run (ms)', lambda t: num(r'Average kernel execution time ([0-9.]+)', t, scale=1e3))]),
 ('gaussian (-q -t -s 4096)', 'gaussian', [('total kernel (s)', lambda t: num(r'Total kernel execution time ([0-9.]+)', t, scale=1e-6))]),
 ('pathfinder (100000 1000 5)', 'pathfinder', [('total kernel (s)', lambda t: num(r'Total kernel execution time: ([0-9.]+)', t))]),
 ('memcpy (8192), pipelined', 'memcpy', [
     ('H2D 64 B (us)', lambda t: num(r'Copy 64 bytes from host to device takes ([0-9.]+)', t)),
     ('D2H 64 B (us)', lambda t: num(r'Copy 64 bytes from device to host takes ([0-9.]+)', t)),
     ('H2D 64 KB (us)', lambda t: num(r'Copy 65536 bytes from host to device takes ([0-9.]+)', t)),
     ('D2H 64 KB (us)', lambda t: num(r'Copy 65536 bytes from device to host takes ([0-9.]+)', t)),
     ('H2D 1 MB (us)', lambda t: num(r'Copy 1048576 bytes from host to device takes ([0-9.]+)', t)),
     ('D2H 1 MB (us)', lambda t: num(r'Copy 1048576 bytes from device to host takes ([0-9.]+)', t)),
     ('H2D 2 MB (us)', lambda t: num(r'Copy 2097152 bytes from host to device takes ([0-9.]+)', t)),
     ('D2H 2 MB (us)', lambda t: num(r'Copy 2097152 bytes from device to host takes ([0-9.]+)', t))]),
 ('black-scholes (100)', 'black-scholes', [('avg kernel (ms)', lambda t: num(r'Average kernel execution time on GPU: ([0-9.]+)', t)),
                                           ('processing time (ms)', lambda t: num(r'Processing time on GPU: ([0-9.]+)', t))]),
 ('binomial', 'binomial', [('avg kernel (ms)', lambda t: num(r'Average kernel execution time : ([0-9.]+) \(us\)', t, scale=1e-3))]),
 ('mandelbrot (1000)', 'mandelbrot', [('avg kernel (ms)', lambda t: num(r'Average kernel execution time: *([0-9.e-]+) ms', t)),
                                      ('avg parallel time (ms)', lambda t: num(r'Average parallel time: *([0-9.e-]+) ms', t))]),
 ('softmax (100000 784 1 100)', 'softmax', [('avg kernel (ms)', lambda t: num(r'Average kernel execution time: ([0-9.]+)', t))]),
 ('histogram (--i=100), first config', 'histogram', [('smem atomics (us)', lambda t: num(r'([0-9.]+) smem atomics', t)),
                                                     ('gmem atomics (us)', lambda t: num(r'([0-9.]+) gmem atomics', t))]),
 ('atomicReduction', 'atomicReduction', [('best GB/s', lambda t: num(r'reduction is ([0-9.]+) GBytes/sec', t, agg='max'))]),
 ('su3 (-i 100 -l 32 -t 128 -v 3 -w 1)', 'su3', [('GFLOP/s', lambda t: num(r'Total GFLOP/s = ([0-9.]+)', t))]),
 ('nbody (16000 30)', 'nbody', [('GFLOP/s', lambda t: num(r'# Average Performance\s*:\s*([0-9.]+)', t))]),
]

def fmt(v):
    if v is None:
        return '—'
    if v >= 100:
        return '%.0f' % v
    if v >= 10:
        return '%.1f' % v
    return '%.3g' % v

def ratio(a, b, higher_better):
    if a is None or b is None or a == 0 or b == 0:
        return '—'
    r = (b / a) if higher_better else (a / b)
    return '%.2f' % r

n1, v1, n2, v2, n3, v3, o = TAGS
print(f'| Benchmark | Metric | {n1} | {v1} | ratio | {n2} | {v2} | ratio | {n3} | {v3} | ratio | {o} | {v3}/{o} |')
print('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for name, prefix, metrics in B:
    for label, fx in metrics:
        v = {}
        for tag in TAGS:
            txt = read(os.path.join(D, f'{prefix}.{tag}.log'))
            v[tag] = fx(txt) if txt else None
        hb = 'GB/s' in label or 'GFLOP' in label
        # ratios are always "Vulkan time / native time": >1 means Vulkan slower
        row = [name, label, fmt(v[n1]), fmt(v[v1]), ratio(v[v1], v[n1], hb),
               fmt(v[n2]), fmt(v[v2]), ratio(v[v2], v[n2], hb),
               fmt(v[n3]), fmt(v[v3]), ratio(v[v3], v[n3], hb),
               fmt(v[o]), ratio(v[v3], v[o], hb)]
        print('| ' + ' | '.join(row) + ' |')
