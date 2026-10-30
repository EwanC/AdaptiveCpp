#!/usr/bin/env python3
# Generate a graph in the Rodinia BFS text format used by HeCBench bfs-sycl.
#   random: like Rodinia graphgen / graph1MW_6 - each node draws 2-4 random
#           neighbours, edges are made undirected (mean degree ~6), few BFS levels
#   grid:   side x side 4-neighbour grid; BFS from node 0 takes 2*side-1 levels,
#           stressing per-iteration H2D/D2H flag copies and launch latency
# usage: gen_bfs_graph.py random <nodes> <out>  |  gen_bfs_graph.py grid <side> <out>
import random, sys

mode, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
random.seed(1234)
if mode == 'random':
    adj = [[] for _ in range(n)]
    for i in range(n):
        for _ in range(random.randint(2, 4)):
            j = random.randrange(n)
            w = random.randint(1, 10)
            adj[i].append((j, w))
            adj[j].append((i, w))
elif mode == 'grid':
    side, n = n, n * n
    adj = [[] for _ in range(n)]
    for y in range(side):
        for x in range(side):
            i = y * side + x
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if 0 <= x + dx < side and 0 <= y + dy < side:
                    adj[i].append(((y + dy) * side + x + dx, 1))
else:
    sys.exit('mode must be random or grid')

with open(out, 'w') as f:
    f.write(f'{n}\n')
    start = 0
    for a in adj:
        f.write(f'{start} {len(a)}\n')
        start += len(a)
    f.write('\n0\n\n')
    f.write(f'{start}\n')
    for a in adj:
        for j, w in a:
            f.write(f'{j} {w}\n')
