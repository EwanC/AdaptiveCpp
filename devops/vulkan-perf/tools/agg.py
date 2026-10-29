import re, collections, sys
txt = open('samples.txt').read()
stacks = []; cur = []
for l in txt.splitlines():
    if l.startswith('Thread '):
        if cur: stacks.append(cur)
        cur = []
    elif l.startswith('#'):
        m = re.search(r' in (.+?) \(', l) or re.search(r'^#\d+\s+(.+?) \(', l)
        cur.append(m.group(1) if m else l)
if cur: stacks.append(cur)
def idle(s):
    top = ' '.join(s[:6])
    return ('futex' in top or 'pthread_cond' in top or 'poll' in top or 'epoll' in top or 'nanosleep' in top) and 'vk_queue' not in ' '.join(s) and 'Semaphore' not in ' '.join(s)
key = sys.argv[1] if len(sys.argv) > 1 else 'main'
sel = [s for s in stacks if any(key in f for f in s)]
busy = [s for s in sel if not idle(s)]
print("stacks", len(stacks), "selected", len(sel), "non-idle", len(busy))
inc = collections.Counter()
for s in sel:
    for f in set(re.sub(r'\(.*', '', x)[:120] for x in s): inc[f] += 1
for f, c in inc.most_common(int(sys.argv[2]) if len(sys.argv) > 2 else 50):
    print("%5.1f%% %s" % (100.0 * c / max(1, len(sel)), f))
