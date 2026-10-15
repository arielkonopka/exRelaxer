# Summary of results/positive/snake_rules.jsonl.gz: test apples per game for every
# readout x mix rule pair under each positive-only setting (see doc/research.md §32).
#   zcat results/positive/snake_rules.jsonl.gz > /tmp/p.jsonl; python3 NNtesting/tools/positive_summary.py /tmp/p.jsonl
import json,sys,collections,statistics as st
rows=collections.defaultdict(list)
for l in open(sys.argv[1]):
    d=json.loads(l)
    if d.get("type")!="trial" or d.get("error"): continue
    p=d["params"]
    cond="none" if p["positive"]=="none" else p["positive"]+":"+p["positive_kind"]
    if p["positive"]=="none" and p["positive_kind"]!="both": continue
    rows[(p["mix_er"],p["readout"],p["mix"],cond)].append(d["metrics"]["apples"])
conds=sorted({k[3] for k in rows}, key=lambda c:(c!="none",c))
for er in ("false","true"):
    print(f"\nmix_er={er}   test apples/game, mean of 10 (se)")
    print(f"{'readout':13}{'mix':13}"+"".join(f"{c:>17}" for c in conds))
    for ro in ("sign","trace","fa","perturbation"):
        for mx in ("frozen","sign","trace","fa","perturbation","oja","bcm"):
            line=f"{ro:13}{mx:13}"
            for c in conds:
                v=rows.get((er,ro,mx,c))
                if not v: line+=f"{'-':>17}"; continue
                m=st.mean(v); se=st.stdev(v)/len(v)**.5
                line+=f"{m:10.2f} ({se:4.2f})"
            print(line)
