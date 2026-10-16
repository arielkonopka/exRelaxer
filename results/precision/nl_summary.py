import json,sys,statistics as st,collections
rows=collections.defaultdict(list)
for l in open(sys.argv[1]):
    d=json.loads(l)
    if d.get('type')!='trial': continue
    p=d['params']; m=d['metrics']
    key=(p['task'],p['model'],p['width'],p['precision'],p.get('readout_precision','float'),p['lr'])
    rows[key].append(m)
# per (task,model,width,prec,rprec) choose lr by median validation_mse
best={}
for k,ms in rows.items():
    g=k[:5]; v=st.median(m['best_validation_mse'] for m in ms)
    if g not in best or v<best[g][0]: best[g]=(v,k[5],ms)
print("task model width precision readout lr | median test_mse [min..max] | success | spikes/sample")
for g in sorted(best):
    v,lr,ms=best[g]
    t=[m['test_mse'] for m in ms]
    print(*g, lr, "| %.2e [%.1e..%.1e] | %d/%d | %.1f" % (st.median(t),min(t),max(t),sum(m['success'] for m in ms),len(ms),st.median(m['spikes_per_sample'] for m in ms)))
