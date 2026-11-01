"""Offline probe: do motion features (frame differences per column) help the threat detector? (research §54 addendum)"""
import sys, numpy as np
sys.path.insert(0, 'NNtesting/experiments/td_actor_critic')
import detector_probe as dp
d = np.load(sys.argv[1] if len(sys.argv) > 1 else 'colour.npz')  # from detector_probe.py --color
screens, pools, cols, gid, colours = d['screens'], d['pools'], d['cols'], d['gid'], d['colours']
s = screens.astype(np.float32) / 255.0
diff = np.zeros_like(s)
same = np.r_[False, gid[1:] == gid[:-1]]
diff[same] = np.abs(s[1:] - s[:-1])[same[1:]]
f = len(s)
# 15 bands x 10 columns of mean abs change
mot = diff.reshape(f, 15, 4, 10, 8).mean(axis=(2, 4)).transpose(0, 2, 1)  # (f, 10, 15)
# subtract the frame's mean change (global motion from turning)
mot_rel = mot - mot.mean(axis=1, keepdims=True)
def pad3(x):
    p = np.pad(x, ((0, 0), (1, 1), (0, 0)))
    return np.concatenate([p[:, :-2], p[:, 1:-1], p[:, 2:]], 2)
base = np.concatenate([dp.column_rows('pool3', screens, pools), pad3(colours)], 2)
games = np.unique(gid); cut = games[int(len(games) * 0.7)]
tr, te = gid < cut, gid >= cut
for name, x, h in (('pool3colour3+h', base, 64), ('motion3+h', pad3(np.concatenate([mot, mot_rel], 2)), 64),
                   ('pool3colour3motion3+h', np.concatenate([base, pad3(np.concatenate([mot, mot_rel], 2))], 2), 64)):
    xtr, ytr = x[tr].reshape(-1, x.shape[2]), cols[tr].ravel()
    xte, yte = x[te].reshape(-1, x.shape[2]), cols[te].ravel()
    pr = dp.fit(xtr, ytr, hidden=h, epochs=400)
    st = pr(xte); r, thr = dp.recall_at(st, yte)
    print(f'{name:24s} AUC {dp.auc(st, yte):.3f} recall at 1% false {r:.2f}', flush=True)
