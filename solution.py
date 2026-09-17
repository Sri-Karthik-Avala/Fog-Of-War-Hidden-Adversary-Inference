"""made by - Karthik"""

import os
import glob
import random
import warnings
import hashlib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

H = W = 32
EPS = 1e-8
NB = 64
OUT_DIR = "working"
OUT_PATH = os.path.join(OUT_DIR, "submission.csv")
DROP_TOKENS = ["ch2", "_cx", "_cy", "_quad", "_pool", "_Ixy"]


def find_paths():
    roots = ["dataset/public", "dataset", ".", "./input", "input"]
    for r in roots:
        c = (os.path.join(r, "train.csv"), os.path.join(r, "test.csv"),
             os.path.join(r, "train_images"), os.path.join(r, "test_images"))
        if os.path.isfile(c[0]) and os.path.isfile(c[1]) and os.path.isdir(c[2]) and os.path.isdir(c[3]):
            return c
    for m in glob.glob("**/train.csv", recursive=True):
        r = os.path.dirname(m)
        c = (m, os.path.join(r, "test.csv"), os.path.join(r, "train_images"), os.path.join(r, "test_images"))
        if os.path.isfile(c[1]) and os.path.isdir(c[2]) and os.path.isdir(c[3]):
            return c
    raise FileNotFoundError("could not locate train.csv/test.csv with image folders")


def load_images(ids, folder):
    from PIL import Image
    arr = np.empty((len(ids), H, W, 3), dtype=np.uint8)
    for i, _id in enumerate(ids):
        a = np.asarray(Image.open(os.path.join(folder, str(_id) + ".png")), dtype=np.uint8)
        if a.ndim == 2:
            a = np.stack([a, a, a], axis=-1)
        arr[i] = a[:, :, :3]
    return arr


def _coords():
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    xn = ((xs + 0.5) / W * 2 - 1).ravel()
    yn = ((ys + 0.5) / H * 2 - 1).ravel()
    r = np.sqrt(xn ** 2 + yn ** 2)
    ring = np.clip(np.digitize(r, np.linspace(0, r.max() + 1e-9, 5)[1:-1]), 0, 3)
    quad = (xn < 0).astype(int) + 2 * (yn < 0).astype(int)
    return xn, yn, ring, quad


XN, YN, RING, QUAD = _coords()


def _channel_feats(I, cname):
    feats = {}

    def add(n, v):
        feats[cname + "_" + n] = v

    mass = I.sum(1)
    nz = I > 0
    area = nz.sum(1).astype(np.float64)
    ms = np.maximum(mass, EPS)
    add("mass", mass)
    add("areaf", area / (H * W))
    add("meannz", np.where(area > 0, mass / np.maximum(area, EPS), 0.0))
    add("max", I.max(1))
    Im = np.where(nz, I, np.nan)
    add("stdnz", np.where(area > 0, np.nan_to_num(np.nanstd(Im, 1)), 0.0))
    for p in [10, 25, 50, 75, 90, 95]:
        add("p%d" % p, np.where(area > 0, np.nan_to_num(np.nanpercentile(Im, p, axis=1)), 0.0))
    cx = np.where(mass > 0, (I * XN).sum(1) / ms, 0.0)
    cy = np.where(mass > 0, (I * YN).sum(1) / ms, 0.0)
    add("cx", cx)
    add("cy", cy)
    dx = XN[None] - cx[:, None]
    dy = YN[None] - cy[:, None]
    rad = np.sqrt(dx * dx + dy * dy)
    rm = (I * rad).sum(1) / ms
    add("spread", np.where(mass > 0, np.sqrt(np.maximum((I * (rad - rm[:, None]) ** 2).sum(1) / ms, 0.0)), 0.0))
    Ixx = (I * dx * dx).sum(1) / ms
    Iyy = (I * dy * dy).sum(1) / ms
    Ixy = (I * dx * dy).sum(1) / ms
    add("Ixx", Ixx)
    add("Iyy", Iyy)
    add("Ixy", Ixy)
    tr_ = Ixx + Iyy
    disc = np.sqrt(np.maximum(tr_ * tr_ / 4 - (Ixx * Iyy - Ixy * Ixy), 0.0))
    l1 = tr_ / 2 + disc
    l2 = tr_ / 2 - disc
    add("ecc", np.where(l1 > EPS, np.sqrt(np.maximum(1 - l2 / np.maximum(l1, EPS), 0.0)), 0.0))
    for rr in range(4):
        add("ring%d" % rr, np.where(mass > 0, I[:, RING == rr].sum(1) / ms, 0.0))
    for qq in range(4):
        add("quad%d" % qq, np.where(mass > 0, I[:, QUAD == qq].sum(1) / ms, 0.0))
    ent = np.zeros(I.shape[0])
    for b in range(8):
        pr = ((I > b / 8.0) & (I <= (b + 1) / 8.0)).sum(1).astype(np.float64) / np.maximum(area, EPS)
        ent -= np.where(pr > 0, pr * np.log(pr + EPS), 0.0)
    add("ent", np.where(area > 0, ent, 0.0))
    g = (I > 0).reshape(-1, H, W).astype(np.int8)
    nb = np.zeros_like(g)
    nb[:, 1:, :] += g[:, :-1, :]
    nb[:, :-1, :] += g[:, 1:, :]
    nb[:, :, 1:] += g[:, :, :-1]
    nb[:, :, :-1] += g[:, :, 1:]
    iso = ((g == 1) & (nb == 0)).reshape(I.shape[0], -1).sum(1).astype(np.float64)
    add("isofrac", iso / np.maximum(area, EPS))
    rows_used = nz.reshape(-1, H, W).any(2).sum(1).astype(np.float64)
    cols_used = nz.reshape(-1, H, W).any(1).sum(1).astype(np.float64)
    add("density", area / np.maximum(rows_used + cols_used, EPS))
    pooled = I.reshape(-1, H, W).reshape(-1, 4, 8, 4, 8).mean((2, 4)).reshape(I.shape[0], -1)
    for k in range(16):
        add("pool%d" % k, pooled[:, k])
    return feats


def build_features(imgs):
    fl = imgs.reshape(-1, H * W, 3).astype(np.float64) / 255.0
    I0, I1 = fl[:, :, 0], fl[:, :, 1]
    T = (imgs[:, :, :, 2] > 0).reshape(-1, H * W).astype(np.float64)
    feats = {}
    feats.update(_channel_feats(I0, "ch0"))
    feats.update(_channel_feats(I1, "ch1"))
    m0, m1 = I0.sum(1), I1.sum(1)
    a0, a1 = (I0 > 0).sum(1).astype(float), (I1 > 0).sum(1).astype(float)
    nz0, nz1 = I0 > 0, I1 > 0
    feats["x_logr"] = np.log((m0 + EPS) / (m1 + EPS))
    feats["x_ovl"] = (nz0 & nz1).sum(1) / (H * W)
    feats["x_vis"] = a1 / (H * W)
    am = I0.mean(1, keepdims=True)
    bm = I1.mean(1, keepdims=True)
    av = I0 - am
    bv = I1 - bm
    den = np.sqrt((av * av).sum(1) * (bv * bv).sum(1))
    feats["x_corr"] = np.where(den > EPS, (av * bv).sum(1) / np.maximum(den, EPS), 0.0)
    feats["x_mdiff"] = m0 - m1
    feats["g_nz"] = ((I0 > 0) | (I1 > 0)).sum(1) / (H * W)
    fresh0 = (I0 * (I0 > 0.66)).sum(1)
    fresh1 = (I1 * (I1 > 0.66)).sum(1)
    expl = ((I0 > 0) | (I1 > 0)).sum(1) / (H * W)
    feats["x_fresh0"] = fresh0
    feats["x_fresh1"] = fresh1
    feats["x_freshA1"] = (I1 > 0.66).sum(1) / (H * W)
    feats["x_visextrap"] = m1 / np.maximum(expl, EPS)
    feats["x_visextrap2"] = m1 / np.maximum(a0 / (H * W), EPS)
    feats["x_freshratio"] = fresh1 / np.maximum(fresh0, EPS)
    tcov = T.sum(1) / (H * W)
    a_onT = ((I1 > 0) * T).sum(1) / np.maximum(a1, EPS)
    o_onT = ((I0 > 0) * T).sum(1) / np.maximum(a0, EPS)
    feats["x_aonT"] = a_onT
    feats["x_aonT_mass"] = (I1 * T).sum(1) / np.maximum(m1, EPS)
    feats["x_oonT"] = o_onT
    feats["x_aT_minus_oT"] = a_onT - o_onT
    feats["x_aonT_vs_cover"] = a_onT - tcov
    names = sorted(feats.keys())
    X = np.stack([feats[n] for n in names], 1).astype(np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), names


def select_invariant(names):
    return [i for i, n in enumerate(names) if not any(t in n for t in DROP_TOKENS)]


def ch2_groups(imgs):
    masks = (imgs[:, :, :, 2] > 0).reshape(len(imgs), -1).astype(np.uint8)
    hsh = [hashlib.md5(r.tobytes()).hexdigest() for r in masks]
    uniq = {h: i for i, h in enumerate(sorted(set(hsh)))}
    return np.array([uniq[h] for h in hsh]), len(uniq)


def group_folds(groups, k, seed):
    ug = np.unique(groups)
    if len(ug) < k:
        rng = np.random.RandomState(seed)
        idx = np.arange(len(groups))
        rng.shuffle(idx)
        return [np.sort(idx[i::k]) for i in range(k)]
    rng = np.random.RandomState(seed)
    rng.shuffle(ug)
    assign = {g: i % k for i, g in enumerate(ug)}
    fo = np.array([assign[g] for g in groups])
    return [np.where(fo == i)[0] for i in range(k)]


def r2_log(t, p):
    return max(0.0, 1.0 - np.sum((t - p) ** 2) / (np.sum((t - t.mean()) ** 2) + EPS))


def f1_macro(y, pred):
    out = []
    for c in [0, 1]:
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        prec = tp / (tp + fp + EPS)
        rec = tp / (tp + fn + EPS)
        out.append(2 * prec * rec / (prec + rec + EPS))
    return float(np.mean(out))


def quad_kappa(y, pred, K=4):
    O = np.zeros((K, K))
    for a, b in zip(y.astype(int), pred.astype(int)):
        O[a, b] += 1
    w = np.zeros((K, K))
    for i in range(K):
        for j in range(K):
            w[i, j] = (i - j) ** 2 / ((K - 1) ** 2)
    E = np.outer(O.sum(1), O.sum(0)) / max(O.sum(), EPS)
    return 1.0 - (w * O).sum() / ((w * E).sum() + EPS)


def opt_air_threshold(prob, y):
    bt, bf = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        f = f1_macro(y, (prob >= t).astype(int))
        if f > bf:
            bf, bt = f, t
    return bt, bf


def opt_phase_cuts(cont, y):
    cand = np.unique(np.round(np.quantile(cont, np.linspace(0.08, 0.92, 13)), 6))
    best, bk = (0.5, 1.5, 2.5), -1.0
    for ai in range(len(cand)):
        for bi in range(ai + 1, len(cand)):
            for ci in range(bi + 1, len(cand)):
                a, b, c = cand[ai], cand[bi], cand[ci]
                k = quad_kappa(y, np.digitize(cont, [a, b, c]))
                if k > bk:
                    bk, best = k, (a, b, c)
    return best, bk


def _make_edges(Xtr):
    edges = []
    for j in range(Xtr.shape[1]):
        q = np.quantile(Xtr[:, j], np.linspace(0, 1, NB + 1)[1:-1])
        edges.append(np.unique(q).astype(np.float32))
    return edges


def _binit(Xa, edges):
    C = np.empty(Xa.shape, np.int16)
    for j in range(Xa.shape[1]):
        C[:, j] = np.searchsorted(edges[j], Xa[:, j], side="left")
    return np.clip(C, 0, NB - 1).astype(np.int16)


def _build_tree(C, g, h, max_depth, lam, min_leaf, feat_idx):
    n = C.shape[0]
    nid = np.zeros(n, np.int32)
    splits = {}
    leaves = {}
    cur = [0]
    node_cnt = 1
    for depth in range(max_depth):
        nn = len(cur)
        comp = np.full(n, -1, np.int32)
        for i, nd in enumerate(cur):
            comp[nid == nd] = i
        valid = comp >= 0
        Gn = np.zeros(nn)
        Hn = np.zeros(nn)
        Cn = np.zeros(nn)
        np.add.at(Gn, comp[valid], g[valid])
        np.add.at(Hn, comp[valid], h[valid])
        np.add.at(Cn, comp[valid], 1.0)
        bestgain = np.zeros(nn)
        bestfeat = np.full(nn, -1, int)
        bestbin = np.full(nn, -1, int)
        cv = comp[valid]
        gv = g[valid]
        hv = h[valid]
        for j in feat_idx:
            idx = cv * NB + C[valid, j]
            hg = np.bincount(idx, weights=gv, minlength=nn * NB).reshape(nn, NB)
            hh = np.bincount(idx, weights=hv, minlength=nn * NB).reshape(nn, NB)
            hc = np.bincount(idx, minlength=nn * NB).reshape(nn, NB).astype(np.float64)
            GL = np.cumsum(hg, 1)[:, :-1]
            HL = np.cumsum(hh, 1)[:, :-1]
            CL = np.cumsum(hc, 1)[:, :-1]
            GR = Gn[:, None] - GL
            HR = Hn[:, None] - HL
            CR = Cn[:, None] - CL
            gain = (GL * GL) / (HL + lam) + (GR * GR) / (HR + lam) - (Gn[:, None] ** 2) / (Hn[:, None] + lam)
            gain = np.where((CL >= min_leaf) & (CR >= min_leaf), gain, -1e18)
            bb = gain.argmax(1)
            bg = gain[np.arange(nn), bb]
            upd = bg > bestgain
            bestgain = np.where(upd, bg, bestgain)
            bestfeat = np.where(upd, j, bestfeat)
            bestbin = np.where(upd, bb, bestbin)
        new = []
        for i, nd in enumerate(cur):
            if bestfeat[i] >= 0 and bestgain[i] > 1e-6:
                lc, rc = node_cnt, node_cnt + 1
                node_cnt += 2
                splits[nd] = (bestfeat[i], bestbin[i], lc, rc)
                sel = comp == i
                nid[sel & (C[:, bestfeat[i]] <= bestbin[i])] = lc
                nid[sel & (C[:, bestfeat[i]] > bestbin[i])] = rc
                new += [lc, rc]
            else:
                leaves[nd] = -Gn[i] / (Hn[i] + lam)
        cur = new
        if not cur:
            break
    for nd in cur:
        sel = nid == nd
        leaves[nd] = -g[sel].sum() / (h[sel].sum() + lam) if sel.any() else 0.0
    return splits, leaves


def _predict_tree(C, splits, leaves):
    n = C.shape[0]
    nid = np.zeros(n, np.int32)
    for _ in range(24):
        moved = False
        for nd, (feat, thr, lc, rc) in splits.items():
            sel = nid == nd
            if not sel.any():
                continue
            nid[sel & (C[:, feat] <= thr)] = lc
            nid[sel & (C[:, feat] > thr)] = rc
            moved = True
        if not moved:
            break
    out = np.zeros(n)
    for nd, v in leaves.items():
        out[nid == nd] = v
    return out


class GBTNumpy:
    def __init__(self, rounds=180, lr=0.05, max_depth=5, lam=5.0, min_leaf=100, colsample=0.7, seed=0, obj="reg", delta=0.9, fair_c=1.0):
        self.rounds = rounds
        self.lr = lr
        self.max_depth = max_depth
        self.lam = lam
        self.min_leaf = min_leaf
        self.colsample = colsample
        self.seed = seed
        self.obj = obj
        self.delta = delta
        self.fair_c = fair_c

    def fit(self, X, y):
        self.edges = _make_edges(X)
        C = _binit(X, self.edges)
        self.trees = []
        rng = np.random.RandomState(self.seed)
        nf = X.shape[1]
        k = max(1, int(self.colsample * nf))
        y = y.astype(np.float64)
        if self.obj == "bin":
            p = min(max(y.mean(), 1e-3), 1 - 1e-3)
            self.base = np.log(p / (1 - p))
        else:
            self.base = y.mean()
        pred = np.full(len(y), self.base)
        for _ in range(self.rounds):
            if self.obj == "bin":
                pr = 1.0 / (1.0 + np.exp(-pred))
                g = pr - y
                h = np.maximum(pr * (1 - pr), 1e-6)
            elif self.obj == "huber":
                res = pred - y
                g = np.clip(res, -self.delta, self.delta)
                h = np.ones_like(y)
            elif self.obj == "fair":
                res = pred - y
                a = 1.0 + np.abs(res) / self.fair_c
                g = res / a
                h = 1.0 / (a * a)
            else:
                g = pred - y
                h = np.ones_like(y)
            feat = rng.choice(nf, k, replace=False)
            sp, lv = _build_tree(C, g, h, self.max_depth, self.lam, self.min_leaf, feat)
            pred += self.lr * _predict_tree(C, sp, lv)
            self.trees.append((sp, lv))
        return self

    def _raw(self, X):
        C = _binit(X, self.edges)
        out = np.full(X.shape[0], self.base)
        for sp, lv in self.trees:
            out += self.lr * _predict_tree(C, sp, lv)
        return out

    def predict(self, X):
        return self._raw(X)

    def predict_proba(self, X):
        return 1.0 / (1.0 + np.exp(-self._raw(X)))


def get_gbm():
    try:
        import lightgbm as lgb
        return "lgb", lgb
    except Exception:
        pass
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
        return "skl", (HistGradientBoostingRegressor, HistGradientBoostingClassifier)
    except Exception:
        pass
    return "numpy", None


def gbt_reg(backend, lib, Xtr, ytr, Xp, seed, objective="l2"):
    if backend == "lgb":
        kw = dict(n_estimators=900, learning_rate=0.02, num_leaves=31, subsample=0.7,
                  colsample_bytree=0.7, min_child_samples=120, reg_lambda=5.0,
                  random_state=seed, verbose=-1, n_jobs=-1)
        if objective == "huber":
            kw["objective"] = "huber"
            kw["alpha"] = 0.9
        elif objective == "fair":
            kw["objective"] = "fair"
        m = lib.LGBMRegressor(**kw)
        m.fit(Xtr, ytr)
        return m.predict(Xp)
    if backend == "skl":
        loss = "absolute_error" if objective in ("huber", "fair") else "squared_error"
        m = lib[0](loss=loss, max_iter=700, learning_rate=0.04, max_leaf_nodes=31, l2_regularization=5.0,
                   min_samples_leaf=120, random_state=seed)
        m.fit(Xtr, ytr)
        return m.predict(Xp)
    ob = {"l2": "reg", "huber": "huber", "fair": "fair"}[objective]
    return GBTNumpy(rounds=180, lr=0.05, max_depth=5, lam=5.0, min_leaf=100, colsample=0.7, seed=seed, obj=ob).fit(Xtr, ytr).predict(Xp)


def gbt_clf(backend, lib, Xtr, ytr, Xp, seed):
    spw = (ytr == 0).sum() / max(1, (ytr == 1).sum())
    if backend == "lgb":
        m = lib.LGBMClassifier(n_estimators=700, learning_rate=0.02, num_leaves=31, subsample=0.7,
                               colsample_bytree=0.7, min_child_samples=120, reg_lambda=5.0,
                               scale_pos_weight=spw, random_state=seed, verbose=-1, n_jobs=-1)
        m.fit(Xtr, ytr)
        return m.predict_proba(Xp)[:, 1]
    if backend == "skl":
        sw = np.where(ytr == 1, spw, 1.0)
        m = lib[1](max_iter=700, learning_rate=0.04, max_leaf_nodes=31, l2_regularization=5.0,
                   min_samples_leaf=120, random_state=seed)
        m.fit(Xtr, ytr, sample_weight=sw)
        return m.predict_proba(Xp)[:, 1]
    return GBTNumpy(rounds=180, lr=0.05, max_depth=5, lam=5.0, min_leaf=100, colsample=0.7, seed=seed, obj="bin").fit(Xtr, ytr).predict_proba(Xp)


def run_gbm(X, Xte, y_air, yv_log, y_phase, folds):
    backend, lib = get_gbm()
    seeds = [0, 1]
    vobjs = ["l2", "huber", "fair"]
    N, M, nf = X.shape[0], Xte.shape[0], len(folds)
    oa, ov, op = np.zeros(N), np.zeros(N), np.zeros(N)
    ta, tv, tp = np.zeros(M), np.zeros(M), np.zeros(M)
    for fi in range(nf):
        va = folds[fi]
        tr = np.concatenate([folds[j] for j in range(nf) if j != fi])
        for ob in vobjs:
            ov[va] += gbt_reg(backend, lib, X[tr], yv_log[tr], X[va], 0, ob) / len(vobjs)
            tv += gbt_reg(backend, lib, X[tr], yv_log[tr], Xte, 0, ob) / (len(vobjs) * nf)
        for sd in seeds:
            op[va] += gbt_reg(backend, lib, X[tr], y_phase[tr].astype(float), X[va], sd) / len(seeds)
            tp += gbt_reg(backend, lib, X[tr], y_phase[tr].astype(float), Xte, sd) / (len(seeds) * nf)
            oa[va] += gbt_clf(backend, lib, X[tr], y_air[tr], X[va], sd) / len(seeds)
            ta += gbt_clf(backend, lib, X[tr], y_air[tr], Xte, sd) / (len(seeds) * nf)
    return {"name": "gbt_" + backend, "oof": (oa, ov, op), "test": (ta, tv, tp)}


def run_torch(imgs, imgs_te, X, Xte, y_air, yv_log, y_phase, folds):
    try:
        import torch
        import torch.nn as nn
    except Exception:
        return None
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    epochs = 30 if dev == "cuda" else 18
    use_folds = folds if dev == "cuda" else folds[:3]
    bs = 512

    ch = np.transpose(imgs[:, :, :, :2].astype(np.float32) / 255.0, (0, 3, 1, 2))
    ch_te = np.transpose(imgs_te[:, :, :, :2].astype(np.float32) / 255.0, (0, 3, 1, 2))
    cm, cs = ch.mean((0, 2, 3), keepdims=True), ch.std((0, 2, 3), keepdims=True) + 1e-6
    ch = (ch - cm) / cs
    ch_te = (ch_te - cm) / cs
    fm, fs = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-6
    Xn = ((X - fm) / fs).astype(np.float32)
    Xten = ((Xte - fm) / fs).astype(np.float32)
    vm, vs = yv_log.mean(), yv_log.std() + 1e-6
    pm, ps = y_phase.mean(), y_phase.std() + 1e-6
    v_std = ((yv_log - vm) / vs).astype(np.float32)
    p_std = ((y_phase - pm) / ps).astype(np.float32)
    nf = X.shape[1]
    KD = 20

    class Net(nn.Module):
        def __init__(s):
            super().__init__()

            def blk(i, o):
                return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.ReLU(True),
                                     nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.ReLU(True))

            s.b1, s.b2, s.b3 = blk(2, 32), blk(32, 64), blk(64, 128)
            s.pool = nn.MaxPool2d(2)
            s.cproj = nn.Sequential(nn.Linear(256, 96), nn.ReLU(True))
            s.mlp = nn.Sequential(nn.Linear(nf, 256), nn.BatchNorm1d(256), nn.ReLU(True), nn.Dropout(0.4),
                                  nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(True), nn.Dropout(0.4))
            s.fc = nn.Sequential(nn.Linear(224, 128), nn.ReLU(True), nn.Dropout(0.3))
            s.ha, s.hv, s.hp = nn.Linear(128, 1), nn.Linear(128, 1), nn.Linear(128, 1)
            s.hvd = nn.Linear(128, KD)

        def forward(s, x, ft):
            z = s.pool(s.b1(x))
            z = s.pool(s.b2(z))
            z = s.b3(z)
            gp = torch.cat([z.mean((2, 3)), z.amax((2, 3))], 1)
            h = s.fc(torch.cat([s.mlp(ft), s.cproj(gp)], 1))
            return s.ha(h).squeeze(1), s.hv(h).squeeze(1), s.hp(h).squeeze(1), s.hvd(h)

    def d4(x):
        x = torch.rot90(x, random.randint(0, 3), [2, 3])
        if random.random() < 0.5:
            x = torch.flip(x, [2])
        if random.random() < 0.5:
            x = torch.flip(x, [3])
        return x

    N, M = X.shape[0], Xte.shape[0]
    oa, ov, op = np.zeros(N), np.zeros(N), np.zeros(N)
    ta, tv, tp = np.zeros(M), np.zeros(M), np.zeros(M)
    cov = np.zeros(N)
    ch_t = torch.tensor(ch)
    X_t = torch.tensor(Xn)
    chte_t = torch.tensor(ch_te).to(dev)
    Xte_t = torch.tensor(Xten).to(dev)
    nfold = len(use_folds)
    for fi in range(nfold):
        va = use_folds[fi]
        tr = np.concatenate([use_folds[j] for j in range(nfold) if j != fi])
        net = Net().to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=3e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        spw = torch.tensor([(y_air[tr] == 0).sum() / max(1, (y_air[tr] == 1).sum())], dtype=torch.float32).to(dev)
        bce = nn.BCEWithLogitsLoss(pos_weight=spw)
        ce = nn.CrossEntropyLoss()
        edges_v = np.quantile(yv_log[tr], np.linspace(0, 1, KD + 1))[1:-1]
        bt_np = np.digitize(yv_log[tr], edges_v)
        anchors_np = np.array([yv_log[tr][bt_np == kk].mean() if (bt_np == kk).any() else 0.0 for kk in range(KD)])
        anch_t = torch.tensor(anchors_np, dtype=torch.float32).to(dev)
        btr = torch.tensor(bt_np).long().to(dev)
        Xtr = X_t[tr].to(dev)
        Ctr = ch_t[tr].to(dev)
        atr = torch.tensor(y_air[tr].astype(np.float32)).to(dev)
        vtr = torch.tensor(v_std[tr]).to(dev)
        ptr = torch.tensor(p_std[tr]).to(dev)
        n = len(tr)
        for ep in range(epochs):
            net.train()
            perm = torch.randperm(n)
            for i in range(0, n, bs):
                b = perm[i:i + bs]
                if len(b) < 2:
                    continue
                la, lv, lp, lvd = net(d4(Ctr[b]), Xtr[b])
                loss = (0.30 * bce(la, atr[b]) + 0.40 * ((lv - vtr[b]) ** 2).mean()
                        + 0.20 * ((lp - ptr[b]) ** 2).mean() + 0.15 * ce(lvd, btr[b]))
                opt.zero_grad()
                loss.backward()
                opt.step()
            sch.step()
        net.eval()
        with torch.no_grad():
            Xva = X_t[va].to(dev)
            Cva = ch_t[va].to(dev)
            pa = torch.zeros(len(va), device=dev)
            pv = torch.zeros(len(va), device=dev)
            pp = torch.zeros(len(va), device=dev)
            pvd = torch.zeros(len(va), device=dev)
            sa = torch.zeros(M, device=dev)
            sv = torch.zeros(M, device=dev)
            sp = torch.zeros(M, device=dev)
            svd = torch.zeros(M, device=dev)
            for k in range(4):
                la, lv, lp, lvd = net(torch.rot90(Cva, k, [2, 3]), Xva)
                pa += torch.sigmoid(la)
                pv += lv
                pp += lp
                pvd += torch.softmax(lvd, 1) @ anch_t
                la2, lv2, lp2, lvd2 = net(torch.rot90(chte_t, k, [2, 3]), Xte_t)
                sa += torch.sigmoid(la2)
                sv += lv2
                sp += lp2
                svd += torch.softmax(lvd2, 1) @ anch_t
            oa[va] = (pa / 4).cpu().numpy()
            ov[va] = ((pv / 4).cpu().numpy() * vs + vm + (pvd / 4).cpu().numpy()) / 2
            op[va] = (pp / 4).cpu().numpy() * ps + pm
            cov[va] = 1.0
            ta += (sa / 4).cpu().numpy() / nfold
            tv += (((sv / 4).cpu().numpy() * vs + vm) + (svd / 4).cpu().numpy()) / 2 / nfold
            tp += ((sp / 4).cpu().numpy() * ps + pm) / nfold
    if cov.sum() < N:
        miss = cov == 0
        oa[miss] = oa[~miss].mean() if (~miss).any() else 0.5
        ov[miss] = ov[~miss].mean() if (~miss).any() else yv_log.mean()
        op[miss] = op[~miss].mean() if (~miss).any() else y_phase.mean()
    return {"name": "torch", "oof": (oa, ov, op), "test": (ta, tv, tp)}


def best_blend(metric_fn, a, b, y):
    if b is None:
        return 1.0
    if a is None:
        return 0.0
    bw, bs = 1.0, -1e9
    for w in np.linspace(0.0, 1.0, 21):
        s = metric_fn(w * a + (1 - w) * b, y)
        if s > bs:
            bs, bw = s, w
    return bw


def main():
    tr_csv, te_csv, tr_dir, te_dir = find_paths()
    tr_df = pd.read_csv(tr_csv)
    te_df = pd.read_csv(te_csv)
    train_ids = tr_df["id"].astype(str).to_numpy()
    test_ids = te_df["id"].astype(str).to_numpy()
    y_air = tr_df["target_air"].to_numpy().astype(np.int64)
    y_value = tr_df["target_value"].to_numpy().astype(np.float64)
    y_phase = tr_df["target_phase"].to_numpy().astype(np.int64)
    yv_log = np.log1p(y_value)

    imgs = load_images(train_ids, tr_dir)
    imgs_te = load_images(test_ids, te_dir)
    Xall, names = build_features(imgs)
    Xall_te, _ = build_features(imgs_te)
    cols = select_invariant(names)
    X = Xall[:, cols]
    Xte = Xall_te[:, cols]

    groups, ng = ch2_groups(imgs)
    folds = group_folds(groups, 5, SEED)
    print("features=%d  ch2_groups=%d  fold_sizes=%s" % (X.shape[1], ng, [len(f) for f in folds]))

    engines = [run_gbm(X, Xte, y_air, yv_log, y_phase, folds)]
    t = run_torch(imgs, imgs_te, X, Xte, y_air, yv_log, y_phase, folds)
    if t is not None:
        engines.append(t)

    a_oof, v_oof, p_oof = engines[0]["oof"]
    a_te, v_te, p_te = engines[0]["test"]
    if len(engines) == 2:
        b = engines[1]
        wa = best_blend(lambda pr, y: opt_air_threshold(pr, y)[1], a_oof, b["oof"][0], y_air)
        wv = best_blend(lambda pr, y: r2_log(y, pr), v_oof, b["oof"][1], yv_log)
        wp = best_blend(lambda pr, y: opt_phase_cuts(pr, y)[1], p_oof, b["oof"][2], y_phase)
        a_oof = wa * a_oof + (1 - wa) * b["oof"][0]
        v_oof = wv * v_oof + (1 - wv) * b["oof"][1]
        p_oof = wp * p_oof + (1 - wp) * b["oof"][2]
        a_te = wa * a_te + (1 - wa) * b["test"][0]
        v_te = wv * v_te + (1 - wv) * b["test"][1]
        p_te = wp * p_te + (1 - wp) * b["test"][2]
        print("blend weights  air=%.2f value=%.2f phase=%.2f" % (wa, wv, wp))

    air_t, air_f1 = opt_air_threshold(a_oof, y_air)
    phase_cuts, phase_k = opt_phase_cuts(p_oof, y_phase)
    val_r2 = r2_log(yv_log, v_oof)
    comp = 0.30 * air_f1 + 0.50 * val_r2 + 0.20 * phase_k
    print("GROUPED-OOF  air_f1=%.4f  value_r2=%.4f  phase_kappa=%.4f  composite=%.4f" % (air_f1, val_r2, phase_k, comp))
    print("engines=%s  air_thr=%.3f  phase_cuts=%s" % ([e["name"] for e in engines], air_t, np.round(phase_cuts, 3).tolist()))

    pred_air = (a_te >= air_t).astype(int)
    pred_phase = np.clip(np.digitize(p_te, list(phase_cuts)), 0, 3).astype(int)
    pred_value = np.clip(np.expm1(v_te), 0.0, float(y_value.max()) * 1.5)

    os.makedirs(OUT_DIR, exist_ok=True)
    sub = pd.DataFrame({"id": test_ids, "target_air": pred_air,
                        "target_value": pred_value, "target_phase": pred_phase})
    sub["target_air"] = sub["target_air"].astype(int)
    sub["target_phase"] = sub["target_phase"].astype(int)
    sub["target_value"] = sub["target_value"].astype(float)
    sub.to_csv(OUT_PATH, index=False)
    print("wrote %s  rows=%d  air_rate=%.3f" % (OUT_PATH, len(sub), pred_air.mean()))


if __name__ == "__main__":
    main()
