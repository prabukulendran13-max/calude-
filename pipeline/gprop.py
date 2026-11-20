"""Stage B: whole-clip bidirectional flow propagation of real pixels + greedy LaMa keyframes
(largest fully-interior holes first), re-propagating after each round for temporal consistency."""
import sys, time, json, numpy as np, cv2, torch
from common_pp import *
cv2.setNumThreads(4); torch.set_num_threads(4)
v = sys.argv[1]
THR = int(sys.argv[2]) if len(sys.argv) > 2 else 300
GAP = int(sys.argv[3]) if len(sys.argv) > 3 else 25
MAXR = int(sys.argv[4]) if len(sys.argv) > 4 else 4
masks = load_masks(v); frames8 = load_frames(v); N = len(masks)
b = json.load(open(f'{WORK}/{v}_band.json')); y0, y1 = b['y0'], b['y1']; hb = y1 - y0
cf = np.load(f'{WORK}/{v}_cflow.npy', mmap_mode='r')
FF = np.ascontiguousarray(np.moveaxis(np.asarray(cf[0]), 1, -1)).astype(np.float32)  # (N-1, hb, PW, 2) t->t+1
FB = np.ascontiguousarray(np.moveaxis(np.asarray(cf[1]), 1, -1)).astype(np.float32)  # t+1->t
mb = masks[:, y0:y1]
import os as _os
_D = int(_os.environ.get('SRCDIL', '0'))
if _D > 0:
    _k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*_D+1, 2*_D+1))
    mb = np.stack([cv2.dilate(m.astype(np.uint8), _k) > 0 for m in mb])
    print('SRCDIL', _D, flush=True)
vals = frames8[:, y0:y1].astype(np.float32) / 255
known = ~mb
import os
if os.environ.get('FIX'):
    G0 = np.load(f'{WORK}/{v}_gprop.npy')
    bad = np.load(f'{WORK}/{v}_bad.npy')
    vals = G0.astype(np.float32) / 255
    print('FIX mode: bad px', int(bad.sum()), 'frames', int(bad.any(axis=(1, 2)).sum()), flush=True)
gx, gy = np.meshgrid(np.arange(PW, dtype=np.float32), np.arange(hb, dtype=np.float32))
if os.environ.get('FIX'):
    # extend flagged regions along the motion (+-K frames), restricted to the fill region, so ghost spans are cleared whole
    K = int(os.environ.get('FIXK', '20'))
    E = bad.copy()
    for k in range(K):
        En = E.copy()
        for t in range(N):
            if t + 1 < N and E[t+1].any():
                En[t] |= cv2.remap(E[t+1].astype(np.uint8), gx + FF[t][..., 0], gy + FF[t][..., 1], cv2.INTER_NEAREST) > 0
            if t - 1 >= 0 and E[t-1].any():
                En[t] |= cv2.remap(E[t-1].astype(np.uint8), gx + FB[t-1][..., 0], gy + FB[t-1][..., 1], cv2.INTER_NEAREST) > 0
        E = En & mb
    E = np.stack([cv2.dilate(e.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0 for e in E]) & mb
    # optionally clear whole mask components that touch flagged regions (removes complete ghost silhouettes)
    if os.environ.get('FIXCOMP'):
        for t in range(N):
            if not E[t].any(): continue
            n_, lab = cv2.connectedComponents(mb[t].astype(np.uint8), connectivity=8)
            ids = np.unique(lab[E[t] & (lab > 0)])
            comp = np.isin(lab, ids)
            # limit to a generous neighbourhood of the flagged area so big merged masks are not wiped entirely
            near = cv2.dilate(E[t].astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))) > 0
            E[t] |= comp & near
    known = ~E
    print('FIX expanded: px', int(E.sum()), 'frames', int(E.any(axis=(1, 2)).sum()), flush=True)
lama = torch.jit.load('/tmp/claude-0/-home-user-calude-/c996a4b3-e028-50ae-a262-aa818c3654dc/scratchpad/models/big-lama.pt', map_location='cpu').eval()

def step(prev, pmiss, cur, cmiss, fp, fc):
    mx, my = gx + fp[..., 0], gy + fp[..., 1]
    inb = (mx >= 0) & (mx <= PW-1) & (my >= 0) & (my <= hb-1)
    fcw = cv2.remap(fc, mx, my, cv2.INTER_LINEAR)
    diff = fp + fcw
    valid = ((diff**2).sum(-1) < 0.01*((fp**2).sum(-1) + (fcw**2).sum(-1)) + 0.5) & inb
    warped = cv2.remap(prev, mx, my, cv2.INTER_NEAREST)
    pmw = cv2.remap(pmiss.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderValue=1) > 0.1
    use = cmiss & valid & ~pmw
    return np.where(use[..., None], warped, cur), cmiss & ~use

def propagate():
    B = np.empty_like(vals); BM = np.empty_like(known)
    B[N-1] = vals[N-1]; BM[N-1] = ~known[N-1]
    for t in range(N-2, -1, -1):
        B[t], BM[t] = step(B[t+1], BM[t+1], vals[t], ~known[t], FF[t], FB[t])
    F = B; FM = BM  # in-place forward pass
    for t in range(1, N):
        if FM[t].any():
            F[t], FM[t] = step(F[t-1], FM[t-1], F[t], FM[t], FB[t-1], FF[t-1])
    return F, FM

def interior_area(U):
    n, lab = cv2.connectedComponents(U.astype(np.uint8), connectivity=8)
    if n <= 1: return 0
    border = set(np.unique(np.concatenate([lab[:, :3].ravel(), lab[:, -3:].ravel(), lab[-3:, :].ravel()]))) - {0}
    cnt = np.bincount(lab.ravel(), minlength=n); cnt[0] = 0
    for bb in border: cnt[bb] = 0
    return int(cnt.sum())

@torch.no_grad()
def lama_fill(img, hole, margin=112, minsz=256):
    """LaMa on a crop around the hole (full-frame img in [0,1]); returns img with hole filled."""
    ys, xs = np.nonzero(hole)
    if len(ys) == 0: return img
    Hh, Ww = hole.shape
    cy0, cy1 = max(0, ys.min()-margin), min(Hh, ys.max()+1+margin)
    cx0, cx1 = max(0, xs.min()-margin), min(Ww, xs.max()+1+margin)
    if cy1-cy0 < minsz: c = (cy0+cy1)//2; cy0, cy1 = max(0, c-minsz//2), min(Hh, max(0, c-minsz//2)+minsz)
    if cx1-cx0 < minsz: c = (cx0+cx1)//2; cx0, cx1 = max(0, c-minsz//2), min(Ww, max(0, c-minsz//2)+minsz)
    sub = img[cy0:cy1, cx0:cx1]; hm = hole[cy0:cy1, cx0:cx1]
    h, w = hm.shape; H8, W8 = -(-h//8)*8, -(-w//8)*8
    im = np.pad(np.transpose(sub, (2, 0, 1)), ((0, 0), (0, H8-h), (0, W8-w)), mode='symmetric')
    mk = np.pad(hm.astype(np.float32)[None], ((0, 0), (0, H8-h), (0, W8-w)), mode='symmetric')
    o = lama(torch.from_numpy(im)[None].float(), torch.from_numpy(mk)[None])[0, :, :h, :w].clamp(0, 1).permute(1, 2, 0).numpy()
    out = img.copy(); out[cy0:cy1, cx0:cx1][hm] = o[hm]
    return out

t0 = time.time(); kfs = []
kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
def apply_kf(t, P, U):
    img = frames8[t].astype(np.float32) / 255
    img[y0:y1] = P[t]
    hole = np.zeros((PH, PW), bool)
    hole[y0:y1] = (cv2.dilate(U[t].astype(np.uint8), kd) > 0) & mb[t]
    out = lama_fill(img, hole)
    hb_ = hole[y0:y1]
    newv = np.where(mb[t][..., None], P[t], vals[t])
    newv[hb_] = out[y0:y1][hb_]
    vals[t] = newv; known[t] = True

MAXB = int(sys.argv[5]) if len(sys.argv) > 5 else 12
phase = 'interior'
for rnd in range(MAXR + MAXB + 1):
    P, U = propagate()
    areas = U.reshape(N, -1).sum(1)
    if phase == 'interior':
        inter = np.array([interior_area(U[t]) if areas[t] > THR else 0 for t in range(N)])
        print(f'round {rnd} [{phase}]: {time.time()-t0:.0f}s, frames with holes: {(areas > 0).sum()}, >{THR}: {(areas > THR).sum()}, interior>{THR}: {(inter > THR).sum()}, total px {areas.sum()}', flush=True)
        if rnd >= MAXR or (inter > THR).sum() < 8:
            phase = 'border'
        else:
            score = inter * (inter > THR)
    if phase == 'border':
        print(f'round {rnd} [{phase}]: {time.time()-t0:.0f}s, frames with holes: {(areas > 0).sum()}, >50: {(areas > 50).sum()}, total px {areas.sum()}', flush=True)
        if (areas > 50).sum() == 0 or rnd >= MAXR + MAXB: break
        score = areas * (areas > 50)
    picks = []
    for t in np.argsort(-score):
        if score[t] <= 0: break
        if all(abs(int(t) - p) >= GAP for p in picks): picks.append(int(t))
    for t in picks:
        apply_kf(t, P, U); kfs.append((rnd, t, int(U[t].sum())))
    print(f'  lama on {len(picks)} keyframes {picks[:10]}... {time.time()-t0:.0f}s', flush=True)
# final cleanup: any remaining holes, per frame (no re-propagation)
rem = [t for t in range(N) if U[t].any()]
for t in rem:
    img = frames8[t].astype(np.float32) / 255; img[y0:y1] = P[t]
    hole = np.zeros((PH, PW), bool); hole[y0:y1] = (cv2.dilate(U[t].astype(np.uint8), kd) > 0) & mb[t]
    P[t] = lama_fill(img, hole)[y0:y1]; U[t] = False
print(f'final per-frame cleanup on {len(rem)} frames, {time.time()-t0:.0f}s', flush=True)
np.save(f'{WORK}/{v}_gprop.npy', np.clip(P*255+0.5, 0, 255).astype(np.uint8))
np.save(f'{WORK}/{v}_gprop_U.npy', U)
json.dump(kfs, open(f'{WORK}/{v}_gprop_kf.json', 'w'))
print('done', v, len(kfs), 'keyframes', round(time.time()-t0))
