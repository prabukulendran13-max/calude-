"""Stage C: ProPainter transformer refinement over globally propagated/LaMa-filled frames (windowed, cross-faded)."""
import sys, time, json, argparse, numpy as np, cv2, torch
from common_pp import *
from model.propainter import InpaintGenerator
ap = argparse.ArgumentParser(); ap.add_argument('v'); ap.add_argument('--win', type=int, default=80); ap.add_argument('--overlap', type=int, default=10)
ap.add_argument('--neighbor', type=int, default=10); ap.add_argument('--ref_stride', type=int, default=10)
a = ap.parse_args(); v = a.v
torch.set_num_threads(4)
masks = load_masks(v); frames8 = load_frames(v); N = len(masks)
b = json.load(open(f'{WORK}/{v}_band.json')); y0, y1 = b['y0'], b['y1']; hb = y1 - y0
cf = np.load(f'{WORK}/{v}_cflow.npy', mmap_mode='r')
G = np.load(f'{WORK}/{v}_gprop.npy', mmap_mode='r'); GU = np.load(f'{WORK}/{v}_gprop_U.npy', mmap_mode='r')
model = InpaintGenerator(model_path=f'{PP}/weights/ProPainter.pth').eval()
out = np.lib.format.open_memmap(f'{WORK}/{v}_pp3.npy', mode='w+', dtype=np.uint8, shape=(N, PH, PW, 3))
out[:] = frames8

def get_ref_index(mid, nids, length, stride, ref_num):
    ref = []
    if ref_num == -1:
        for i in range(0, length, stride):
            if i not in nids: ref.append(i)
    else:
        s = max(0, mid - stride*(ref_num//2)); e = min(length, mid + stride*(ref_num//2))
        for i in range(s, e, stride):
            if i not in nids:
                if len(ref) > ref_num: break
                ref.append(i)
    return ref

@torch.no_grad()
def window(s, e):
    t0 = time.time(); L = e - s
    mk = masks[s:e, y0:y1]
    upd = torch.from_numpy(np.asarray(G[s:e])).permute(0, 3, 1, 2).float().div(127.5).sub(1)[None]
    m_in = torch.from_numpy(mk).float()[None, :, None]
    m_up = torch.from_numpy(np.asarray(GU[s:e])).float()[None, :, None]
    pf = (torch.from_numpy(np.asarray(cf[0, s:e-1]).astype(np.float32))[None], torch.from_numpy(np.asarray(cf[1, s:e-1]).astype(np.float32))[None])
    orig = frames8[s:e, y0:y1].astype(np.float32)
    comp = [None]*L; ns = a.neighbor//2
    ref_num = -1 if L <= 80 else 80 // a.ref_stride
    for f in range(0, L, ns):
        nids = list(range(max(0, f-ns), min(L, f+ns+1)))
        rids = get_ref_index(f, nids, L, a.ref_stride, ref_num)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            pred = model(upd[:, nids+rids], (pf[0][:, nids[:-1]], pf[1][:, nids[:-1]]), m_in[:, nids+rids], m_up[:, nids+rids], len(nids))
        pred = ((pred.float().view(-1, 3, hb, PW) + 1)/2).clamp(0, 1).permute(0, 2, 3, 1).numpy()*255
        for k, idx in enumerate(nids):
            m = mk[idx][..., None]
            img = pred[k]*m + orig[idx]*(1-m)
            comp[idx] = img if comp[idx] is None else 0.5*comp[idx] + 0.5*img
    print(f'{v} refine {s}-{e} {time.time()-t0:.0f}s', flush=True)
    return np.stack(comp)

starts = []; s = 0
while True:
    e = min(N, s + a.win); starts.append((s, e))
    if e >= N: break
    s = e - a.overlap
prev = None
for wi, (s, e) in enumerate(starts):
    comp = window(s, e)
    if prev is not None:
        pe, pc = prev; ov = pe - s
        for k in range(ov):
            w = (k+1)/(ov+1); comp[k] = (1-w)*pc[k] + w*comp[k]
    last = wi == len(starts)-1
    for k in range(e - s):
        if not last and k >= e - s - a.overlap: continue
        fr = frames8[s+k].copy(); fr[y0:y1] = np.clip(comp[k]+0.5, 0, 255).astype(np.uint8); out[s+k] = fr
    prev = None if last else (e, comp[e-s-a.overlap:])
    out.flush()
print('done', v)
