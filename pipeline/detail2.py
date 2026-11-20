"""Streaming full-resolution bidirectional flow propagation of ORIGINAL pixels (whole clip),
using ProPainter's completed flows upsampled to full res. Transfers high-frequency detail into
the upscaled inpainted fill wherever the propagated real pixels agree with it."""
import sys, os, time, argparse, numpy as np, cv2
from vio import *
cv2.setNumThreads(4)
ap = argparse.ArgumentParser()
ap.add_argument('v'); ap.add_argument('--fill', default='pp3'); ap.add_argument('--mask', default='mask')
ap.add_argument('--grow', type=int, default=8); ap.add_argument('--sigma', type=float, default=1.6)
ap.add_argument('--tol', type=float, default=0.03)
a = ap.parse_args()
yuv = yuv_mm(a.v); N = yuv.shape[0]
fill = np.load(f'{WORK}/{a.v}_{a.fill}.npy', mmap_mode='r')
import json
flows = np.load(f'{WORK}/{a.v}_cflow.npy', mmap_mode='r')
_b = json.load(open(f'{WORK}/{a.v}_band.json')); BY0, BY1 = _b['y0'], _b['y1']
mpk = np.load(f'{WORK}/{a.v}_{a.mask}.npy', mmap_mode='r')
ph, pw = fill.shape[1:3]; sx, sy = W/pw, H/ph
kg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*a.grow+1,)*2)
B = np.lib.format.open_memmap(f'{WORK}/{a.v}_tmpB.npy', mode='w+', dtype=np.float16, shape=(N, H, W, 3))
BM = np.lib.format.open_memmap(f'{WORK}/{a.v}_tmpBM.npy', mode='w+', dtype=np.uint8, shape=(N, H, W))
det = np.lib.format.open_memmap(f'{WORK}/{a.v}_{a.fill}_detail.npy', mode='w+', dtype=np.int8, shape=(N, H, W, 3))
gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
alpha2 = 0.5 * sx * sy

def mask(i): return cv2.dilate(np.unpackbits(mpk[i], axis=-1)[:, :W], kg) > 0
def rgb(i): return np.clip(yuv2rgb(*split(yuv[i])), 0, 1)
def ppu(i): return cv2.resize(np.asarray(fill[i]).astype(np.float32)/255, (W, H), interpolation=cv2.INTER_CUBIC)
def upf(fl):
    full = np.zeros((2, ph, pw), np.float32); full[:, BY0:BY1] = fl
    f = np.stack([cv2.resize(full[0], (W, H)) * sx, cv2.resize(full[1], (W, H)) * sy], -1)
    return f
def step(prev, prev_missing, cur, cur_missing, fp, fc):
    mx, my = gx + fp[..., 0], gy + fp[..., 1]
    inb = (mx >= 0) & (mx <= W-1) & (my >= 0) & (my <= H-1)
    fcw = cv2.remap(fc, mx, my, cv2.INTER_LINEAR)
    diff = fp + fcw
    valid = ((diff**2).sum(-1) < 0.01*((fp**2).sum(-1) + (fcw**2).sum(-1)) + alpha2) & inb
    warped = cv2.remap(prev, mx, my, cv2.INTER_NEAREST)
    pmw = cv2.remap(prev_missing.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderValue=1) > 0.1
    use = cur_missing & valid & ~pmw
    out = np.where(use[..., None], warped, cur)
    return out, cur_missing & ~use

t0 = time.time()
prev = None
for t in range(N-1, -1, -1):
    m = mask(t)
    if not m.any() and prev is None:
        B[t] = 0; BM[t] = 0; continue
    cur = rgb(t); cur[m] = 0
    if prev is None:
        out, miss = cur, m
    else:
        out, miss = step(prev[0], prev[1], cur, m, upf(flows[0, t]), upf(flows[1, t]))
    B[t] = out.astype(np.float16); BM[t] = miss
    prev = (out, miss)
    if t % 50 == 0: print('bwd', t, round(time.time()-t0), flush=True)
prev = None
for t in range(N):
    m = mask(t)
    cur = B[t].astype(np.float32); cm = BM[t] > 0
    if prev is None:
        out, miss = cur, cm
    else:
        out, miss = step(prev[0], prev[1], cur, cm, upf(flows[1, t-1]), upf(flows[0, t-1]))
    prev = (out, miss)
    if not m.any():
        det[t] = 0; continue
    q = ppu(t)
    filled = m & ~miss
    p = np.where(filled[..., None], out, q)
    p[~m] = q[~m]
    lowp = cv2.GaussianBlur(p, (0, 0), a.sigma); lowq = cv2.GaussianBlur(q, (0, 0), a.sigma)
    d = p - lowp
    cons = np.abs(cv2.GaussianBlur(lowp - lowq, (0, 0), 3.0)).mean(-1) < a.tol
    w = (filled & cons).astype(np.float32)
    w = cv2.erode(w, np.ones((5, 5), np.uint8)); w = cv2.GaussianBlur(w, (0, 0), 2.0)
    det[t] = np.clip(np.round(d * w[..., None] * 255), -127, 127).astype(np.int8)
    if t % 50 == 0: print('fwd', t, round(time.time()-t0), 'filled', round(float(filled.sum())/max(1, m.sum()), 2), 'used', round(float((w > 0.5).sum())/max(1, m.sum()), 2), flush=True)
det.flush()
del B, BM
os.remove(f'{WORK}/{a.v}_tmpB.npy'); os.remove(f'{WORK}/{a.v}_tmpBM.npy')
print('done')
