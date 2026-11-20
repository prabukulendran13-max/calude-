"""Blend inpainted fill (low-res RGB8, HLG code space) into the original 10-bit YUV at full res.
Untouched pixels keep their exact original YUV values."""
import sys, argparse, numpy as np, cv2
from vio import *
ap = argparse.ArgumentParser()
ap.add_argument('v'); ap.add_argument('--fill', default='pp3'); ap.add_argument('--mask', default='mask')
ap.add_argument('--grow', type=int, default=8); ap.add_argument('--feather', type=float, default=4.0)
ap.add_argument('--detail', default=''); ap.add_argument('--membrane', type=int, default=1)  # optional full-res detail layer npy name
ap.add_argument('--out', default='final'); ap.add_argument('--start', type=int, default=0); ap.add_argument('--end', type=int, default=-1)
a = ap.parse_args()
yuv = yuv_mm(a.v); N = yuv.shape[0]
end = N if a.end < 0 else a.end
fill = np.load(f'{WORK}/{a.v}_{a.fill}.npy', mmap_mode='r')
mpk = np.load(f'{WORK}/{a.v}_{a.mask}.npy', mmap_mode='r')
det = np.load(f'{WORK}/{a.v}_{a.fill}_detail.npy', mmap_mode='r') if a.detail else None
outp = f'{WORK}/{a.v}_{a.out}.yuv'
import os
if not os.path.exists(outp) or os.path.getsize(outp) != yuv.nbytes:
    np.memmap(outp, dtype=np.uint16, mode='w+', shape=yuv.shape).flush()
out = np.memmap(outp, dtype=np.uint16, mode='r+', shape=yuv.shape)
kg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*a.grow+1,)*2)
rng = np.random.default_rng(0)
for i in range(a.start, end):
    y, u, v = split(yuv[i])
    m = np.unpackbits(mpk[i], axis=-1)[:, :W]
    if not m.any():
        out[i] = yuv[i]; continue
    m = cv2.dilate(m, kg)
    alpha = cv2.GaussianBlur(m.astype(np.float32), (0, 0), a.feather)
    alpha = np.maximum(alpha, m.astype(np.float32))  # fully replace inside the mask
    f = cv2.resize(np.asarray(fill[i]).astype(np.float32)/255, (W, H), interpolation=cv2.INTER_CUBIC)
    if det is not None: f = f + np.asarray(det[i]).astype(np.float32)/255
    if a.membrane:
        from membrane import membrane_correction
        f = f + membrane_correction(yuv2rgb(y, u, v), f, m > 0)
    Yf, Uf, Vf = rgb2yuv(f)
    Yf = Yf + rng.uniform(-0.5, 0.5, Yf.shape)  # dither 8->10 bit
    ys, xs = np.nonzero(alpha > 0)
    y0, y1, x0, x1 = ys.min(), ys.max()+1, xs.min(), xs.max()+1
    Yo = y.astype(np.float32)
    Yo[y0:y1, x0:x1] = Yo[y0:y1, x0:x1]*(1-alpha[y0:y1, x0:x1]) + Yf[y0:y1, x0:x1]*alpha[y0:y1, x0:x1]
    a2 = cv2.resize(alpha, (W//2, H//2), interpolation=cv2.INTER_AREA)
    U2 = cv2.resize(Uf, (W//2, H//2), interpolation=cv2.INTER_AREA)
    V2 = cv2.resize(Vf, (W//2, H//2), interpolation=cv2.INTER_AREA)
    Uo = u.astype(np.float32)*(1-a2) + U2*a2
    Vo = v.astype(np.float32)*(1-a2) + V2*a2
    Yn = np.where(alpha > 0, np.clip(np.round(Yo), 0, 1023), y).astype(np.uint16)
    Un = np.where(a2 > 0, np.clip(np.round(Uo), 0, 1023), u).astype(np.uint16)
    Vn = np.where(a2 > 0, np.clip(np.round(Vo), 0, 1023), v).astype(np.uint16)
    out[i] = np.concatenate([Yn.ravel(), Un.ravel(), Vn.ravel()])
out.flush()
print('composited', a.v, a.start, end)
