"""pp4 = globally propagated/LaMa fill (sharper, real pixels) everywhere in the mask, transformer output (pp3)
only where propagation left holes (U), feathered."""
import sys, json, numpy as np, cv2
from common_pp import *
v = sys.argv[1]
b = json.load(open(f'{WORK}/{v}_band.json')); y0 = b['y0']
G = np.load(f'{WORK}/{v}_gprop.npy', mmap_mode='r'); U = np.load(f'{WORK}/{v}_gprop_U.npy', mmap_mode='r')
P3 = np.load(f'{WORK}/{v}_pp3.npy', mmap_mode='r'); N = len(P3)
out = np.lib.format.open_memmap(f'{WORK}/{v}_pp4.npy', mode='w+', dtype=np.uint8, shape=P3.shape)
for i in range(N):
    fr = np.asarray(P3[i]).copy()
    g = np.asarray(G[i]).astype(np.float32); p = fr[y0:].astype(np.float32)
    u = cv2.dilate(np.asarray(U[i]).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))).astype(np.float32)
    a = np.maximum(cv2.GaussianBlur(u, (0, 0), 2.0), np.asarray(U[i]).astype(np.float32))[..., None]
    fr[y0:] = np.clip(g*(1-a) + p*a + 0.5, 0, 255).astype(np.uint8)
    out[i] = fr
out.flush(); print('done', v)
