import sys, os, json, numpy as np, cv2, torch
PP = '/tmp/claude-0/-home-user-calude-/c996a4b3-e028-50ae-a262-aa818c3654dc/scratchpad/models/ProPainter'
sys.path.insert(0, PP)
from vio import *
PW, PH = 648, 1152

def load_masks(v, dil=4, name='mask'):
    mpk = np.load(f'{WORK}/{v}_{name}.npy', mmap_mode='r')
    N = mpk.shape[0]
    out = np.zeros((N, PH, PW), bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dil+1,)*2)
    for i in range(N):
        m = np.unpackbits(mpk[i], axis=-1)[:, :W].astype(np.float32)
        m = cv2.resize(m, (PW, PH), interpolation=cv2.INTER_AREA) > 0.01
        out[i] = cv2.dilate(m.astype(np.uint8), k) > 0
    return out

def band_of(masks, margin=64):
    rows = np.nonzero(masks.any(axis=(0, 2)))[0]
    y0 = max(0, (rows.min() - margin)//8*8)
    return int(y0), PH

def load_frames(v):
    fr = np.load(f'{WORK}/in/{v}_rgb8.npy', mmap_mode='r')
    return np.stack([cv2.resize(np.asarray(fr[i]), (PW, PH), interpolation=cv2.INTER_AREA) for i in range(len(fr))])
