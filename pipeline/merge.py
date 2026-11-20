"""pp4 = original frame with the mask band replaced by the globally propagated + LaMa fill."""
import sys, json, numpy as np, cv2
from common_pp import *
v = sys.argv[1]
b = json.load(open(f'{WORK}/{v}_band.json')); y0 = b['y0']
G = np.load(f'{WORK}/{v}_gprop.npy', mmap_mode='r')
frames8 = load_frames(v); N = len(frames8)
out = np.lib.format.open_memmap(f'{WORK}/{v}_pp4.npy', mode='w+', dtype=np.uint8, shape=(N, PH, PW, 3))
for i in range(N):
    fr = frames8[i].copy(); fr[y0:] = G[i]; out[i] = fr
out.flush(); print('done', v)
