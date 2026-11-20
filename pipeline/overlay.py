import sys, numpy as np, cv2
from vio import *
v = sys.argv[1]; frames = [int(x) for x in sys.argv[2].split(',')]; name = sys.argv[3]
sc = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3
fr = np.load(f'{WORK}/in/{v}_rgb8.npy', mmap_mode='r')
mk = np.load(f'{WORK}/{v}_mask.npy', mmap_mode='r')
sh = np.load(f'{WORK}/{v}_shadow.npy', mmap_mode='r')
tiles = []
for i in frames:
    img = np.asarray(fr[i]).copy()
    m = np.unpackbits(mk[i], axis=-1)[:, :W] > 0
    s = np.unpackbits(sh[i], axis=-1)[:, :W] > 0
    img[m] = (img[m]*0.55 + np.array([255, 0, 0])*0.45).astype(np.uint8)
    img[s & m] = (img[s & m]*0.6 + np.array([0, 255, 0])*0.4).astype(np.uint8)
    t = cv2.resize(img, (int(W*sc), int(H*sc)), interpolation=cv2.INTER_AREA)
    cv2.putText(t, str(i), (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    tiles.append(t)
cols = min(len(tiles), int(sys.argv[5]) if len(sys.argv) > 5 else 6)
while len(tiles) % cols: tiles.append(np.zeros_like(tiles[0]))
img = np.vstack([np.hstack(tiles[r*cols:(r+1)*cols]) for r in range(len(tiles)//cols)])
cv2.imwrite(f'{WORK}/preview/{name}.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
