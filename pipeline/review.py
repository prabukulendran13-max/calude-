import sys, numpy as np, pickle, cv2
from vio import *
v = sys.argv[1]; ids = [int(x) for x in sys.argv[2].split(',')]; name = sys.argv[3]
res = pickle.load(open(f'{WORK}/{v}_det.pkl', 'rb'))
tracks = pickle.load(open(f'{WORK}/{v}_tracks.pkl', 'rb'))
fr = np.load(f'{WORK}/in/{v}_rgb8.npy', mmap_mode='r')
tiles = []
for tid in ids:
    t = tracks[tid]
    f, k = max(t['items'], key=lambda fk: res[fk[0]][fk[1]]['conf'])
    d = res[f][k]; x1, y1, x2, y2 = d['box']
    cx, cy = (x1+x2)//2, (y1+y2)//2; s = max(x2-x1, y2-y1)*2 + 150
    X0, Y0 = max(0, cx-s//2), max(0, cy-s//2); X1, Y1 = min(W, X0+s), min(H, Y0+s)
    crop = np.asarray(fr[f][Y0:Y1, X0:X1]).copy()
    cv2.rectangle(crop, (x1-X0, y1-Y0), (x2-X0, y2-Y0), (255, 0, 0), 2)
    crop = cv2.resize(crop, (300, int(300*crop.shape[0]/crop.shape[1])))
    tile = np.full((330, 300, 3), 255, np.uint8); tile[:min(300,crop.shape[0])] = crop[:300]
    cv2.putText(tile, f"{tid} f{f} c{d['conf']:.2f}", (3, 322), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
    tiles.append(tile)
cols = 6
while len(tiles) % cols: tiles.append(np.full((330,300,3),255,np.uint8))
img = np.vstack([np.hstack(tiles[r*cols:(r+1)*cols]) for r in range(len(tiles)//cols)])
cv2.imwrite(f'{WORK}/preview/{name}.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
