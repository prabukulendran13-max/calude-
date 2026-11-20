import sys, numpy as np, pickle, cv2
from vio import *

def inst_mask(d):
    m = np.unpackbits(d['mask'])[:d['mshape'][0]*d['mshape'][1]].reshape(d['mshape']).astype(bool)
    return m

def iou(a, b):
    x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/max(ua,1)

def build_tracks(res, cls=(0,), min_conf=0.1, gap=6, thr=0.25):
    tracks = []  # each: dict(id, items=[(frame, idx)], last_box, last_f)
    for i, inst in enumerate(res):
        cand = [(k, d) for k, d in enumerate(inst) if d['cls'] in cls and d['conf'] >= min_conf]
        cand.sort(key=lambda kd: -kd[1]['conf'])
        used = set()
        for k, d in cand:
            best, bs = None, thr
            for t in tracks:
                if i - t['last_f'] > gap or t['last_f'] == i: continue
                s = iou(d['box'], t['last_box'])
                if s > bs: best, bs = t, s
            if best is None:
                best = dict(id=len(tracks), items=[], last_box=None, last_f=-99); tracks.append(best)
            best['items'].append((i, k)); best['last_box'] = d['box']; best['last_f'] = i
    for t in tracks:
        confs = [res[f][k]['conf'] for f, k in t['items']]
        t['maxc'] = max(confs); t['meanc'] = float(np.mean(confs)); t['len'] = len(confs)
        t['f0'] = t['items'][0][0]; t['f1'] = t['items'][-1][0]
    return tracks

if __name__ == '__main__':
    v = sys.argv[1]
    res = pickle.load(open(f'{WORK}/{v}_det.pkl', 'rb'))
    fr = np.load(f'{WORK}/in/{v}_rgb8.npy', mmap_mode='r')
    tracks = build_tracks(res)
    pickle.dump(tracks, open(f'{WORK}/{v}_tracks.pkl', 'wb'))
    print(len(tracks), 'tracks')
    # contact sheet of track crops (best conf frame)
    tiles = []
    import os
    mc = float(os.environ.get('MINC', '0'))
    for t in tracks:
        if t['maxc'] < mc: continue
        f, k = max(t['items'], key=lambda fk: res[fk[0]][fk[1]]['conf'])
        d = res[f][k]; x1, y1, x2, y2 = d['box']
        pad = 10
        crop = np.asarray(fr[f][max(0,y1-pad):y2+pad, max(0,x1-pad):x2+pad])
        crop = crop.copy()
        m = inst_mask(d)
        # tint mask
        mm = np.zeros(crop.shape[:2], bool)
        oy, ox = d['mbox'][1]-max(0,y1-pad), d['mbox'][0]-max(0,x1-pad)
        hh, ww = m.shape
        sub = mm[max(0,oy):oy+hh, max(0,ox):ox+ww]
        sub |= m[:sub.shape[0], :sub.shape[1]]
        crop[mm] = (crop[mm]*0.6 + np.array([255,0,0])*0.4).astype(np.uint8)
        s = 160/max(crop.shape[:2]); crop = cv2.resize(crop, (max(1,int(crop.shape[1]*s)), max(1,int(crop.shape[0]*s))))
        tile = np.full((200, 160, 3), 255, np.uint8)
        tile[:crop.shape[0], :crop.shape[1]] = crop
        cv2.putText(tile, f"{t['id']} c{t['maxc']:.2f}", (2, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1)
        cv2.putText(tile, f"f{t['f0']}-{t['f1']} n{t['len']}", (2, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1)
        tiles.append(tile)
    cols = 12
    while len(tiles) % cols: tiles.append(np.full((200,160,3),255,np.uint8))
    rows = [np.hstack(tiles[r*cols:(r+1)*cols]) for r in range(len(tiles)//cols)]
    for p in range(0, len(rows), 5):
        cv2.imwrite(f'{WORK}/preview/{v}_tracks_{p//5}.jpg', cv2.cvtColor(np.vstack(rows[p:p+5]), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
