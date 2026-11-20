import sys, json, numpy as np, pickle, cv2
from vio import *
from tracks import inst_mask, iou

ACC = {24, 25, 26, 28, 67}

def paste(full, d, m=None):
    if m is None: m = inst_mask(d)
    x0, y0, x1, y1 = d['mbox']
    full[y0:y1, x0:x1] |= m[:y1-y0, :x1-x0]

def person_masks(v, drop_tracks, min_track_conf=0.35, extra=None):
    """returns list of per-frame bool masks (people+accessories), plus per-frame kept person boxes"""
    res = pickle.load(open(f'{WORK}/{v}_det.pkl', 'rb'))
    tracks = pickle.load(open(f'{WORK}/{v}_tracks.pkl', 'rb'))
    n = len(res)
    keep = {}  # (f,k) -> track id
    for t in tracks:
        if t['id'] in drop_tracks: continue
        if t['maxc'] < min_track_conf: continue
        if t['len'] < 2 and t['maxc'] < 0.5: continue
        for f, k in t['items']: keep[(f, k)] = t['id']
    masks = [np.zeros((H, W), bool) for _ in range(n)]
    boxes = [[] for _ in range(n)]
    for (f, k), tid in keep.items():
        paste(masks[f], res[f][k]); boxes[f].append(res[f][k]['box'])
    # gap fill within tracks
    for t in tracks:
        if t['id'] in drop_tracks or t['maxc'] < min_track_conf: continue
        it = t['items']
        for (fa, ka), (fb, kb) in zip(it[:-1], it[1:]):
            if fb - fa > 1:
                for f in range(fa+1, fb):
                    paste(masks[f], res[fa][ka]); paste(masks[f], res[fb][kb])
                    boxes[f].append(res[fa][ka]['box'])
    # accessories touching people
    k40 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    for f in range(n):
        near = None
        for d in res[f]:
            if d['cls'] in ACC and d['conf'] >= 0.15:
                if near is None: near = cv2.dilate(masks[f].astype(np.uint8), k40) > 0
                m = np.zeros((H, W), bool); paste(m, d)
                if (m & near).sum() > 0.2*m.sum():
                    masks[f] |= m
    return masks, boxes

def flow_warp_mask(prev_img, cur_img, m_prev, dis):
    """warp mask from prev frame into cur frame using DIS flow at 1/4 res"""
    s = 4
    a = cv2.cvtColor(cv2.resize(cur_img, (W//s, H//s), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    b = cv2.cvtColor(cv2.resize(prev_img, (W//s, H//s), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    fl = dis.calc(a, b, None)  # cur -> prev
    fl = cv2.resize(fl, (W, H)) * s
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    return cv2.remap(m_prev.astype(np.uint8), gx+fl[..., 0], gy+fl[..., 1], cv2.INTER_NEAREST) > 0

def dis_flow(dis, cur_img, other_img, s=4):
    a = cv2.cvtColor(cv2.resize(cur_img, (W//s, H//s), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    b = cv2.cvtColor(cv2.resize(other_img, (W//s, H//s), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    return dis.calc(a, b, None)  # cur -> other, at 1/s res

def warp_small(m, fl):
    h, w = fl.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(m.astype(np.uint8), gx+fl[..., 0], gy+fl[..., 1], cv2.INTER_NEAREST) > 0

def warp_full(m, fl, s=4):
    f = cv2.resize(fl, (W, H)) * s
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    return cv2.remap(m.astype(np.uint8), gx+f[..., 0], gy+f[..., 1], cv2.INTER_NEAREST) > 0

if __name__ == '__main__':
    from shadow import shadow_mask
    v = sys.argv[1]
    cfg = json.load(open(f'{WORK}/{v}_maskcfg.json'))
    masks, boxes = person_masks(v, set(cfg.get('drop', [])), cfg.get('min_track_conf', 0.35))
    fr = np.load(f'{WORK}/in/{v}_rgb8.npy', mmap_mode='r')
    n = len(masks)
    for f, poly in cfg.get('extra_polys', []):   # manual additions [frame, [[x,y],...]]
        cv2.fillPoly(masks[f].view(np.uint8), [np.array(poly, np.int32)], 1)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    out = np.lib.format.open_memmap(f'{WORK}/{v}_mask.npy', mode='w+', dtype=np.uint8, shape=(n, H, (W+7)//8))
    shp = np.lib.format.open_memmap(f'{WORK}/{v}_shadow.npy', mode='w+', dtype=np.uint8, shape=(n, H, (W+7)//8))
    kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*cfg.get('dil', 8)+1,)*2)
    prev_sh4 = None; last_seed = -999
    sc = cfg.get('shadow', {})
    for f in range(n):
        cur = np.asarray(fr[f])
        m = masks[f].copy()
        fl_prev = dis_flow(dis, cur, np.asarray(fr[f-1])) if f > 0 else None
        if f > 0 and masks[f-1].any(): m |= warp_full(masks[f-1], fl_prev)
        if f < n-1 and masks[f+1].any(): m |= warp_full(masks[f+1], dis_flow(dis, cur, np.asarray(fr[f+1])))
        sh = np.zeros((H, W), bool)
        if sc.get('on', True) and sc.get('f0', 0) <= f < sc.get('f1', 10**6):
            da = sc.get('dark_attach', [0, 0, 0])
            sh, cand, _ = shadow_mask(cur, m, boxes[f], ratio=sc.get('ratio', 0.72), dblue=sc.get('dblue', 0.012),
                                      reach=sc.get('reach', 1.2), up=sc.get('up', 0.12), env_rel=sc.get('env_rel', 0.75), ret_dbg=True,
                                      dir_deg=sc.get('dir'), cone=sc.get('cone', 75),
                                      dark_attach_h=da[2] if da[0] <= f < da[1] else 0)
            sh4 = cv2.resize(sh.astype(np.uint8), (W//4, H//4), interpolation=cv2.INTER_AREA) > 0
            if sh4.sum() > 20: last_seed = f
            if sc.get('track', True) and prev_sh4 is not None and prev_sh4.any() and f - last_seed <= sc.get('track_len', 40):
                wp = warp_small(prev_sh4, fl_prev)
                trk = wp & (cv2.dilate(cand, np.ones((3, 3), np.uint8)) > 0) & ~sh4
                cap = 1.05*prev_sh4.sum() - sh4.sum()
                if trk.sum() > 0 and trk.sum() <= max(cap, 0):
                    sh4 = sh4 | trk
                    sh |= cv2.resize(cv2.dilate(trk.astype(np.uint8), np.ones((3,3),np.uint8)), (W, H), interpolation=cv2.INTER_LINEAR) > 0
            prev_sh4 = sh4
        shp[f] = np.packbits(sh, axis=-1)
        m = cv2.dilate((m | sh).astype(np.uint8), kd) > 0
        # extra dilation around legs/feet of large (near-camera) people: contact shadows, motion blur
        for (bx1, by1, bx2, by2) in boxes[f]:
            bh = by2 - by1
            if bh < cfg.get('feet_min_h', 250): continue
            r = int(cfg.get('feet_r', 0.03) * bh)
            if r < 2: continue
            fy0 = int(by1 + 0.7*bh); X0, X1 = max(0, bx1 - 2*r), min(W, bx2 + 2*r); Y0, Y1 = max(0, fy0 - r), min(H, by2 + 2*r)
            sub = m[Y0:Y1, X0:X1].copy()
            sub[:max(0, fy0 - Y0)] = False
            grown = cv2.dilate(sub.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r+1,)*2)) > 0
            m[Y0:Y1, X0:X1] |= grown
        out[f] = np.packbits(m, axis=-1)
        if f % 50 == 0: print(v, f, flush=True)
    out.flush(); shp.flush()
    pickle.dump(boxes, open(f'{WORK}/{v}_boxes.pkl', 'wb'))
    print('done', v)
