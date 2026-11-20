import numpy as np, cv2
from vio import W, H
S = 4
w4, h4 = W//S, H//S

def shadow_mask(img, P, boxes, ratio=0.72, dblue=0.012, reach=1.2, up=0.12, env_rel=0.75, ret_dbg=False,
                dir_deg=None, cone=75, dark_attach_h=0):
    """Shadows cast by people onto bright sunlit ground.
    img: HxWx3 uint8, P: HxW bool person mask, boxes: person boxes (full res)."""
    sm = cv2.resize(img, (w4, h4), interpolation=cv2.INTER_AREA).astype(np.float32)
    p4 = cv2.resize(P.astype(np.uint8), (w4, h4), interpolation=cv2.INTER_AREA) > 0
    p4d = cv2.dilate(p4.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    L = sm.mean(-1)
    b = sm[..., 2] / np.maximum(sm.sum(-1), 1)
    Lf = cv2.inpaint(np.clip(L, 0, 255).astype(np.uint8), p4d.astype(np.uint8), 5, cv2.INPAINT_TELEA).astype(np.float32)
    env = cv2.dilate(cv2.blur(Lf, (3, 3)), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))
    env = cv2.GaussianBlur(env, (0, 0), 8)
    lit = (L > 0.85*env) & ~p4d
    wl = cv2.GaussianBlur(lit.astype(np.float32), (0, 0), 15)
    bl = cv2.GaussianBlur(b*lit, (0, 0), 15) / np.maximum(wl, 1e-3)
    # bright ground reference: lit level in lower half of frame
    ground = np.percentile(env[h4//2:], 90)
    cand = (L < ratio*env) & (b > bl + dblue) & ~p4 & (env > env_rel*ground)
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    ncc, lab = cv2.connectedComponents(cand | p4d.astype(np.uint8), connectivity=8)
    out = np.zeros((h4, w4), bool)
    yy, xx = np.mgrid[0:h4, 0:w4]
    for (x1, y1, x2, y2) in boxes:
        x1, y1, x2, y2 = [c/S for c in (x1, y1, x2, y2)]
        hb = y2 - y1; wb = x2 - x1
        if hb < 8: continue
        fy0, fy1 = int(max(0, y2 - 0.1*hb)), int(min(h4, y2 + 0.06*hb + 2))
        fx0, fx1 = int(max(0, x1 - 0.1*wb)), int(min(w4, x2 + 0.1*wb))
        if fy1 <= fy0 or fx1 <= fx0: continue
        # feet must stand on bright ground
        ring = env[fy0:fy1, fx0:fx1]
        if np.median(ring) < env_rel*ground: continue
        seed = lab[fy0:fy1, fx0:fx1]
        ids = np.unique(seed[seed > 0])
        if len(ids) == 0: continue
        reg = np.isin(lab, ids) & (cand > 0)
        cx, cy = (x1+x2)/2, y2
        d = np.sqrt((xx-cx)**2 + (yy-cy)**2)
        reg &= (d < reach*hb) & (yy > y2 - up*hb)
        if dir_deg is not None:
            ang = np.degrees(np.arctan2(yy - cy, xx - cx))
            da = np.abs((ang - dir_deg + 180) % 360 - 180)
            reg &= (da < cone) | (d < 0.12*hb)
            # keep only the part connected to the feet after the cone cut
            n2, l2 = cv2.connectedComponents((reg | p4d).astype(np.uint8), connectivity=8)
            sd = l2[fy0:fy1, fx0:fx1]; ids2 = np.unique(sd[sd > 0])
            reg &= np.isin(l2, ids2)
        out |= reg
        if dark_attach_h and hb*S >= dark_attach_h:
            # dark objects held by / attached to this person (tools) and their thin shadows
            ex0, ex1 = int(max(0, x1 - 0.8*wb)), int(min(w4, x2 + 0.3*wb))
            ey0, ey1 = int(max(0, y1 + 0.55*hb)), int(min(h4, y2 + 0.35*hb))
            dark = (L < 0.6*env) & ~p4
            dk = np.zeros((h4, w4), np.uint8)
            dk[ey0:ey1, ex0:ex1] = dark[ey0:ey1, ex0:ex1]
            near = cv2.dilate(p4.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(0.3*wb)|1,)*2)) > 0
            band = np.zeros((h4, w4), bool); band[int(y1 + 0.3*hb):int(y2), :] = True
            dk |= (dark & near & band).astype(np.uint8)
            nb, lb = cv2.connectedComponents(dk | p4d.astype(np.uint8), connectivity=8)
            ids3 = np.unique(lb[p4d & (lb > 0)])
            blu = (b > bl + dblue)
            dkm = np.isin(lb, ids3) & (dk > 0)
            if dir_deg is not None:   # bluish (shadow-like) parts must respect the sun direction
                dkm &= ~blu | (np.abs((np.degrees(np.arctan2(yy - cy, xx - cx)) - dir_deg + 180) % 360 - 180) < cone)
            out |= dkm
    out = cv2.dilate(out.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
    full = cv2.resize(out.astype(np.uint8), (W, H), interpolation=cv2.INTER_LINEAR) > 0
    if ret_dbg: return full, cand, env
    return full
