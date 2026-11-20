import numpy as np, cv2

def push_pull(d, w):
    """Smoothly interpolate values d (HxWxC) known with weights w (HxW, >0 where known)."""
    D = [d * w[..., None]]; Wt = [w.astype(np.float32)]
    while min(Wt[-1].shape) > 4:
        Wn = cv2.resize(Wt[-1], (Wt[-1].shape[1]//2, Wt[-1].shape[0]//2), interpolation=cv2.INTER_AREA) * 4
        Dn = cv2.resize(D[-1], (D[-1].shape[1]//2, D[-1].shape[0]//2), interpolation=cv2.INTER_AREA) * 4
        Wt.append(Wn); D.append(Dn)
    vals = [Dl / np.maximum(Wl, 1e-6)[..., None] for Dl, Wl in zip(D, Wt)]
    cur = vals[-1]
    for l in range(len(vals)-2, -1, -1):
        up = cv2.resize(cur, (vals[l].shape[1], vals[l].shape[0]), interpolation=cv2.INTER_LINEAR)
        if up.ndim == 2: up = up[..., None]
        a = np.clip(Wt[l], 0, 1)[..., None]
        cur = a * vals[l] + (1 - a) * up
    return cur

def membrane_correction(orig, fill, mask, s=4, ring=3, sigma=3.0, smooth=2.0):
    """orig, fill: HxWx3 float; mask: HxW bool (region replaced by fill).
    Returns correction field (HxWx3) to add to fill inside mask so it matches surroundings at the boundary."""
    H, W = mask.shape
    h, w = H//s, W//s
    o = cv2.resize(orig, (w, h), interpolation=cv2.INTER_AREA)
    f = cv2.resize(fill, (w, h), interpolation=cv2.INTER_AREA)
    m = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) > 0.5
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*ring+1,)*2)
    rin = m & ~(cv2.erode(m.astype(np.uint8), k) > 0)
    rout = (cv2.dilate(m.astype(np.uint8), k) > 0) & ~m
    # don't trust outside ring right at image borders handled implicitly
    def nconv(img, wt):
        num = cv2.GaussianBlur(img * wt[..., None], (0, 0), sigma)
        den = cv2.GaussianBlur(wt.astype(np.float32), (0, 0), sigma)
        return num / np.maximum(den, 1e-4)[..., None], den
    O, dO = nconv(o, rout.astype(np.float32))
    I, dI = nconv(f, rin.astype(np.float32))
    valid = rin & (dO > 0.02) & (dI > 0.02)
    d = np.where(valid[..., None], O - I, 0).astype(np.float32)
    D = push_pull(d, valid.astype(np.float32))
    D = cv2.GaussianBlur(D, (0, 0), smooth)
    D = D * m[..., None]
    D = cv2.GaussianBlur(D, (0, 0), 1.0)
    return cv2.resize(D, (W, H), interpolation=cv2.INTER_LINEAR)
