import numpy as np, cv2, os
W, H = 1080, 1920
WORK = '/tmp/claude-0/-home-user-calude-/c996a4b3-e028-50ae-a262-aa818c3654dc/scratchpad/work'
FS = W*H*3//2  # samples per frame

def yuv_mm(v, mode='r'):
    p = f'{WORK}/in/{v}.yuv'
    n = os.path.getsize(p)//(FS*2)
    return np.memmap(p, dtype=np.uint16, mode=mode, shape=(n, FS))

def split(fr):
    y = fr[:W*H].reshape(H, W)
    u = fr[W*H:W*H+W*H//4].reshape(H//2, W//2)
    v = fr[W*H+W*H//4:].reshape(H//2, W//2)
    return y, u, v

def yuv2rgb(y, u, v):
    """10-bit limited bt2020 -> float32 RGB (HLG code values, 0..1)"""
    Y = (y.astype(np.float32)-64)/876
    U = cv2.resize((u.astype(np.float32)-512)/896, (W, H), interpolation=cv2.INTER_LINEAR)
    V = cv2.resize((v.astype(np.float32)-512)/896, (W, H), interpolation=cv2.INTER_LINEAR)
    R = Y + 1.4746*V
    G = Y - 0.16455*U - 0.57135*V
    B = Y + 1.8814*U
    return np.stack([R, G, B], -1)

def rgb2yuv(rgb):
    """float RGB 0..1 -> float Y (full res), U,V (full res) in 10-bit code units"""
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    Yf = 0.2627*R + 0.6780*G + 0.0593*B
    U = (B - Yf)/1.8814
    V = (R - Yf)/1.4746
    return Yf*876+64, U*896+512, V*896+512

def frame_rgb8(mm, i):
    rgb = yuv2rgb(*split(mm[i]))
    return (np.clip(rgb, 0, 1)*255+0.5).astype(np.uint8)
