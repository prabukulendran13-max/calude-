"""Track specific hard-to-detect people (heads over walls, people behind lattices/statues) from seed boxes,
using zoomed crops + YOLO-seg, following background flow. Writes {v}_extra.npy (packed bits per frame)."""
import sys, json, numpy as np, cv2, torch
from vio import *
from ultralytics import YOLO
torch.set_num_threads(4)
v = sys.argv[1]; seeds = json.loads(sys.argv[2]); maxlen = int(sys.argv[3]) if len(sys.argv) > 3 else 60
m = YOLO('/tmp/claude-0/-home-user-calude-/c996a4b3-e028-50ae-a262-aa818c3654dc/scratchpad/models/yolo11x-seg.pt')
fr = np.load(f'{WORK}/in/{v}_rgb8.npy', mmap_mode='r'); N = len(fr)
extra = np.zeros((N, H, W), bool)
dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
def flow(a, b):
    s = 4
    A = cv2.cvtColor(cv2.resize(np.asarray(fr[a]), (W//s, H//s), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    B = cv2.cvtColor(cv2.resize(np.asarray(fr[b]), (W//s, H//s), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
    return dis.calc(A, B, None) * s  # a -> b at 1/4 res
def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2-x1)*max(0, y2-y1); u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i
    return i / max(u, 1)
def detect(t, box):
    x1, y1, x2, y2 = box; bw, bh = x2-x1, y2-y1
    mg = int(max(bw, bh)*1.5) + 48
    X0, Y0, X1, Y1 = max(0, x1-mg), max(0, y1-mg), min(W, x2+mg), min(H, y2+mg)
    crop = np.asarray(fr[t][Y0:Y1, X0:X1])
    sc = min(4.0, max(1.0, 640 / max(crop.shape[:2])))
    up = cv2.resize(crop, (int(crop.shape[1]*sc), int(crop.shape[0]*sc)), interpolation=cv2.INTER_CUBIC)
    r = m.predict(np.ascontiguousarray(up[..., ::-1]), imgsz=640, conf=0.12, classes=[0], retina_masks=True, verbose=False)[0]
    best = None
    if r.masks is not None:
        for j, bx in enumerate(r.boxes.xyxy.numpy()):
            fb = (bx / sc + [X0, Y0, X0, Y0])
            s_ = iou(fb, box)
            if s_ > 0.15 and (best is None or s_ > best[0]):
                mk = cv2.resize(r.masks.data[j].numpy().astype(np.uint8), (X1-X0, Y1-Y0), interpolation=cv2.INTER_NEAREST) > 0
                best = (s_, fb.astype(int).tolist(), mk, float(r.boxes.conf[j]))
    if best is None: return None
    full = np.zeros((H, W), bool); full[Y0:Y1, X0:X1] = best[2]
    return best[1], full, best[3]
log = []
for seed in seeds:
    t0, box0 = seed[0], seed[1]
    for direction in (1, -1):
        box = list(box0); t = t0; miss = 0; n = 0
        while 0 <= t < N and n < maxlen:
            d = detect(t, box)
            if d is None:
                miss += 1
                if miss > 4: break
                # keep previous box region masked (it is a person that the detector briefly lost)
                x1, y1, x2, y2 = box; extra[t, y1:y2, x1:x2] = True
            else:
                miss = 0; box, full, c = d; extra[t] |= full; log.append((t, c))
            nt = t + direction
            if not (0 <= nt < N): break
            f = flow(t, nt)
            x1, y1, x2, y2 = box
            fx = f[max(0, y1//4):max(1, y2//4), max(0, x1//4):max(1, x2//4)]
            dx, dy = (np.median(fx[..., 0]), np.median(fx[..., 1])) if fx.size else (0, 0)
            box = [int(x1+dx), int(y1+dy), int(x2+dx), int(y2+dy)]
            box = [max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3])]
            if box[2] - box[0] < 4 or box[3] - box[1] < 4: break
            t = nt; n += 1
    print('seed', seed, 'frames so far', int(extra.any(axis=(1, 2)).sum()), flush=True)
np.save(f'{WORK}/{v}_extra.npy', np.packbits(extra, axis=-1))
print('done', v, 'frames with extra', int(extra.any(axis=(1, 2)).sum()))
