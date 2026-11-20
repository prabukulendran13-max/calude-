"""Verify: run person segmentation on the filled frames; flag person-like detections that lie mostly inside the fill."""
import sys, json, numpy as np, cv2, torch
from common_pp import *
from ultralytics import YOLO
v = sys.argv[1]; thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
frames_sel = None
if len(sys.argv) > 3 and sys.argv[3]: frames_sel = set(json.load(open(sys.argv[3])))
torch.set_num_threads(int(sys.argv[4]) if len(sys.argv) > 4 else 4)
m = YOLO('/tmp/claude-0/-home-user-calude-/c996a4b3-e028-50ae-a262-aa818c3654dc/scratchpad/models/yolo11x-seg.pt')
masks = load_masks(v); frames8 = load_frames(v); N = len(masks)
b = json.load(open(f'{WORK}/{v}_band.json')); y0, y1 = b['y0'], b['y1']
G = np.load(f'{WORK}/{v}_gprop.npy', mmap_mode='r')
bad = np.zeros((N, y1-y0, PW), bool); report = []
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
for t in range(N):
    if frames_sel is not None and t not in frames_sel: continue
    img = frames8[t].copy(); img[y0:y1] = G[t]
    r = m.predict(np.ascontiguousarray(img[..., ::-1]), imgsz=int(__import__('os').environ.get('DIMG', '1152')), conf=float(__import__('os').environ.get('DCONF', '0.12')), classes=[0], retina_masks=True, verbose=False)[0]
    if r.masks is None: continue
    md = r.masks.data.numpy() > 0.5
    fill = masks[t]
    for j in range(len(md)):
        dm = md[j]
        if dm.sum() < 30: continue
        ov = (dm & fill).sum() / dm.sum()
        if ov >= thr:
            g = (cv2.dilate(dm.astype(np.uint8), k) > 0) & fill
            bad[t] |= g[y0:y1]
            report.append((t, round(float(r.boxes.conf[j]), 2), round(float(ov), 2), int(dm.sum())))
    if t % 50 == 0: print(v, t, len(report), flush=True)
np.save(f'{WORK}/{v}_bad.npy', bad) if not __import__('os').environ.get('BADOUT') else np.save(__import__('os').environ['BADOUT'], bad)
json.dump(report, open(f'{WORK}/{v}_bad.json' if not __import__('os').environ.get('BADOUT') else __import__('os').environ['BADOUT'] + '.json', 'w'))
print('done', v, 'flagged frames', len(set(r[0] for r in report)), 'dets', len(report))
