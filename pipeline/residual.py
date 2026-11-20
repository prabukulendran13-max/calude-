"""Find people still visible in the current output: tiled 2x detection on output frames, report detections mostly outside the fill."""
import sys, json, numpy as np, cv2, torch
from common_pp import *
from ultralytics import YOLO
v = sys.argv[1]; step = int(sys.argv[2]); torch.set_num_threads(int(sys.argv[3]))
m = YOLO('/tmp/claude-0/-home-user-calude-/c996a4b3-e028-50ae-a262-aa818c3654dc/scratchpad/models/yolo11x-seg.pt')
fr = np.load(f'{WORK}/in/{v}_rgb8.npy', mmap_mode='r'); N = len(fr)
mpk = np.load(f'{WORK}/{v}_mask.npy', mmap_mode='r')
b = json.load(open(f'{WORK}/{v}_band.json')); Y0 = b['y0'] * H // PH
out = []
for t in range(0, N, step):
    img = np.asarray(fr[t]); fill = np.unpackbits(mpk[t], axis=-1)[:, :W] > 0
    # tiles over the people band at 2x
    for x0 in (0, 440):
        for y0 in range(Y0, H, 520):
            y1 = min(H, y0 + 640); x1 = x0 + 640
            tile = cv2.resize(img[y0:y1, x0:x1], ((x1-x0)*2, (y1-y0)*2), interpolation=cv2.INTER_CUBIC)
            r = m.predict(np.ascontiguousarray(tile[..., ::-1]), imgsz=1280, conf=0.25, classes=[0], verbose=False)[0]
            for bx, c in zip(r.boxes.xyxy.numpy(), r.boxes.conf.numpy()):
                X1, Y1, X2, Y2 = (bx/2 + [x0, y0, x0, y0]).astype(int)
                ov = fill[Y1:Y2, X1:X2].mean() if (Y2 > Y1 and X2 > X1) else 1
                if ov < 0.6: out.append((t, float(c), int(X1), int(Y1), int(X2), int(Y2), float(ov)))
    print(t, len(out), flush=True)
json.dump(out, open(f'{WORK}/{v}_residual.json', 'w'))
print('done')
