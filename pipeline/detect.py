import sys, numpy as np, pickle, time, torch
from vio import *
from ultralytics import YOLO
v = sys.argv[1]; torch.set_num_threads(int(sys.argv[2]))
m = YOLO('../models/yolo11x-seg.pt')
fr = np.load(f'{WORK}/in/{v}_rgb8.npy', mmap_mode='r')
res = []
t0 = time.time()
for i in range(len(fr)):
    img = np.ascontiguousarray(fr[i][..., ::-1])
    r = m.predict(img, imgsz=1280, conf=0.08, classes=[0,24,25,26,28,67], retina_masks=True, verbose=False)[0]
    inst = []
    if r.masks is not None:
        md = r.masks.data.cpu().numpy() > 0.5
        for k in range(len(r.boxes)):
            x1,y1,x2,y2 = r.boxes.xyxy[k].cpu().numpy().round().astype(int)
            x1=max(0,x1);y1=max(0,y1);x2=min(W,x2);y2=min(H,y2)
            mk = md[k]
            ys, xs = np.nonzero(mk)
            if len(ys)==0: bb=(x1,y1,x2,y2); crop=np.zeros((1,1),bool)
            else:
                bb=(xs.min(), ys.min(), xs.max()+1, ys.max()+1); crop = mk[bb[1]:bb[3], bb[0]:bb[2]]
            inst.append(dict(cls=int(r.boxes.cls[k]), conf=float(r.boxes.conf[k]), box=(x1,y1,x2,y2), mbox=bb, mask=np.packbits(crop), mshape=crop.shape))
    res.append(inst)
    if i%20==0: print(v, i, len(inst), round(time.time()-t0), flush=True)
pickle.dump(res, open(f'{WORK}/{v}_det.pkl','wb'))
print('done', v)
