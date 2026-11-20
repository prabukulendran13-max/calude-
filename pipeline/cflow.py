"""Stage A: RAFT + ProPainter recurrent flow completion over the clip's mask band; saves completed flows."""
import sys, time, json, numpy as np, torch
from common_pp import *
from model.modules.flow_comp_raft import RAFT_bi
from model.recurrent_flow_completion import RecurrentFlowCompleteNet
v = sys.argv[1]; torch.set_num_threads(4)
masks = load_masks(v); frames8 = load_frames(v); N = len(masks)
y0, y1 = band_of(masks); hb = y1 - y0
json.dump({'y0': y0, 'y1': y1}, open(f'{WORK}/{v}_band.json', 'w'))
raft = RAFT_bi(f'{PP}/weights/raft-things.pth', torch.device('cpu'))
fcomp = RecurrentFlowCompleteNet(f'{PP}/weights/recurrent_flow_completion.pth').eval()
out = np.lib.format.open_memmap(f'{WORK}/{v}_cflow.npy', mode='w+', dtype=np.float16, shape=(2, N-1, 2, hb, PW))
rs = 0.67; rh, rw = int(round(hb*rs/8))*8, int(round(PW*rs/8))*8
t0 = time.time()
# raw flows for all pairs (chunks of 2 pairs)
gtf = np.zeros((N-1, 2, hb, PW), np.float16); gtb = np.zeros((N-1, 2, hb, PW), np.float16)
with torch.no_grad():
    for i in range(0, N-1, 2):
        j = min(N, i+3)
        x = torch.from_numpy(frames8[i:j, y0:y1]).permute(0, 3, 1, 2).float().div(127.5).sub(1)
        xs = torch.nn.functional.interpolate(x, size=(rh, rw), mode='area')[None]
        f_, b_ = raft(xs, iters=12)
        for arr, src in ((gtf, f_), (gtb, b_)):
            y = torch.nn.functional.interpolate(src[0], size=(hb, PW), mode='bilinear', align_corners=False)
            y[:, 0] *= PW/rw; y[:, 1] *= hb/rh
            arr[i:i+y.shape[0]] = y.numpy().astype(np.float16)
        if i % 50 == 0: print(v, 'raft', i, round(time.time()-t0), flush=True)
    # flow completion in chunks of 40 with pad 5
    mk = torch.from_numpy(masks[:, y0:y1]).float()[None, :, None]
    sub, pad = 40, 5
    for f0 in range(0, N-1, sub):
        s_f = max(0, f0-pad); e_f = min(N-1, f0+sub+pad)
        ps = f0 - s_f; pe = e_f - min(N-1, f0+sub)
        g = (torch.from_numpy(gtf[s_f:e_f].astype(np.float32))[None], torch.from_numpy(gtb[s_f:e_f].astype(np.float32))[None])
        with torch.autocast('cpu', dtype=torch.bfloat16):
            pr, _ = fcomp.forward_bidirect_flow(g, mk[:, s_f:e_f+1])
            pr = fcomp.combine_flow(g, pr, mk[:, s_f:e_f+1])
        out[0, f0:f0+sub] = pr[0][0, ps:e_f-s_f-pe].float().numpy().astype(np.float16)
        out[1, f0:f0+sub] = pr[1][0, ps:e_f-s_f-pe].float().numpy().astype(np.float16)
        print(v, 'fcomp', f0, round(time.time()-t0), flush=True)
out.flush()
print('done', v, y0, y1)
