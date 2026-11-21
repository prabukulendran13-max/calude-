# People removal: IMG_4711 and IMG_4713

## Outputs (`output/`)

| File | What it is |
|---|---|
| `IMG_4711_no_people.mp4` | Clip 1 with every person removed. 1080×1920, 10-bit HEVC, HDR (HLG / BT.2020), all 345 frames, original audio. |
| `IMG_4713_no_people.mp4` | Clip 2 with every person removed. Same format, all 415 frames, original audio. |
| `*_before_after_SDR.mp4` | Side-by-side previews (original on the left, result on the right), tone-mapped to SDR so they look right on any screen. |

Pixels outside the removed regions (and a narrow blend edge) are bit-identical to the original 10-bit frames. Only the areas where people, their shadows, carried objects and floor reflections were have been rebuilt.

## How it was done (`pipeline/`)

1. **Segmentation.** YOLO11x-seg runs on every frame. Detections are linked into tracks, and each track was reviewed by eye to drop the temple's statues, which the detector mistakes for people. Held objects (bags, phones, a string trimmer) and cast shadows (found from colour and brightness relative to the sunlit ground, aligned with the sun direction) are added. Floor reflections are added in the second clip, and partly hidden people (heads above walls, figures behind lattices) are found with zoomed, tracked re-detection. Masks are joined with the neighbouring frames using optical flow so nothing flickers through (`detect.py`, `tracks.py`, `build_mask.py`, `shadow.py`, `targeted_track.py`).
2. **Flow completion.** RAFT optical flow plus ProPainter's recurrent flow-completion network estimate how the background moves behind each person (`cflow.py`).
3. **Whole-clip propagation.** Real background pixels from every other frame in the clip are carried into each hole along the completed flow. Where the background is never visible, LaMa fills one keyframe at a time (holes fully inside the frame first, then holes at the frame edge), and the result is re-propagated so the fill stays consistent over time (`gprop.py`).
4. **Verify and repair.** Person detection runs again on the filled frames. Anything person-shaped found in a filled area is cleared, along with its motion trail, and rebuilt (`detect_fill.py`, `gprop.py` with `FIX=1`).
5. **Full-resolution finish.** Fine detail is copied back from the full-resolution frames along the flow, colours are matched at the edges of each filled region, and the fill is blended into the original 10-bit HDR frames, then encoded with x265 (`detail2.py`, `membrane.py`, `composite.py`, `encode.sh`).

## Known limitations

Some background spots were never visible in either clip because people stood in front of them the whole time. The main ones are the crowd in front of the temple steps in IMG_4713 around frames 230–290, and the area behind the man in green next to the golden statue near the end. Those areas are synthesized rather than recovered, and they show some smudging or texture repetition when viewed up close.
