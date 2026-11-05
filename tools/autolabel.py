"""Turn recorded rig frames into a YOLO dataset, labelled by the OpenCV + subpixel detector.

  python tools/autolabel.py data/raw/20261105_*            # one or more recording folders
  python tools/autolabel.py data/raw/* --val-every 5       # every 5th time block goes to val
Then fine-tune from the synthetic model:
  yolo detect train model=runs/keys/weights/best.pt data=dataset_real/data.yaml imgsz=640 epochs=40 device=mps

Classes match tools/make_dataset.py and sam2_keys.py: 0 = mask square, 1 = wafer cross.
Frames recorded while a motor moved are skipped (motion blur). Frames where the detector
does not find both keys are listed in unlabelled.txt: those are the hard cases to label by
hand (or with SAM2) later. Train/val are split by contiguous time blocks, not frame by
frame, because frames 0.5 s apart are near duplicates. review_*.jpg are contact sheets with
the labels drawn: skim them and delete any wrong frame from dataset_real before training.
"""

import sys
from pathlib import Path as _Path

# tools/ lives one level below the app modules
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from subpixel import PreciseKeys

OUT = Path(__file__).resolve().parents[1]/'dataset_real'
BLOCK = 20          # frames per time block (10 s at the 0.5 s recording interval)
SHEET_TILES = 24    # frames per review contact sheet


def yolo_line(cls, x0, y0, x1, y1, w, h):
    x0, x1 = max(0, x0), min(w, x1)
    y0, y1 = max(0, y0), min(h, y1)
    if x1-x0 < 3 or y1-y0 < 3:
        return None
    return f'{cls} {(x0+x1)/2/w:.6f} {(y0+y1)/2/h:.6f} {(x1-x0)/w:.6f} {(y1-y0)/h:.6f}'


def label(det, frame):
    """YOLO label lines and boxes for one frame, or None when either key is missing."""
    h, w = frame.shape[:2]
    md, wd, *_ = det.process(frame, 'THETA', token=None, stationary=False)
    if not md or not wd:
        return None
    lines, boxes = [], []
    # Only measured squares; a reconstructed 4th square may sit under the wafer or off-frame.
    for q in md['quads'][:md['observed']]:
        q = np.asarray(q, np.float64)
        pad = 2
        b = (q[:, 0].min()-pad, q[:, 1].min()-pad, q[:, 0].max()+pad, q[:, 1].max()+pad)
        line = yolo_line(0, *b, w, h)
        if line:
            lines.append(line)
            boxes.append((0, b))
    x, y, n, _ = wd['rect']
    b = (x, y, x+n, y+n)
    line = yolo_line(1, *b, w, h)
    if line:
        lines.append(line)
        boxes.append((1, b))
    return lines, boxes


def draw(frame, boxes):
    out = frame.copy()
    for cls, (x0, y0, x1, y1) in boxes:
        color = (0, 200, 0) if cls == 0 else (0, 0, 255)
        cv2.rectangle(out, (int(x0), int(y0)), (int(x1), int(y1)), color, 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('folders', nargs='+')
    ap.add_argument('--val-every', type=int, default=5, help='every Nth time block goes to validation')
    ap.add_argument('--out', default=str(OUT))
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    for split in ('train', 'val'):
        (out/'images'/split).mkdir(parents=True)
        (out/'labels'/split).mkdir(parents=True)

    det = PreciseKeys(center_tolerance=6.0)
    stats = {'train': 0, 'val': 0, 'moving': 0, 'unlabelled': 0}
    unlabelled, tiles = [], []
    for folder in map(Path, args.folders):
        meta_path = folder/'frames.jsonl'
        if not meta_path.is_file():
            print(f'skip {folder}: no frames.jsonl')
            continue
        metas = [json.loads(l) for l in meta_path.read_text().splitlines() if l.strip()]
        for i, meta in enumerate(metas):
            if meta.get('moving'):
                stats['moving'] += 1
                continue
            frame = cv2.imread(str(folder/meta['file']))
            if frame is None:
                continue
            result = label(det, frame)
            name = f'{folder.name}_{Path(meta["file"]).stem}'
            if result is None:
                stats['unlabelled'] += 1
                unlabelled.append(str(folder/meta['file']))
                continue
            lines, boxes = result
            block, blocks = i//BLOCK, (len(metas)+BLOCK-1)//BLOCK
            # Short recordings: their last block is val, so val is never empty.
            last = blocks < args.val_every and blocks > 1 and block == blocks-1
            split = 'val' if block % args.val_every == args.val_every-1 or last else 'train'
            shutil.copy(folder/meta['file'], out/'images'/split/f'{name}.jpg')
            (out/'labels'/split/f'{name}.txt').write_text('\n'.join(lines)+'\n')
            stats[split] += 1
            if len(tiles) < 2000:
                tile = cv2.resize(draw(frame, boxes), (320, 240))
                cv2.putText(tile, name[-22:], (4, 14), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 0), 1)
                tiles.append(tile)

    (out/'data.yaml').write_text(f'path: {out}\ntrain: images/train\nval: images/val\n'
                                 'names:\n  0: square\n  1: cross\n')
    (out/'unlabelled.txt').write_text('\n'.join(unlabelled)+('\n' if unlabelled else ''))
    for k in range(0, len(tiles), SHEET_TILES):
        chunk = tiles[k:k+SHEET_TILES]
        chunk += [np.zeros_like(tiles[0])]*(SHEET_TILES-len(chunk))
        rows = [np.hstack(chunk[r:r+6]) for r in range(0, SHEET_TILES, 6)]
        cv2.imwrite(str(out/f'review_{k//SHEET_TILES:03d}.jpg'), np.vstack(rows))
    print(f"train {stats['train']}  val {stats['val']}  skipped moving {stats['moving']}  "
          f"unlabelled {stats['unlabelled']} (see unlabelled.txt)")
    print(f'dataset: {out}   review sheets: {out}/review_*.jpg')


if __name__ == '__main__':
    main()
