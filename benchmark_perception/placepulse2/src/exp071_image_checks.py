"""EXP-071 follow-up: are model errors concentrated on few images, and on dark or degenerate frames?

Reads outputs/exp071/model_errors_for_review.csv and random_hits_for_review.csv (written by exp071_error_analysis.py).
Brightness = mean grey level (0-255) of the image; 'dark' = below 40.
Usage: python src/exp071_image_checks.py      Output: outputs/exp071/image_checks.json
"""
from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import OUTPUTS, ZIP_PATH  # noqa: E402

OUT = OUTPUTS / "exp071"


def brightness(zf, paths):
    out = {}
    for p in paths:
        im = Image.open(io.BytesIO(zf.read(p))).convert("L").resize((64, 48))
        out[p] = float(np.asarray(im).mean())
    return out


def main() -> None:
    e = pd.read_csv(OUT / "model_errors_for_review.csv")
    h = pd.read_csv(OUT / "random_hits_for_review.csv")
    # concentration: how many error pairs involve images that appear in several error pairs
    imgs = pd.concat([e["image_left"], e["image_right"]])
    cnt = imgs.value_counts()
    res = {"n_model_errors": int(len(e)), "unique_images_in_errors": int(cnt.size),
           "images_in_3plus_errors": int((cnt >= 3).sum()),
           "share_error_pairs_touching_3plus_images": float(
               (e["image_left"].map(cnt).ge(3) | e["image_right"].map(cnt).ge(3)).mean()),
           "top10_images": cnt.head(10).to_dict()}
    rng = np.random.default_rng(42)
    es = e.head(300)  # most confident errors
    with zipfile.ZipFile(ZIP_PATH) as zf:
        be = brightness(zf, pd.unique(pd.concat([es["image_left"], es["image_right"]])))
        hb = brightness(zf, pd.unique(pd.concat([h["image_left"], h["image_right"]])))
    be_v, hb_v = np.array(list(be.values())), np.array(list(hb.values()))
    res["dark_share_top300_error_images"] = float((be_v < 40).mean())
    res["dark_share_random_hit_images"] = float((hb_v < 40).mean())
    # in confident errors, was the dark image the voted one?
    dark_voted = 0
    dark_pairs = 0
    for _, r in es.iterrows():
        bl, br = be[r["image_left"]], be[r["image_right"]]
        if min(bl, br) < 40:
            dark_pairs += 1
            dark_side_left = bl < br
            dark_voted += int((r["vote_left"] == 1) == dark_side_left)
    res["top300_errors_with_a_dark_image"] = dark_pairs
    res["of_which_dark_image_was_voted_safer"] = dark_voted
    (OUT / "image_checks.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
