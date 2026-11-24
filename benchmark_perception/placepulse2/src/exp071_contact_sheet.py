"""EXP-071 — contact sheets of model errors and random hits for visual review (PNG, small).

Usage: python src/exp071_contact_sheet.py [--n 8]   Output: outputs/exp071/figures/review_{errors,hits}.png
"""
from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import OUTPUTS, ZIP_PATH  # noqa: E402

OUT = OUTPUTS / "exp071"
W, H = 320, 240


def sheet(df: pd.DataFrame, path: Path) -> None:
    canvas = Image.new("RGB", (2 * W + 30, len(df) * (H + 26)), "white")
    d = ImageDraw.Draw(canvas)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        for k, (_, r) in enumerate(df.iterrows()):
            y0 = k * (H + 26)
            d.text((4, y0 + 2), f"#{k}  conf={r['model_conf']:.2f}  cons={r['consensus_margin']:+.2f}  "
                               f"wealth={r['human_wealth_margin']:+.2f}", fill="black")
            for j, side in enumerate(["left", "right"]):
                im = Image.open(io.BytesIO(zf.read(r[f"image_{side}"]))).convert("RGB").resize((W, H))
                x0 = j * (W + 30)
                canvas.paste(im, (x0, y0 + 16))
                voted = (r["vote_left"] == 1) == (side == "left")
                pick = (r["model_margin"] > 0) == (side == "left")
                if voted:
                    d.rectangle([x0, y0 + 16, x0 + W - 1, y0 + 16 + H - 1], outline="lime", width=6)
                if pick:
                    d.rectangle([x0 + 8, y0 + 24, x0 + 60, y0 + 40], fill="red")
                    d.text((x0 + 12, y0 + 26), "MODEL", fill="white")
                d.text((x0 + 4, y0 + 16 + H - 14), f"{r[f'city_{side}']} s={r[f's_{side}']:+.2f}", fill="yellow")
    canvas.save(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    e = pd.read_csv(OUT / "model_errors_for_review.csv").head(a.n)
    h = pd.read_csv(OUT / "random_hits_for_review.csv").head(a.n)
    sheet(e, OUT / "figures" / "review_errors.png")
    sheet(h, OUT / "figures" / "review_hits.png")
    print("ok")


if __name__ == "__main__":
    main()
