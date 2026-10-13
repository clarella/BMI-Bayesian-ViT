"""
Estimate BMI + uncertainty for one image or a folder of images.

Examples
--------
    python predict.py photo.jpg
    python predict.py my_folder/ --out predictions.csv
    python predict.py faces_already_cropped/ --no-detect
"""
import argparse
import csv
from pathlib import Path

import torch

from bmi_model import BMIPredictor

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    ap = argparse.ArgumentParser(description="Bayesian-ViT BMI estimation")
    ap.add_argument("input", help="image file or folder (searched recursively)")
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--weights", default=str(Path(__file__).parent / "weights"))
    ap.add_argument("--no-detect", action="store_true",
                    help="skip MTCNN face cropping (inputs are already face crops)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--mc", type=int, default=30, help="MC samples per model (paper: 30)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0, help="seed for MC sampling (reproducible output)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    inp = Path(args.input)
    if inp.is_dir():
        paths = sorted(p for p in inp.rglob("*") if p.suffix.lower() in IMG_EXT)
    else:
        paths = [inp]
    if not paths:
        raise SystemExit(f"No images found in {inp}")

    predictor = BMIPredictor(args.weights, device=args.device, detect_face=not args.no_detect)
    print(f"{len(paths)} image(s) | {len(predictor.models)} models x {args.mc} MC samples "
          f"| device={predictor.device}")
    rows = predictor.predict(paths, batch_size=args.batch_size, n_mc=args.mc)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    for r in rows[:10]:
        print(f"  {Path(r['image']).name}: BMI {r['bmi']:.1f} +/- {r['sigma']:.1f} ({r['who_class']})")
    if len(rows) > 10:
        print(f"  ... ({len(rows) - 10} more)")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
