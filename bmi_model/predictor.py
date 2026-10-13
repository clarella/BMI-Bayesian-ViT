"""
High-level inference: face crop -> ensemble of 4 Bayesian-ViT -> BMI + uncertainty.

Prediction protocol (the one used in the paper)
-----------------------------------------------
4 models (seeds 42, 123, 456, 789) x 30 Monte Carlo forward passes
= 120 BMI samples per image.

    bmi       = mean of the 120 samples
    variance  = variance of the 120 samples (sigma^2, uncertainty used in the paper)
    sigma     = sqrt(variance), in kg/m^2
    entropy   = 0.5 * ln(2*pi*e * variance)  (Gaussian entropy, monotone in variance)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from .model import load_member

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224
N_MC = 30
LOG_2PIE = float(np.log(2.0 * np.pi * np.e))

# Same MTCNN settings used to build the training crops
MTCNN_MARGIN = 40
MTCNN_MIN_FACE = 40


def who_class(bmi: float) -> str:
    if bmi < 25:
        return "Normal"
    if bmi < 30:
        return "Overweight"
    if bmi < 35:
        return "Obesity I"
    return "Obesity II"


class BMIPredictor:
    """
    Args:
        weights_dir: folder with the 4 `bayes_vit_b16_seed*.pt` files.
        device: "cuda", "cpu" or None (auto).
        detect_face: crop the face with MTCNN first (as in training). Set it to
            False when the inputs are already tight face crops.
    """

    def __init__(self, weights_dir="weights", device=None, detect_face=True):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        weights_dir = Path(weights_dir)
        files = sorted(weights_dir.glob("bayes_vit_b16_seed*.pt"))
        if not files:
            raise FileNotFoundError(f"No bayes_vit_b16_seed*.pt files in {weights_dir}")
        self.models = [load_member(f, self.device) for f in files]
        # train() mode is REQUIRED: it makes the variational layers sample
        # new weights on every forward pass (eval() would return only the mean).
        for m in self.models:
            m.train()

        ref = weights_dir / "reference_variance_visualbmi_test.json"
        self.ref_var = np.sort(np.array(json.loads(ref.read_text()))) if ref.exists() else None

        self.tf = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self.mtcnn = None
        if detect_face:
            from facenet_pytorch import MTCNN
            self.mtcnn = MTCNN(image_size=IMAGE_SIZE, margin=MTCNN_MARGIN,
                               min_face_size=MTCNN_MIN_FACE, keep_all=False,
                               select_largest=True, post_process=False,
                               device=self.device)

    def _load(self, path):
        """Returns (tensor, face_detected). Falls back to the full image if no face."""
        img = Image.open(path).convert("RGB")
        if self.mtcnn is None:
            return self.tf(img), None
        face = self.mtcnn(img)
        if face is None:
            return self.tf(img), False
        face = Image.fromarray(face.permute(1, 2, 0).cpu().numpy().clip(0, 255).astype("uint8"))
        return self.tf(face), True

    @torch.no_grad()
    def mc_samples(self, x: torch.Tensor, n_mc: int = N_MC) -> np.ndarray:
        """x: [B,3,224,224] -> samples [n_models*n_mc, B]."""
        x = x.to(self.device)
        out = [m(x)[0] for m in self.models for _ in range(n_mc)]
        return torch.stack(out, 0).cpu().numpy()

    def predict(self, paths, batch_size=32, n_mc=N_MC):
        """Returns one dict per image."""
        rows = []
        paths = [str(p) for p in paths]
        for i in range(0, len(paths), batch_size):
            chunk = paths[i:i + batch_size]
            loaded = [self._load(p) for p in chunk]
            x = torch.stack([t for t, _ in loaded])
            s = self.mc_samples(x, n_mc)
            mean, var = s.mean(0), s.var(0)
            for p, (_, det), mu, v in zip(chunk, loaded, mean, var):
                row = {
                    "image": p,
                    "bmi": round(float(mu), 2),
                    "sigma": round(float(np.sqrt(v)), 3),
                    "variance": round(float(v), 4),
                    "entropy": round(0.5 * (LOG_2PIE + float(np.log(max(v, 1e-12)))), 4),
                    "who_class": who_class(float(mu)),
                }
                if self.ref_var is not None:
                    # % of VisualBMI test images with HIGHER variance than this one
                    # (100 = as confident as the most confident test images)
                    idx = np.searchsorted(self.ref_var, v, side="left")
                    row["confidence_pct"] = round(100.0 * (1 - idx / len(self.ref_var)), 1)
                if det is not None:
                    row["face_detected"] = det
                rows.append(row)
        return rows
