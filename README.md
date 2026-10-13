# Bayesian-ViT: BMI estimation from a single face image

Code and trained model for the paper
**"Body Mass Index (BMI) estimation from a single frontal face image using a Bayesian-ViT architecture"**
Carlos Naranjo, Francisca Pérez, Rodrigo Wagner and Claudia Arellano. School of Business, Universidad Adolfo Ibáñez, Santiago, Chile. IMVIP 2026.

Given a photo, the model crops the face and returns the estimated BMI **plus an
uncertainty** (variance). The uncertainty can be used to keep only the most reliable predictions.

| VisualBMI test set (1,144 images) | MAE (kg/m²) |
|---|---|
| All images | **4.434** |
| Male / Female | 4.073 / 5.209 |
| 75 % most confident (lowest variance) | 3.668 |
| 50 % most confident | 3.100 |
| 25 % most confident | 2.814 |
| 10 % most confident | 2.560 |

---

## 1. Installation

```bash
git clone <this-repo>
cd <this-repo>
pip install -r requirements.txt
pip install facenet-pytorch==2.6.0 --no-deps
```

A GPU is recommended but not required (CPU works, just slower).

## 2. Weights

The 4 trained models (~345 MB each, 1.4 GB total) go in `weights/`.
GitHub does not accept files larger than 100 MB in a normal commit, so in the
online repository they are distributed separately (e.g. as GitHub Release
assets); download them into `weights/`:

```
weights/
├── bayes_vit_b16_seed42.pt
├── bayes_vit_b16_seed123.pt
├── bayes_vit_b16_seed456.pt
├── bayes_vit_b16_seed789.pt
└── reference_variance_visualbmi_test.json   (already in the repo)
```

## 3. Usage

### Command line

```bash
# one image (full photo: the face is detected and cropped automatically)
python predict.py photo.jpg

# a whole folder (recursive) -> predictions.csv
python predict.py my_photos/ --out predictions.csv

# images that are ALREADY tight face crops: skip face detection
python predict.py my_face_crops/ --no-detect
```

### Python

```python
from bmi_model import BMIPredictor

predictor = BMIPredictor("weights")            # detect_face=False for face crops
results = predictor.predict(["photo1.jpg", "photo2.jpg"])
print(results[0])
# {'image': 'photo1.jpg', 'bmi': 32.13, 'sigma': 3.442, 'variance': 11.8504,
#  'entropy': 2.6551, 'who_class': 'Obesity I', 'confidence_pct': 35.3, 'face_detected': True}
```

### Output columns

| Column | Meaning |
|---|---|
| `bmi` | Predicted BMI = mean of the 120 Monte Carlo samples |
| `sigma` | Standard deviation of the 120 samples (kg/m²) |
| `variance` | sigma² = predictive variance, **the uncertainty used in the paper** (lower = more reliable) |
| `entropy` | Gaussian entropy 0.5·ln(2πe·sigma²); gives the same ranking as `variance` |
| `who_class` | WHO class of the prediction: Normal (<25), Overweight (25-30), Obesity I (30-35), Obesity II (≥35) |
| `confidence_pct` | % of VisualBMI test images whose variance is higher than this one (100 = as confident as the most confident test images) |
| `face_detected` | Whether MTCNN found a face (if not, the whole image is used and the prediction is not reliable) |

**Notes**
- The prediction is stochastic: running the same image twice can change the BMI by a few tenths. Use `--seed` for reproducible output.
- The model was trained on adults from r/progresspics (VisualBMI, BMI ≈ 18-58). Predictions outside that population (children, extreme poses, very low-resolution faces) are less reliable, and usually show a higher variance.

---

## 4. How the model works

```
photo -> MTCNN face crop (224x224, margin 40) -> ViT-B/16 -> variational head -> BMI sample
```

- **Backbone:** ViT-B/16 (`timm`: `vit_base_patch16_224.augreg_in21k_ft_in1k`), pre-trained on ImageNet-21k and fine-tuned on ImageNet-1k.
- **Bayesian head:** two variational linear layers (`bayesian-torch` `LinearReparameterization`, 768 → 256 → 1). Each weight is a Gaussian N(μ, σ²) and a new set of weights is sampled at every forward pass.
- **Ensemble + Monte Carlo:** 4 models trained with seeds 42, 123, 456 and 789, with **30 forward passes each = 120 BMI samples per image**. The prediction is the mean of the 120 samples and the uncertainty is their variance.

## 5. Reproducing the paper results

`results/mc_samples_visualbmi_test.npz` contains the 120 Monte Carlo samples of
every test image used for the paper tables, so the exact numbers can be
reproduced without a GPU or the images:

```bash
python results/reproduce_tables.py
```

```
                     Normal  Overweight  Obesity I  Obesity II   Male  Female  Overall
Bayesian-ViT (10%)    1.927       2.577      7.435      10.462  2.579   2.483    2.560
Bayesian-ViT (25%)    2.125       2.149      5.699      10.382  2.743   3.079    2.814
Bayesian-ViT (50%)    2.718       1.874      4.430       9.785  2.941   3.546    3.100
Bayesian-ViT (75%)    3.112       2.129      3.430       9.040  3.420   4.257    3.668
Bayesian-ViT (100%)   3.361       2.497      3.400       8.196  4.073   5.209    4.434
```

It also prints the MAE per variance decile and writes `results/test_predictions.csv`
(prediction, variance and error of each test image). Figures are in `results/figures/`.

Running the models again (`python training/evaluate.py`, needs the images) draws
new Monte Carlo samples, so the numbers change slightly (e.g. overall MAE 4.437
instead of 4.434).

## 6. Reproducing the training (optional)

Everything needed is in `training/`.

**Data.** Download the VisualBMI images (`bodyface_1to17/`, from the Kaggle dataset *visual-bmi*) into `training/data/`.
`training/data/visual_bmi_annotations.csv` (5,897 images) is already included.

```bash
# 1. Face crops with MTCNN  -> 5,741 images with a face
python training/prepare_faces.py

# 2. Train/test split: already provided as training/visualbmi_split.csv
#    (make_split.py is the script that created it, kept for transparency)

# 3. Train the 4 seeds and evaluate the ensemble
bash training/train_all_seeds.sh

# Evaluate the released weights on the test set
python training/evaluate.py
```

**Split** (`training/visualbmi_split.csv`, 5,642 images):
- person-level split: no person appears in both train and test;
- test = 1,144 images (canonical 80/20 split by person);
- train = 4,498 images, after **removing every train image whose face is a near-duplicate of a test face** (FaceNet similarity > 0.80). VisualBMI contains the same person under different IDs, so without this step part of the test set leaks into train;
- validation = 10 % of the train persons (chosen with each run's seed), used for model selection only.

**Hyper-parameters** (defaults in `train_bayes.py`):

| | |
|---|---|
| Loss | ELBO: L1 + KL(q(w) ‖ N(0,1)) / n_batches |
| Optimizer | AdamW, weight decay 1e-4 |
| Phase 1 (epochs 1-10) | backbone frozen, lr 1e-4 |
| Phase 2 (epochs 11-60) | full fine-tuning, lr 1e-5 |
| LR schedule | linear warm-up 500 steps + cosine decay to 1e-6 (each phase) |
| Batch size | 8, gender-balanced sampler |
| Regularization | dropout 0.3 (head), gradient clipping 1.0, EMA of weights (0.999) |
| Augmentation | horizontal flip, rotation ±15°, color jitter 0.2 |
| Model selection | best validation MAE, early stopping (patience 10) |
| Precision | fp32 |

## Repository structure

```
├── predict.py                 # command-line inference
├── bmi_model/
│   ├── model.py               # Bayesian-ViT architecture
│   └── predictor.py           # face crop + ensemble + Monte Carlo sampling
├── weights/                   # 4 trained models + reference variances
├── results/
│   ├── reproduce_tables.py    # exact paper tables from the saved MC samples
│   ├── mc_samples_visualbmi_test.npz
│   ├── test_predictions.csv
│   └── figures/               # risk-coverage, MAE per variance decile, calibration
└── training/
    ├── prepare_faces.py       # step 1: MTCNN face crops
    ├── make_split.py          # step 2: split + duplicate removal (already done)
    ├── visualbmi_split.csv    # exact split used in the paper
    ├── dataset.py
    ├── train_bayes.py         # trains one ensemble member
    ├── train_all_seeds.sh     # trains the 4 members
    ├── evaluate.py            # test-set metrics of the ensemble
    └── data/visual_bmi_annotations.csv
```

## Citation

```bibtex
@inproceedings{bayesvit_bmi_2026,
  title     = {Body Mass Index (BMI) estimation from a single frontal face image using a Bayesian-ViT architecture},
  author    = {Naranjo, Carlos and P{\'e}rez, Francisca and Wagner, Rodrigo and Arellano, Claudia},
  booktitle = {Irish Machine Vision and Image Processing Conference (IMVIP)},
  year      = {2026}
}
```

## Acknowledgements

VisualBMI dataset: Kocabey et al., *Face-to-BMI: Using Computer Vision to Infer Body Mass Index on Social Media*, ICWSM 2017.
