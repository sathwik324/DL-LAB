# Experiment 1 — Single Layer Perceptron from Scratch

Implementation of a **Single Layer Perceptron (SLP)** for binary
classification, built entirely from scratch using NumPy, and evaluated on
the [UCI Banknote Authentication dataset](https://archive.ics.uci.edu/ml/datasets/banknote+authentication).

No `sklearn` model (perceptron, preprocessing, or metrics) is used anywhere
in the pipeline — activation, the training loop, the weight-update rule,
normalization, the train/test split, and every evaluation metric are all
handwritten with NumPy/Pandas.

## Project Structure

```
Experiment_1/
│
├── src/
│   ├── main.py            # Orchestrates the full end-to-end pipeline
│   ├── perceptron.py       # From-scratch Single Layer Perceptron
│   ├── preprocessing.py    # Loading, validation, normalization, split
│   ├── eda.py               # Exploratory data analysis summaries
│   ├── visualization.py    # All plot-generation functions (300 dpi)
│   ├── evaluation.py        # Accuracy / Precision / Recall / F1 / CM
│   └── utils.py              # Seeding, logging, I/O helpers
│   all combined into a single source code file :- Experiment1
├── data/
│   └── data_banknote_authentication.txt
├── outputs/                  # Generated CSV results (metrics, history)
├── plots/                    # Generated figures (300 dpi PNGs)
├── README.md
├── requirements.txt
└── .gitignore
```

## Dataset

The **Banknote Authentication Dataset** (UCI Machine Learning Repository)
contains 1,372 samples describing features extracted via Wavelet Transform
from images of genuine and forged banknote-like specimens:

| Feature    | Description                                   |
|------------|------------------------------------------------|
| variance   | Variance of Wavelet Transformed image           |
| skewness   | Skewness of Wavelet Transformed image           |
| curtosis   | Kurtosis of Wavelet Transformed image           |
| entropy    | Entropy of the image                            |
| class      | 0 = Genuine, 1 = Forged                         |

Class balance: 762 genuine (0) / 610 forged (1).

## How to Run

```bash
# 1. Create and activate a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full standalone pipeline
python src/Experiment1.py

## Results Summary (η = 0.01, seed = 42)

| Metric     | Test Set Value |
|------------|-----------------|
| Accuracy   | 0.9781          |
| Precision  | 0.9840          |
| Recall     | 0.9685          |
| F1-score   | 0.9762          |

Full results, the learning-rate comparison, and interpretation of every
generated plot are available in `report`.

## The Perceptron Learning Rule

The model is trained using the classical rule (Rosenblatt, 1958):


w ← w + η (y − ŷ) x
b ← b + η (y − ŷ)


where `ŷ = step(w·x + b)` and `step(z) = 1 if z ≥ 0 else 0`.

## Reproducibility

All random operations (weight initialization, train/test shuffling) are
controlled by a fixed seed (`RANDOM_SEED = 42` in `src/utils.py`), so
re-running `python src/main.py` reproduces identical results.

## Author

Sathwik — B.Tech Artificial Intelligence and Data Science, Shiv Nadar
University, Chennai.
