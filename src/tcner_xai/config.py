"""Project-wide paths, seeds and constants."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"
CACHE_DIR = ROOT / "artifacts"  # git-ignored: cleaned titles + fitted model

TRAIN_FILE = "DBLPTrainset.txt"
TEST_FILE = "DBLPTestset.txt"
TRUTH_FILE = "DBLPTestGroundTruth.txt"

SEED = 42

CLASSES = ["INFOCOM", "ISCAS", "SIGGRAPH", "VLDB", "WWW"]
CLASS_DESCRIPTIONS = {
    "INFOCOM": "computer networking",
    "ISCAS": "circuits & systems",
    "SIGGRAPH": "computer graphics",
    "VLDB": "databases",
    "WWW": "the Web",
}

# Same TF-IDF configuration as the original course submission.
VECTORIZER_PARAMS = dict(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words="english",
    min_df=3,
    max_df=0.85,
)
# liblinear fits one binary classifier per venue (one-vs-rest), which is what the
# original `multi_class="ovr"` asked for; that argument is deprecated in sklearn>=1.5.
CLASSIFIER_PARAMS = dict(solver="liblinear", max_iter=1000, random_state=SEED)


def ensure_dirs() -> None:
    for d in (FIG_DIR, RESULTS_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)
