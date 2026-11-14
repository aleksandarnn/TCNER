"""The venue classifier (TF-IDF + one-vs-rest logistic regression) and a cached
"bundle" with data, features and the fitted model that every notebook shares."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .config import CACHE_DIR, CLASSIFIER_PARAMS, VECTORIZER_PARAMS, ensure_dirs
from .data import SpacyPreprocessor, load_dblp


@dataclass
class VenueClassifier:
    """TF-IDF + logistic regression with explicit, reusable prediction functions.

    Every explainer goes through :meth:`proba_clean` / :meth:`logit_clean` (input =
    already-cleaned text) or :meth:`proba_raw` (input = raw title, cleaned with the
    *same* spaCy pipeline used for training). The original notebook fed raw titles
    straight into the vectoriser for LIME and Anchors, skipping lemmatisation, so
    those explanations described a slightly different model.
    """

    vectorizer_params: dict = field(default_factory=lambda: dict(VECTORIZER_PARAMS))
    classifier_params: dict = field(default_factory=lambda: dict(CLASSIFIER_PARAMS))

    def __post_init__(self):
        self.vectorizer = TfidfVectorizer(**self.vectorizer_params)
        self.clf = LogisticRegression(**self.classifier_params)
        self.preprocess: SpacyPreprocessor | None = None

    # -- training -------------------------------------------------------------------
    def fit(self, clean_texts, labels):
        X = self.vectorizer.fit_transform(clean_texts)
        self.clf.fit(X, labels)
        return self

    @property
    def classes_(self) -> np.ndarray:
        return self.clf.classes_

    @property
    def feature_names_(self) -> np.ndarray:
        return self.vectorizer.get_feature_names_out()

    def class_index(self, label: str) -> int:
        return int(np.flatnonzero(self.classes_ == label)[0])

    # -- prediction -----------------------------------------------------------------
    def transform(self, clean_texts):
        return self.vectorizer.transform(list(clean_texts))

    def proba_clean(self, clean_texts) -> np.ndarray:
        return self.clf.predict_proba(self.transform(clean_texts))

    def logit_clean(self, clean_texts) -> np.ndarray:
        """Per-venue log-odds of the one-vs-rest binary classifiers."""
        return self.clf.decision_function(self.transform(clean_texts))

    def predict_clean(self, clean_texts) -> np.ndarray:
        return self.classes_[self.proba_clean(clean_texts).argmax(1)]

    def proba_raw(self, titles) -> np.ndarray:
        if self.preprocess is None:
            self.preprocess = SpacyPreprocessor()
        return self.proba_clean(self.preprocess(list(titles)))

    def predict_raw(self, titles) -> np.ndarray:
        return self.classes_[self.proba_raw(titles).argmax(1)]

    # -- persistence (spaCy is reloaded, not pickled) --------------------------------
    def __getstate__(self):
        state = self.__dict__.copy()
        state["preprocess"] = None
        return state


def _config_hash() -> str:
    payload = json.dumps([VECTORIZER_PARAMS, CLASSIFIER_PARAMS], sort_keys=True, default=str)
    return hashlib.md5(payload.encode()).hexdigest()[:8]


def load_bundle(force: bool = False, verbose: bool = True) -> SimpleNamespace:
    """Load data, clean titles, fit the classifier and cache everything.

    Returns a namespace with ``train``, ``test`` (DataFrames with ``title``,
    ``label``, ``clean``), ``train_raw`` (before de-duplication), ``model``,
    ``X_train``, ``X_test``, ``y_train``, ``y_test`` and ``pre`` (preprocessor).
    """
    ensure_dirs()
    key = _config_hash()
    clean_path = CACHE_DIR / "cleaned_titles.pkl"
    model_path = CACHE_DIR / f"model_{key}.joblib"

    train_raw, test = load_dblp()
    pre = SpacyPreprocessor()

    if clean_path.exists() and not force:
        pre._cache.update(pd.read_pickle(clean_path))
    n_cached = len(pre._cache)
    pre(list(train_raw["title"]) + list(test["title"]))
    if len(pre._cache) != n_cached or not clean_path.exists():
        pd.to_pickle(dict(pre._cache), clean_path)

    train_raw = train_raw.assign(clean=pre(list(train_raw["title"])))
    test = test.assign(clean=pre(list(test["title"])))
    # Same de-duplication as the original notebook: keep the first copy of a title.
    train = train_raw.drop_duplicates(subset="title", keep="first").reset_index(drop=True)

    if model_path.exists() and not force:
        model = joblib.load(model_path)
    else:
        model = VenueClassifier().fit(train["clean"], train["label"])
        joblib.dump(model, model_path)
    model.preprocess = pre

    X_train = model.transform(train["clean"])
    X_test = model.transform(test["clean"])
    if verbose:
        acc = (model.clf.predict(X_test) == test["label"].values).mean()
        print(f"train={len(train):,} test={len(test):,} features={X_train.shape[1]:,} "
              f"test accuracy={acc:.4f}")
    return SimpleNamespace(
        train=train, train_raw=train_raw, test=test, model=model, pre=pre,
        X_train=X_train, X_test=X_test,
        y_train=train["label"].values, y_test=test["label"].values,
    )
