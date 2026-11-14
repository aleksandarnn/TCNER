"""Data loading, spaCy preprocessing and a data-quality audit for the DBLP titles."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import DATA_DIR, TEST_FILE, TRAIN_FILE, TRUTH_FILE

# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def load_dblp(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(train, test)`` with columns ``label`` and ``title``.

    The files are read exactly as in the original submission (tab separated, no
    header) so the row counts match the report: 21,643 train / 3,373 test titles.
    """
    missing = [f for f in (TRAIN_FILE, TEST_FILE, TRUTH_FILE) if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing data file(s) {missing} in {data_dir}. See data/README.md."
        )
    # Every file starts with a numeric id column.
    read = dict(sep="\t", header=None, index_col=0)
    train = pd.read_csv(data_dir / TRAIN_FILE, names=["id", "label", "title"], **read)
    test_titles = pd.read_csv(data_dir / TEST_FILE, names=["id", "title"], **read)
    test_truth = pd.read_csv(data_dir / TRUTH_FILE, names=["id", "label"], **read)
    if not test_titles.index.equals(test_truth.index):
        raise ValueError("Test titles and ground truth are not aligned")
    train = train.reset_index(drop=True)
    test = pd.DataFrame({"label": test_truth["label"].values, "title": test_titles["title"].values})
    for df in (train, test):
        df["title"] = df["title"].astype(str)
    return train, test


def normalise_title(title: str) -> str:
    """Light normalisation used only for duplicate / overlap detection."""
    t = re.sub(r"<[^>]+>", "", title)  # strip HTML tags such as <i>...</i>
    t = re.sub(r"[^\w\s]", " ", t.lower())
    return re.sub(r"\s+", " ", t).strip()


# --------------------------------------------------------------------------------------
# Preprocessing (identical rules to the original notebook)
# --------------------------------------------------------------------------------------


class SpacyPreprocessor:
    """Lower-case, lemmatise, drop stop words / punctuation / tokens of <= 2 chars.

    Results are cached per input string, which matters because explainers call
    the model thousands of times on overlapping texts. The dependency parser and
    NER are disabled: they do not influence lemmas or stop-word flags, so the
    output is identical to ``nlp(text.lower())`` but several times faster.
    """

    def __init__(self, model: str = "en_core_web_sm"):
        import spacy

        self.model_name = model
        self.nlp = spacy.load(model, disable=["parser", "ner"])
        self._cache: dict[str, str] = {}

    @staticmethod
    def _clean_doc(doc) -> str:
        return " ".join(
            tok.lemma_ for tok in doc if not tok.is_stop and not tok.is_punct and len(tok) > 2
        )

    def __call__(self, texts, batch_size: int = 512):
        single = isinstance(texts, str)
        texts = [texts] if single else list(texts)
        todo = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if todo:
            docs = self.nlp.pipe((t.lower() for t in todo), batch_size=batch_size)
            for t, doc in zip(todo, docs):
                self._cache[t] = self._clean_doc(doc)
        out = [self._cache[t] for t in texts]
        return out[0] if single else out


# --------------------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------------------

# Titles that describe a proceedings artefact rather than a research contribution.
BOILERPLATE_PATTERN = re.compile(
    r"^(front matter|back matter|frontmatter|title page|table of contents|contents|"
    r"author index|index|preface|foreword|editorial|introduction|welcome( message)?|"
    r"message from .*|letter from .*|keynote( address| talk)?|invited talk|"
    r"panel( session| discussion)?|session \w+|program committee|organizing committee|"
    r"committees?|reviewers?( list)?|list of reviewers|"
    r"copyright|sponsors?|conference program|tutorials?|workshops?|"
    r"(general|program) chairs?'? (message|welcome)|.*\b(proceedings)\b.*)$"
)


HTML_TAG = r"</?(?:i|b|sub|sup|tt|em|strong)>"


def is_boilerplate(title: str) -> bool:
    return bool(BOILERPLATE_PATTERN.match(normalise_title(title)))


@dataclass
class AuditReport:
    n_train: int
    n_test: int
    train_dup_titles: int
    train_conflicting_titles: int
    test_dup_titles: int
    cross_split_overlap: int
    cross_split_label_mismatch: int
    train_boilerplate: int
    test_boilerplate: int
    html_markup_train: int
    html_markup_test: int
    empty_after_cleaning_train: int
    empty_after_cleaning_test: int

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"value": self.__dict__}).rename_axis("check")


def audit(train: pd.DataFrame, test: pd.DataFrame) -> tuple[AuditReport, dict[str, pd.DataFrame]]:
    """Data-quality audit. ``train``/``test`` must have ``title``, ``label`` and ``clean``."""
    tr = train.assign(norm=train["title"].map(normalise_title))
    te = test.assign(norm=test["title"].map(normalise_title))

    labels_per_title = tr.groupby("norm")["label"].nunique()
    conflicting = labels_per_title[labels_per_title > 1].index

    train_label = tr.drop_duplicates("norm").set_index("norm")["label"]
    overlap = te[te["norm"].isin(train_label.index)]
    mismatch = overlap[overlap["label"].values != train_label.loc[overlap["norm"]].values]

    tr_bp = tr[tr["title"].map(is_boilerplate)]
    te_bp = te[te["title"].map(is_boilerplate)]

    report = AuditReport(
        n_train=len(tr),
        n_test=len(te),
        train_dup_titles=int(tr.duplicated("norm").sum()),
        train_conflicting_titles=len(conflicting),
        test_dup_titles=int(te.duplicated("norm").sum()),
        cross_split_overlap=len(overlap),
        cross_split_label_mismatch=len(mismatch),
        train_boilerplate=len(tr_bp),
        test_boilerplate=len(te_bp),
        html_markup_train=int(tr["title"].str.contains(HTML_TAG).sum()),
        html_markup_test=int(te["title"].str.contains(HTML_TAG).sum()),
        empty_after_cleaning_train=int((tr["clean"].str.len() == 0).sum()),
        empty_after_cleaning_test=int((te["clean"].str.len() == 0).sum()),
    )
    tables = {
        "most_common_train_titles": tr["norm"].value_counts().head(20).rename("count").to_frame(),
        "conflicting_titles": tr[tr["norm"].isin(conflicting)][["title", "label"]],
        "cross_split_overlap": overlap[["title", "label"]],
        "train_boilerplate_by_label": tr_bp.groupby(["norm", "label"]).size().rename("count")
        .reset_index().sort_values("count", ascending=False),
        "test_boilerplate": te_bp[["title", "label"]],
        "html_markup_examples": tr[tr["title"].str.contains(HTML_TAG)][["title", "clean", "label"]],
    }
    return report, tables
