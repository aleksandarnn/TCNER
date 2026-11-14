# Data: DBLP paper titles → venue

Three tab-separated text files, provided for the *Text Classification* (TCNER) assignment of
the University of Twente IENLP course and reused in the XAI course project:

| file | rows | content |
|---|---|---|
| `DBLPTrainset.txt` | 21,643 | `<venue>\t<title>` |
| `DBLPTestset.txt` | 3,373 | `<title>` |
| `DBLPTestGroundTruth.txt` | 3,373 | `<venue>` (same order as the test titles) |

Venues: **INFOCOM** (computer networking), **ISCAS** (circuits & systems), **SIGGRAPH**
(computer graphics), **VLDB** (databases), **WWW** (the Web).

The titles come from the [dblp computer science bibliography](https://dblp.org), whose
metadata is released under [CC0 1.0](https://dblp.org/db/about/copyright.html). The
train/test split itself was made by the course staff.

Known quirks (quantified in `notebooks/01_data_model_audit.ipynb`):

* the venue mix differs strongly between train and test (ISCAS ≈ 35% of train, ≈ 61% of test);
* the training set contains duplicate titles, and a few titles appear in both splits;
* some entries are proceedings boilerplate (*Front Matter*, *Keynote*, *Panel session*, …)
  rather than papers.
