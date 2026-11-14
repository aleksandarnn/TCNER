"""Explainable venue recommendation for research-paper titles (DBLP / TCNER).

The package holds everything the notebooks share, so every notebook explains
*the same* trained model with *the same* preprocessing:

- :mod:`tcner_xai.config`      paths, seeds and constants
- :mod:`tcner_xai.data`        loading, spaCy preprocessing and the data audit
- :mod:`tcner_xai.model`       the TF-IDF + logistic-regression venue classifier
- :mod:`tcner_xai.explain`     local explainers (exact Shapley, LIME, KernelSHAP,
                               linear SHAP, occlusion, Anchors, counterfactuals)
- :mod:`tcner_xai.metrics`     faithfulness / agreement / stability metrics
- :mod:`tcner_xai.calibration` entropy, calibration, temperature scaling,
                               selective prediction
- :mod:`tcner_xai.plotting`    a shared matplotlib style
"""

__version__ = "2.0.0"
