# Review of supplied CHROMAP v1.0 archive

The supplied archive is a compact annual-map reference implementation. Its intended flow is:

1. extract predictors at stations and across a pre-aligned grid;
2. fit Ridge, Random Forest, XGBoost, and MLP regressors;
3. diagnose feature importance;
4. execute selected models across the grid; and
5. publish the median ensemble plus ensemble-spread uncertainty.

The Netherlands redevelopment preserves that conceptual ensemble, predictor extraction, interpretable-model diagnostics, and uncertainty reporting. It does not execute the archive unchanged because the requested product is monthly, must handle hourly/daily completeness and SOMO35, and needs a frozen Netherlands grid and source manifests.

## Issues to correct in the redevelopment

- The reference cross-validation is stratified by station type but is not spatially blocked. The Netherlands pipeline uses grouped spatial folds so nearby sites cannot leak spatial information into validation.
- Hyperparameters are hard-coded by pollutant after an optional annual search. Monthly models require a documented pilot comparison between pooled seasonal, pollutant-specific, and month-specific schemes.
- In `Model_creation.py`, the saved `Pred_RF_vc` column is populated from the neural-network predictions rather than the random-forest predictions. This must not be copied.
- In `Feature_evaluation.py`, the SHAP section refers to undefined names (`modele` and `poll_name`) instead of the loaded model and pollutant identifier. This must be corrected before SHAP diagnostics are enabled.
- The reference clips only predictions below the observed minimum and does not symmetrically enforce the observed maximum despite calculating it. Production constraints need indicator-specific, documented handling rather than accidental clipping.
- The archive assumes all predictors are already pre-aligned `.npy` arrays and contains no acquisition/version provenance. The redevelopment records URLs, request payloads, retrieval timestamps, licences, and SHA-256 hashes.
- Annual observation input cannot produce monthly SOMO35 directly. The redevelopment calculates it from hourly O3 with cross-month 8-hour windows and explicit completeness rules.

These findings make the archive useful as an algorithmic reference, but not as a reproducible turnkey input for the requested Netherlands monthly product.
