# January 2021 NO₂ paired spatial validation

## Decision

For January 2021 NO₂, the NL CAMS-plus-residual method is the supported choice.
It achieved the lowest held-out RMSE in every geographic fold and overall.

| Method | RMSE | MAE | Bias (prediction − observation) | R² |
|---|---:|---:|---:|---:|
| CAMS baseline | 5.518 | 4.039 | -2.524 | 0.430 |
| CHROMAP-style direct median ensemble | 4.770 | 3.813 | -0.595 | 0.574 |
| NL Extra Trees residual correction | **3.413** | **2.619** | **-0.116** | **0.782** |

The NL method reduced RMSE by 28.4% relative to the CHROMAP-style ensemble and
38.1% relative to CAMS. It reduced MAE by 31.3% relative to CHROMAP-style.

## Fair comparison design

- Population: 258 valid January 2021 NO₂ station rows; one row per station.
- Folds: five geographic K-means station clusters.
- Pairing: all methods predicted exactly the same held-out station rows.
- Leakage control: predictor preprocessing and all model fitting occurred inside
  each training fold.
- CHROMAP-style: direct Ridge, Random Forest, XGBoost, and MLP predictions,
  lower-bound rule, then median ensemble.
- NL method: CAMS plus an Extra Trees prediction of observation-minus-CAMS.
- Selection metric: lowest overall out-of-fold RMSE.

The station-paired descriptive bootstrap estimated the RMSE improvement of NL
over CHROMAP-style at 1.358 µg m⁻³, with a 95% interval of 1.031–1.679. The
interval is descriptive because it does not model residual spatial dependence.

## Caveats

- The decision applies to January 2021 NO₂ only.
- Geographic K-means folds are not fixed-distance exclusion buffers.
- The supplied CHROMAP configuration is an annual method adapted here to monthly
  targets.
- Its MLP reached the 200-iteration limit in all five folds.
- CAMS assimilates monitoring observations, so CAMS-based predictions are not
  entirely independent of surface observations.

Reproduce with `python scripts/validate_no2_2021_01_approaches.py`.
