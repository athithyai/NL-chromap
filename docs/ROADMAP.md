# Roadmap

The `main` branch documents one January 2021 five-pollutant CHROMAP-style
experiment. It is not a complete historical product.

1. Reconcile the predictor availability and definitions for each year from
   2013 onward, especially satellite NO₂ and static layers.
2. Compare independent monthly training with pooled multi-month training
   using the same geographic held-out evaluation. Pooling could increase
   sample size and robustness, but it adds a time dimension to the model.
3. Generalise Dutch-only acquisition and feature scripts to parameterised
   study boundaries, dates and station sources.
4. Build all requested NO₂, O₃, PM₁₀, PM₂.₅ and SO₂ months; validate each
   release before making claims beyond the January pilot.
5. Publish data products separately from this code repository, subject to
   provider licences, with manifests and checksums.
