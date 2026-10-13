# January 2021 spatial validation

The [metrics table](results/chromap_2021_01_held_out_metrics.csv) contains
CHROMAP-adaptation results only. The public README shows the Netherlands
subset because its ground-measurement figures show Dutch stations only.
All-training-station scores additionally include eligible neighbouring-country
EEA sites.

For each pollutant, station coordinates in EPSG:3035 were assigned to five
geographic K-means groups (`random_state=42`, `n_init=50`). Each group was
predicted by models fitted and preprocessed on the other four groups. Thus no
station's own monthly target entered its held-out prediction. RMSE, MAE and
bias were calculated from those out-of-fold predictions; bias is prediction
minus observation. A fold is a spatial block, not an independently collected
external test dataset.

The final map was fitted to **all** eligible training stations. Overlaying
those same stations on the map shows geographic coverage and concentration
patterns; sampling the final map at those stations would measure in-sample fit,
not held-out accuracy. Neighbouring-country training sites help constrain the
Dutch map, and Dutch-only scores are a subset of the cross-validated rows.

The five maps have different predictor stacks: NO₂ uses the documented
22-feature paper stack, whereas the other four use the project-prepared
monthly stack. SO₂ uses a project-defined four-model preset because the
supplied CHROMAP archive did not include an SO₂ preset. Do not interpret
between-pollutant score differences as a direct model ranking; concentration
scales and station counts differ. January 2021 results do not establish
performance for Germany or any other month.
