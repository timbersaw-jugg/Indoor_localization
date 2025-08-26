# Indoor Localization (RSSI) — CNN-LSTM + AE + K-Fold

End-to-end indoor localization from Bluetooth RSSI. Includes classical baselines
(KNN/SVM/RF), a deep CNN→LSTM classifier (optional autoencoder warm-start), and
k-fold ensembling. Evaluation utilities include a **CDF of localization error**,
confusion matrix, and t-SNE.

## Quickstart
1. Open `notebooks/Indoor_Localization.ipynb` in Jupyter/Colab.
2. Provide `iBeacon_RSSI_Labeled.csv` (or the processed CSV).
3. Run cells in order to train, evaluate, and plot the **CDF of error**.

## CDF of Localization Error
- Curve shows the fraction of samples with error ≤ x.
- **E50/E90/E95** = 50th/90th/95th percentile errors.
- “% ≤ k” = share of predictions within k grid units (or meters if scaled).

## Repo Layout

## Environment
numpy, pandas, scikit-learn, torch, matplotlib, seaborn, plotly.

## Resume Blurb
Built an indoor localization system from Bluetooth RSSI using Conv1D→LSTM with
AE warm-start and k-fold ensemble; added baselines and interpretability via
CDF-of-error. Clean, reproducible repo with notebooks and plots.

License: MIT
