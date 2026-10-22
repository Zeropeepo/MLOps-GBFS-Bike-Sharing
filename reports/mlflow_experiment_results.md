# Hasil Eksperimen MLflow

Experiment: `GBFS Bike Availability Forecasting`
Git revision: `15186de5deb72469d21fcfadfa88feba8f172bdf`
Git worktree dirty: `true`
Training script SHA-256: `fb265a48728a0121270c8a5adc85d43f9ca389885a3c5e44df67eeed4bfd3b78`
Feature table SHA-256: `005a8b2607e5f434e4c0d285324e5ae43d7dbe6d584302b1a632e55f274b27b0`
Pointer DVC SHA-256: `0a360bb41383b840b38e44bfc18a1a284e955b57f63ddab1ebfc35dac04cb9d2`
Train rows: 7452; validation rows: 2484.

| Run ID | Hyperparameters | Bikes RMSE | Docks RMSE | Mean RMSE |
| --- | --- | ---: | ---: | ---: |
| `7b033a8642ec43bf9846ba46a8b1e315` | `{'n_estimators': 100, 'max_depth': 10, 'min_samples_leaf': 2}` | 3.7317 | 3.6008 | 3.6663 |
| `7a1a6c9ab27a44378c36b69a6072f9e3` | `{'n_estimators': 200, 'max_depth': 16, 'min_samples_leaf': 2}` | 3.4491 | 3.3028 | 3.3760 |
| `1ec2b1befb3a48f69e5f7a3602a6e911` | `{'n_estimators': 300, 'max_depth': None, 'min_samples_leaf': 4}` | 4.0190 | 3.9147 | 3.9669 |

## Model terpilih

- Nama registry: `GBFSBikeAvailabilityRegressor`
- Versi registry: `3`
- Run ID: `7a1a6c9ab27a44378c36b69a6072f9e3`
- Parameter: `{'n_estimators': 200, 'max_depth': 16, 'min_samples_leaf': 2}`
- MAE bikes: 1.7105
- RMSE bikes: 3.4491
- MAE docks: 1.6799
- RMSE docks: 3.3028
- Mean RMSE used for selection: 3.3760

Validation memakai waktu observasi terbaru dan tidak diacak. Perbandingan ini merupakan eksperimen awal; evaluasi deployment perlu periode test yang terpisah.
