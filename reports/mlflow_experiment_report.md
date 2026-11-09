# Laporan Training MLflow dari Artifact GitHub Actions

## Sumber data

Artifact diambil dari GitHub Actions run [39779055751](https://github.com/Zeropeepo/MLOps-GBFS-Bike-Sharing/actions/runs/39779055751), berstatus sukses pada branch `main`. Artifact bernama `gbfs-snapshot-39779055751` memuat pasangan feed GBFS untuk snapshot UTC `2026-10-22T12:31:40.978848Z` serta manifest capture.

Pasangan file snapshot ditambahkan ke `data/raw/`, lalu feature table dibangun ulang menggunakan `src/preprocess.py`. Hasilnya:

| Pemeriksaan | Sebelum artifact | Setelah artifact |
| --- | ---: | ---: |
| Baris feature table | 29.786 | 32.270 |
| Baris dengan kedua target 30 menit | 7.452 | 9.936 |
| Penambahan baris berlabel | — | 2.484 |

Preprocessing mencatat 13 snapshot lengkap, 32.270 baris dipertahankan, dan 479 baris ditolak. File sumber, manifest, dan salinan artifact yang diunduh tersedia di workspace untuk audit.

## Tracking dataset dengan DVC

Pointer DVC diperbarui setelah preprocessing. `dvc status` melaporkan data dan pipeline up to date.

| Pointer | Directory hash | Ukuran |
| --- | --- | ---: |
| `data/raw.dvc` | `dde211cf74f3163b60922999bf33ac19.dir` | 25.316.393 bytes |
| `data/processed.dvc` | `cf62315487830a1c9775b568f5c82f68.dir` | 7.861.741 bytes |

## Rancangan training

Skrip `src/modeling/train.py` melatih tiga `RandomForestRegressor` untuk target jumlah sepeda tersedia dan dock kosong sekitar 30 menit ke depan. Data dibagi secara kronologis tanpa shuffling. Dari 9.936 baris berlabel, 7.452 baris dipakai untuk training dan 2.484 untuk validation. Training berakhir pada `2026-10-22T12:34:14.770Z`; validation dimulai pada `2026-10-22T12:39:14.770Z`.

Imputasi dan one-hot encoding berada di dalam pipeline, sehingga preprocessing di-fit hanya pada training set. Pemilihan model memakai mean RMSE dari dua target; nilai lebih rendah lebih baik.

## Hasil eksperimen

Semua run tersimpan di experiment `GBFS Bike Availability Forecasting` pada MLflow lokal dan berstatus `FINISHED`.

| Run | Parameter (`n_estimators`, `max_depth`, `min_samples_leaf`) | Bikes MAE | Bikes RMSE | Docks MAE | Docks RMSE | Mean RMSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `random-forest-1` | `100, 10, 2` | 2.0237 | 3.7317 | 2.0156 | 3.6008 | 3.6663 |
| `random-forest-2` | `200, 16, 2` | **1.7105** | **3.4491** | **1.6799** | **3.3028** | **3.3760** |
| `random-forest-3` | `300, None, 4` | 2.0782 | 4.0190 | 2.0544 | 3.9147 | 3.9669 |

### Model terpilih

- **Registry:** `GBFSBikeAvailabilityRegressor`, versi `3`, status `READY`
- **Run ID:** `7a1a6c9ab27a44378c36b69a6072f9e3`
- **Parameter:** `n_estimators=200`, `max_depth=16`, `min_samples_leaf=2`, `random_state=42`
- **Validation:** bikes R² `0.9240`; docks R² `0.9362`; mean RMSE `3.3760`

Run MLflow diberi tag `github_actions_run_id=39779055751`, URL run, nama artifact, dan waktu snapshot untuk menghubungkan eksperimen dengan sumber datanya. Database MLflow dan model artifacts tersimpan lokal di workspace, bukan di cloud.

## Membuka dashboard

Saat server MLflow berjalan, buka `http://127.0.0.1:5000` dan pilih experiment `GBFS Bike Availability Forecasting`. Jalankan server dari root repository bila belum aktif:

```bash
source .venv/bin/activate
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

## Batas evaluasi

Walau validation berisi 2.484 baris stasiun, rentang waktunya hanya satu kelompok observasi terbaru. Hasil ini berguna sebagai demonstrasi alur artifact → preprocessing → DVC → training → MLflow, tetapi belum cukup untuk menyimpulkan performa pada hari atau periode lain. Simpan screenshot halaman experiment MLflow sebagai bukti visual jika format pengumpulan mewajibkannya.
