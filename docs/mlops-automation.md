# Otomatisasi training GBFS

Workflow `.github/workflows/mlops-automation.yaml` berjalan setelah push ke `main`
atau pull request milik pemilik repository. Runner `gbfs-local` harus aktif pada
komputer yang menjalankan MinIO dan MLflow. Pull request dari luar repository atau
dari akun lain dilewati agar kode tidak dipercaya tidak berjalan di komputer lokal.

Urutannya adalah pytest, DVC pull, validasi dataset, training, evaluasi metrik,
dan registrasi model yang lolos ke Staging. Jika salah satu tahap gagal, tahap
berikutnya tidak berjalan. Model Production tidak dipindahkan oleh workflow.

Ambang demonstrasi di `config/model_validation.yaml` adalah RMSE maksimal 5 untuk
masing-masing target. Angka ini bersifat sementara karena nilai ambang numerik
tidak ditemukan dalam dokumen LK-01 yang tersedia. Ganti sesuai ketentuan dosen.
Satu timestamp validasi saat ini belum cukup untuk menyimpulkan performa produksi.

Runner memakai GitHub Actions secrets `DVC_ACCESS_KEY_ID` dan
`DVC_SECRET_ACCESS_KEY` untuk mengakses MinIO lokal, dan `MLFLOW_TRACKING_URI`
diatur ke `http://127.0.0.1:5000`. Repository variable `GBFS_BASE_PYTHON` berisi
lokasi Python 3.11 di runner dan dipakai untuk membuat virtual environment.
Runner juga perlu menyediakan `uv` untuk memasang dependensi proyek.
Karena penyimpanan berada di komputer lokal,
komputer, runner, MinIO, dan MLflow harus menyala agar workflow selesai.
`dvc pull` mengambil versi data sesuai pointer di commit pemicu, bukan menarik
data baru dari feed GBFS. Pengumpulan snapshot dan pembaruan pointer dilakukan
secara terpisah sebelum commit.

Simulasi perubahan kode dilakukan dengan mengubah `n_estimators` konfigurasi
pertama dari 100 menjadi 110. Commit tersebut memicu ulang workflow pada PR.
