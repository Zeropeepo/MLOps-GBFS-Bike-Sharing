# MLOps GBFS Bike-Sharing Forecasting

## Project Overview

Proyek ini bertujuan memprediksi jumlah sepeda tersedia dan slot dock
kosong pada stasiun bike-sharing untuk horizon 30 menit ke depan.

## Machine Learning Task

Task utama adalah time-series forecasting menggunakan data status
stasiun yang diperoleh dari GBFS.

## Data Source

Data berasal dari feed real-time Citi Bike dalam format General
Bikeshare Feed Specification.

## Current Status

Saat ini proyek berada pada tahap penyiapan lingkungan pengembangan,
struktur repository, data ingestion, validasi, dan initial EDA.

## Repository Structure

Jelaskan fungsi folder data, models, notebooks, src, config, dan tests.

## Running with GitHub Codespaces

1. Buka repository.
2. Pilih Code.
3. Pilih Codespaces.
4. Pilih Create codespace on main.
5. Tunggu instalasi dependency selesai.

## Development Checks

```bash
pip check
ruff check .
pytest -q