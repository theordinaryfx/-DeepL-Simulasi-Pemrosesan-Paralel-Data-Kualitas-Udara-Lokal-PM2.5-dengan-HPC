# 🌫️ Simulasi & Pemrosesan Paralel Data Kualitas Udara Lokal (PM2.5) dengan Arsitektur HPC

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Parallel](https://img.shields.io/badge/Parallel-multiprocessing-orange)
![Scheduler](https://img.shields.io/badge/Scheduler-Slurm-green)
![Domain](https://img.shields.io/badge/Domain-Environmental%20HPC-lightgrey)

Proyek portofolio teknik komputasi yang mensimulasikan bagaimana **High-Performance Computing (HPC)** mempercepat analisis data kualitas udara dari jaringan sensor lokal. Setiap pengamatan PM2.5 diproses dengan simulasi penyebaran partikel (Monte Carlo), dijalankan **sekuensial vs paralel**, lalu dirangkum menjadi **prediksi risiko polusi per wilayah**.

Proyek ini terinspirasi oleh konsep **"Bridging Global Infrastructure to Address Local Challenge in HPC"** (Asst. Prof. Dr. Worawan Diaz Carballo): infrastruktur komputasi global dimanfaatkan untuk menyelesaikan masalah lingkungan dan masyarakat di tingkat lokal.

> ⚠️ **Catatan:** Data adalah **data dummy** dan model fisika **disederhanakan** untuk demonstrasi komputasi. Hasilnya bukan dasar keputusan kesehatan publik yang sesungguhnya.

---

## 📑 Daftar Isi
1. [Background & Local Challenge](#1--background--local-challenge)
2. [Global Infrastructure Integration](#2--global-infrastructure-integration)
3. [Contoh Script Job Scheduler (Slurm)](#3--contoh-script-job-scheduler-slurm)
4. [Benchmarking Results](#4--benchmarking-results)
5. [Impact & Integrator Role](#5--impact--integrator-role)
6. [Cara Menjalankan Kode](#6--cara-menjalankan-kode)
7. [Struktur Repositori](#7--struktur-repositori)

---

## 1. 🏘️ Background & Local Challenge

Polusi udara partikulat **PM2.5** berukuran sangat kecil sehingga dapat masuk jauh ke saluran pernapasan. Dampaknya terasa langsung di tingkat lokal: anak sekolah, lansia, pekerja luar ruangan, dan warga di sekitar kawasan industri atau lalu lintas padat.

**Tantangannya adalah waktu.** Jaringan sensor IoT menghasilkan data terus-menerus, sementara simulasi penyebaran partikel per pengamatan memerlukan ribuan hingga jutaan perhitungan. Bila diproses di satu komputer secara berurutan, hasil bisa terlambat dari kebutuhan keputusan.

**Ilustrasi skala (hipotetis):**

| Skenario | Perhitungan |
|---|---|
| 1.000.000 simulasi × 0,4 detik, 1 core | ≈ 400.000 detik ≈ **4,6 hari** |
| Sama, 256 core, efisiensi paralel 85% | ≈ 1.840 detik ≈ **31 menit** |

Angka di atas adalah ilustrasi untuk menjelaskan prinsip. Hasil nyata dapat Anda ukur sendiri di bagian [Benchmarking](#4--benchmarking-results).

> 💡 **Catatan akurasi:** Percepatan paralel secara teori **maksimal linear** terhadap jumlah core (dibatasi *Amdahl's Law*), bukan eksponensial.

---

## 2. 🌐 Global Infrastructure Integration

Kode dirancang **portabel** agar dapat berjalan di berbagai tingkat infrastruktur tanpa perubahan:

| Lingkungan | Cara kerja |
|---|---|
| 💻 **Laptop/PC** | Mendeteksi jumlah core secara otomatis |
| 🏫 **Kluster HPC kampus (Slurm)** | Membaca `SLURM_CPUS_PER_TASK`, sehingga tepat memakai alokasi core yang diminta |
| ☁️ **Cloud (VM multi-core / kluster Slurm cloud)** | Pola *data-parallel* yang sama, tinggal menaikkan jumlah core |

**Prinsip desain:**
- **Embarrassingly parallel** — setiap pengamatan sensor independen, sehingga mudah dibagi ke banyak core.
- **Reprodusibel** — seed acak per tugas membuat hasil sekuensial dan paralel identik (diverifikasi otomatis lewat `assert`).
- **Parameterisasi** — beban kerja diatur lewat argumen (`--particles`, `--steps`, `--max-workers`).
- **Skalabel** — untuk lintas node, pola yang sama dapat dikembangkan dengan MPI (`mpi4py`) atau Slurm job array.

---

## 3. 📜 Contoh Script Job Scheduler (Slurm)

Simpan sebagai `job_slurm.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=pm25_hpc
#SBATCH --partition=compute          # sesuaikan nama partisi di kluster Anda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16           # jumlah core yang diminta
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/pm25_%j.out
#SBATCH --error=logs/pm25_%j.err

# Muat lingkungan (sesuaikan dengan modul di kluster Anda)
module load python/3.10
source venv/bin/activate

# Hindari oversubscription thread dari library numerik
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Jalankan simulasi dengan beban lebih berat
python hpc_parallel_forecasting.py --particles 500000 --steps 200
```

Pengajuan job:

```bash
mkdir -p logs
sbatch job_slurm.sh      # ajukan job
squeue -u $USER          # pantau antrean
cat logs/pm25_<JOBID>.out  # lihat hasil
```

---

## 4. ⏱️ Benchmarking Results

Skrip otomatis menjalankan mode sekuensial lalu paralel dengan 2, 4, 8, ... core, kemudian menyimpan hasilnya ke `hasil_benchmark.csv`.

**Metrik:**
- **Speedup** = T<sub>sekuensial</sub> / T<sub>paralel</sub>
- **Efisiensi** = Speedup / jumlah core × 100%

**Hasil di mesin saya** (isi dari output Anda, sebutkan spesifikasi CPU):

| Mode | Core | Waktu (detik) | Speedup | Efisiensi |
|---|---|---|---|---|
| Sekuensial | 1 | … | 1.00× | 100% |
| Paralel | 2 | … | …× | …% |
| Paralel | 4 | … | …× | …% |
| Paralel | 8 | … | …× | …% |

**Cara membaca hasil:**
- Speedup mendekati jumlah core berarti beban sangat paralel.
- Efisiensi menurun pada core banyak karena *overhead* pembuatan proses, pembagian tugas, dan bagian program yang tetap serial (*Amdahl's Law*).
- Dengan hanya 30 tugas, jumlah core di atas ±30 tidak lagi menambah kecepatan. Naikkan `--particles` atau perbanyak data untuk menguji skalabilitas.

---

## 5. 🎯 Impact & Integrator Role

Peran **integrator** adalah menjembatani tiga hal: **data lokal** (sensor), **daya komputasi** (HPC), dan **keputusan masyarakat**.

Keluaran skrip mengklasifikasikan wilayah ke kategori risiko dan memberi rekomendasi:

| Kategori | Contoh tindakan bagi pengambil keputusan |
|---|---|
| 🟢 Baik | Aktivitas luar ruangan normal |
| 🟡 Sedang | Kelompok sensitif mengurangi aktivitas berat di luar |
| 🟠 Tidak Sehat | Batasi aktivitas luar ruangan; olahraga sekolah dipindah ke dalam ruangan |
| 🔴 Sangat Tidak Sehat | Pertimbangkan **penutupan sekolah** atau pembelajaran jarak jauh; pembatasan aktivitas luar ruangan |
| ⚫ Berbahaya | Status darurat, tutup sekolah, pembagian masker, pembatasan aktivitas publik |

**Siapa yang terbantu:** dinas lingkungan hidup, dinas pendidikan dan kesehatan, pengelola kawasan industri, serta perangkat desa/kelurahan.

**Nilai tambah HPC:** keputusan dapat didukung data yang **cukup cepat** (hitungan menit, bukan hari), sehingga peringatan dini benar-benar bisa ditindaklanjuti.

---

## 6. ▶️ Cara Menjalankan Kode

```bash
# 1. Clone repositori
git clone https://github.com/<username>/<nama-repositori>.git
cd <nama-repositori>

# 2. (Opsional) Buat virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Instal dependensi
pip install -r requirements.txt

# 4. Jalankan dengan pengaturan bawaan
python hpc_parallel_forecasting.py

# 5. Atau atur beban kerja & batas core
python hpc_parallel_forecasting.py --particles 300000 --steps 150 --max-workers 8
```

**Keluaran:** tabel benchmark, ringkasan risiko per wilayah, rekomendasi tindakan, serta file `hasil_benchmark.csv` dan `hasil_risiko_per_wilayah.csv`.

---

## 7. 📁 Struktur Repositori

```
├── data_sensor_pm25_lokal.csv     # Data dummy sensor (30 baris)
├── hpc_parallel_forecasting.py    # Simulasi & benchmark paralel
├── job_slurm.sh                   # (opsional) contoh job Slurm
├── requirements.txt               # Dependensi Python
└── README.md                      # Dokumentasi proyek
```

---

## 🚀 Pengembangan Lanjutan
- Skala lintas node dengan **MPI (`mpi4py`)** atau **Slurm job array**.
- Percepatan dengan **Numba** atau **GPU (CuPy)**.
- Ganti model sederhana dengan model dispersi tervalidasi (mis. Gaussian plume penuh).
- Integrasi data sensor real-time (MQTT/API) dan dashboard peringatan dini.
