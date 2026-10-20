"""
Simulasi & Pemrosesan Paralel Data Kualitas Udara Lokal (PM2.5) dengan HPC
==========================================================================
Alur kerja:
  1. Muat data log sensor PM2.5 lokal.
  2. Untuk setiap pengamatan, jalankan simulasi penyebaran partikel
     (Monte Carlo random walk: adveksi angin + difusi turbulen).
  3. Jalankan secara SEKUENSIAL (1 core) lalu PARALEL (multi-core).
  4. Bandingkan waktu eksekusi: speedup dan efisiensi paralel.
  5. Cetak ringkasan prediksi risiko polusi per wilayah.

Skrip ini sadar-Slurm: jumlah core dibaca otomatis dari SLURM_CPUS_PER_TASK,
sehingga kode yang sama jalan di laptop, kluster HPC lokal, maupun cloud.

Catatan: model fisika di sini disederhanakan untuk tujuan demonstrasi
komputasi, bukan model dispersi atmosfer tervalidasi.
"""

# ---------------------------------------------------------------
# 1. IMPORT LIBRARY
# ---------------------------------------------------------------
import argparse
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

# ---------------------------------------------------------------
# 2. KONSTANTA MODEL
# ---------------------------------------------------------------
WAKTU_SIMULASI_DETIK = 300   # horizon simulasi: 5 menit
JARAK_RESEPTOR_M = 400       # jarak reseptor (mis. sekolah/permukiman) searah angin
RADIUS_RESEPTOR_M = 250      # radius zona reseptor
SEED_DASAR = 2025            # seed dasar agar hasil dapat direproduksi

# Kelas risiko mengacu pada ambang batas ISPU PM2.5 (ug/m3, rata-rata 24 jam).
# Sesuaikan dengan regulasi terbaru bila dipakai untuk keperluan nyata.
TINDAKAN = {
    "Baik": "Aktivitas luar ruangan normal.",
    "Sedang": "Kelompok sensitif (anak, lansia, penderita asma) kurangi aktivitas berat di luar.",
    "Tidak Sehat": "Batasi aktivitas luar ruangan; pertimbangkan olahraga sekolah dipindah ke dalam ruangan.",
    "Sangat Tidak Sehat": "Hentikan aktivitas luar ruangan; pertimbangkan penutupan/pembelajaran jarak jauh sekolah.",
    "Berbahaya": "Status darurat: tutup sekolah, pembatasan aktivitas publik, bagikan masker.",
}


def kategori_risiko(pm25: float) -> str:
    """Klasifikasi risiko berdasarkan konsentrasi PM2.5 (ug/m3)."""
    if pm25 <= 15.5:
        return "Baik"
    if pm25 <= 55.4:
        return "Sedang"
    if pm25 <= 150.4:
        return "Tidak Sehat"
    if pm25 <= 250.4:
        return "Sangat Tidak Sehat"
    return "Berbahaya"


# ---------------------------------------------------------------
# 3. FUNGSI BEBAN KOMPUTASI BERAT: SIMULASI PENYEBARAN PARTIKEL
# ---------------------------------------------------------------
def simulasi_penyebaran_partikel(tugas):
    """
    Simulasi Monte Carlo penyebaran partikel PM2.5 dari satu sumber.

    Fisika sederhana:
      - Adveksi : partikel terbawa angin, x += u * dt
      - Difusi  : lompatan acak turbulen, sigma = sqrt(2 * K * dt)
      - K (difusivitas eddy) naik seiring kecepatan angin
    Keluaran: fraksi partikel yang mencapai zona reseptor, lalu dipakai
    untuk memproyeksikan PM2.5 yang dirasakan masyarakat.
    """
    idx, rec, n_partikel, n_langkah = tugas

    # RNG independen per tugas -> hasil identik di mode sekuensial maupun paralel
    rng = np.random.default_rng(SEED_DASAR + idx)

    u = rec["Kecepatan_Angin"] / 3.6          # km/h -> m/s
    dt = WAKTU_SIMULASI_DETIK / n_langkah     # langkah waktu (detik)
    K = 5.0 + 4.0 * u                         # difusivitas turbulen (m2/s)
    sigma = np.sqrt(2.0 * K * dt)             # simpangan acak per langkah (m)

    x = np.zeros(n_partikel)
    y = np.zeros(n_partikel)

    # Loop waktu: bagian inilah yang memakan CPU
    for _ in range(n_langkah):
        x += u * dt + sigma * rng.standard_normal(n_partikel)
        y += sigma * rng.standard_normal(n_partikel)

    # Fraksi partikel yang jatuh di dalam zona reseptor
    jarak2 = (x - JARAK_RESEPTOR_M) ** 2 + y ** 2
    fraksi = float(np.mean(jarak2 <= RADIUS_RESEPTOR_M ** 2))

    # Proyeksi PM2.5: observasi dikoreksi aktivitas, kelembapan, dan paparan
    faktor_emisi = 0.85 + 0.30 * (rec["Indeks_Aktivitas_Lokal"] / 100.0)
    faktor_lembap = 1.0 + 0.004 * max(0.0, rec["Kelembapan"] - 60.0)
    faktor_paparan = 1.0 + 0.25 * fraksi
    pm25_proyeksi = rec["Konsentrasi_PM25"] * faktor_emisi * faktor_lembap * faktor_paparan

    return {
        "Stasiun_ID": rec["Stasiun_ID"],
        "Wilayah_Lokal": rec["Wilayah_Lokal"],
        "Tanggal_Waktu": rec["Tanggal_Waktu"],
        "PM25_Observasi": rec["Konsentrasi_PM25"],
        "Fraksi_Mencapai_Reseptor": fraksi,
        "PM25_Proyeksi": pm25_proyeksi,
    }


# ---------------------------------------------------------------
# 4. EKSEKUSI SEKUENSIAL vs PARALEL
# ---------------------------------------------------------------
def jalankan_sekuensial(daftar_tugas):
    """Satu core: tugas diproses satu per satu."""
    t0 = time.perf_counter()
    hasil = [simulasi_penyebaran_partikel(t) for t in daftar_tugas]
    return hasil, time.perf_counter() - t0


def jalankan_paralel(daftar_tugas, n_worker):
    """Multi-core: tugas dibagi ke n_worker proses (pola data-parallel)."""
    t0 = time.perf_counter()
    with Pool(processes=n_worker) as pool:
        # chunksize=1 -> pembagian beban dinamis, tiap worker ambil tugas berikutnya
        hasil = pool.map(simulasi_penyebaran_partikel, daftar_tugas, chunksize=1)
    return hasil, time.perf_counter() - t0


def core_tersedia():
    """Deteksi core: hormati alokasi Slurm bila dijalankan di kluster."""
    if "SLURM_CPUS_PER_TASK" in os.environ:
        return int(os.environ["SLURM_CPUS_PER_TASK"])
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def daftar_jumlah_worker(maks):
    """Jumlah worker yang diuji: 2, 4, 8, ... hingga core maksimum."""
    pilihan = {w for w in (2, 4, 8, 16, 32, 64, 128) if w <= maks}
    if maks > 1:
        pilihan.add(maks)
    return sorted(pilihan)


# ---------------------------------------------------------------
# 5. PROGRAM UTAMA
# ---------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Simulasi PM2.5 paralel (HPC)")
    parser.add_argument("--data", default="data_sensor_pm25_lokal.csv")
    parser.add_argument("--particles", type=int, default=150_000,
                        help="jumlah partikel per simulasi")
    parser.add_argument("--steps", type=int, default=100,
                        help="jumlah langkah waktu per simulasi")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="batas core yang diuji (default: semua core tersedia)")
    args = parser.parse_args()

    maks_core = args.max_workers or core_tersedia()

    # --- Muat dataset ---
    df = pd.read_csv(args.data, parse_dates=["Tanggal_Waktu"])
    print("=" * 62)
    print(" SIMULASI PARALEL KUALITAS UDARA LOKAL (PM2.5) - HPC")
    print("=" * 62)
    print(f"Data           : {len(df)} pengamatan dari {df['Stasiun_ID'].nunique()} stasiun")
    print(f"Beban per tugas: {args.particles:,} partikel x {args.steps} langkah")
    print(f"Core tersedia  : {maks_core}\n")

    daftar_tugas = [
        (i, rec, args.particles, args.steps)
        for i, rec in enumerate(df.to_dict("records"))
    ]

    # --- Benchmark ---
    print(">> Menjalankan SEKUENSIAL (1 core) ...")
    hasil_seq, t_seq = jalankan_sekuensial(daftar_tugas)
    print(f"   Selesai dalam {t_seq:.2f} detik")

    baris_benchmark = [{"Mode": "Sekuensial", "Core": 1, "Waktu_detik": t_seq,
                        "Speedup": 1.0, "Efisiensi_%": 100.0}]

    for n in daftar_jumlah_worker(maks_core):
        print(f">> Menjalankan PARALEL ({n} core) ...")
        hasil_par, t_par = jalankan_paralel(daftar_tugas, n)

        # Verifikasi: hasil paralel harus identik dengan sekuensial
        pm_seq = np.array([h["PM25_Proyeksi"] for h in hasil_seq])
        pm_par = np.array([h["PM25_Proyeksi"] for h in hasil_par])
        assert np.allclose(pm_seq, pm_par), "Hasil paralel berbeda dari sekuensial!"

        speedup = t_seq / t_par
        baris_benchmark.append({"Mode": "Paralel", "Core": n, "Waktu_detik": t_par,
                                "Speedup": speedup, "Efisiensi_%": speedup / n * 100})
        print(f"   Selesai dalam {t_par:.2f} detik (speedup {speedup:.2f}x)")

    benchmark = pd.DataFrame(baris_benchmark)
    print("\n=== HASIL BENCHMARK ===")
    print(benchmark.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("Speedup = T_sekuensial / T_paralel | Efisiensi = Speedup / jumlah core")
    benchmark.to_csv("hasil_benchmark.csv", index=False)

    # --- Ringkasan prediksi risiko per wilayah ---
    hasil = pd.DataFrame(hasil_seq)
    hasil["Kategori_Risiko"] = hasil["PM25_Proyeksi"].apply(kategori_risiko)
    tidak_sehat_ke_atas = ["Tidak Sehat", "Sangat Tidak Sehat", "Berbahaya"]

    ringkasan = hasil.groupby("Wilayah_Lokal").agg(
        Rata2_PM25_Observasi=("PM25_Observasi", "mean"),
        Rata2_PM25_Proyeksi=("PM25_Proyeksi", "mean"),
        Puncak_PM25_Proyeksi=("PM25_Proyeksi", "max"),
        Pengamatan_Tidak_Sehat=("Kategori_Risiko",
                                lambda s: int(s.isin(tidak_sehat_ke_atas).sum())),
        Total_Pengamatan=("Kategori_Risiko", "count"),
    )
    ringkasan["Kategori_Wilayah"] = ringkasan["Rata2_PM25_Proyeksi"].apply(kategori_risiko)
    ringkasan = ringkasan.sort_values("Rata2_PM25_Proyeksi", ascending=False)

    print("\n=== RINGKASAN PREDIKSI RISIKO POLUSI PER WILAYAH ===")
    print(ringkasan.to_string(float_format=lambda v: f"{v:.1f}"))

    print("\n=== REKOMENDASI TINDAKAN PER WILAYAH ===")
    for wilayah, baris in ringkasan.iterrows():
        print(f"- {wilayah} [{baris['Kategori_Wilayah']}]: {TINDAKAN[baris['Kategori_Wilayah']]}")

    ringkasan.to_csv("hasil_risiko_per_wilayah.csv")
    print("\nFile keluaran: hasil_benchmark.csv, hasil_risiko_per_wilayah.csv")


# Guard ini WAJIB agar multiprocessing aman di Windows/macOS
if __name__ == "__main__":
    main()
