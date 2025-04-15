#!/bin/bash

# Aktifkan environment (ganti path sesuai lokasi env kamu)
source .env/bin/activate  

echo "=== Menu Deteksi Drone ==="
echo "1. Deteksi real-time (tanpa rekam)"
echo "2. Deteksi real-time + rekam"
echo "3. Rekam saja (tanpa deteksi)"
echo "4. Keluar"
read -p "Pilih menu (1/2/3/4): " pilihan

case $pilihan in
  1)
    echo "Menjalankan deteksi real-time..."
    python realtime.py
    ;;
  2)
    echo "Menjalankan deteksi + rekam..."
    python record.py
    ;;
  3)
    echo "Menjalankan perekaman video..."
    python no-model.py
    ;;
  4)
    echo "Keluar..."
    deactivate  # Keluar dari virtual environment
    exit 0
    ;;
  *)
    echo "Pilihan tidak valid."
    ;;
esac
