from ultralytics import YOLO
import cv2

model = YOLO('best-1.pt')  # Ganti dengan path ke model .pt kamu

# Fungsi untuk deteksi dan rekam video
def detect_and_record():
    cap = cv2.VideoCapture(0)  # Menggunakan kamera default (biasanya kamera laptop)
    
    # Tentukan codec dan buat VideoWriter untuk menyimpan video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_without_model.avi', fourcc, 20.0, (640, 480)) 
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Lakukan inferensi pada frame
        results = model(frame)  # YOLOv8 memakai cara ini untuk inferensi
        
        # Ambil hasil deteksi dan gambar dengan kotak pembatas
        annotated_frame = results[0].plot()  # Menggunakan plot untuk menambahkan kotak pembatas ke frame
        
        # Tampilkan hasil deteksi dengan menggunakan OpenCV
        cv2.imshow("Detected Objects", annotated_frame)
        
        # Simpan hasil deteksi ke file output
        out.write(annotated_frame)  # Menyimpan frame yang sudah dianotasi ke file
        
        # Tekan 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Panggil fungsi deteksi dan rekam
detect_and_record()
