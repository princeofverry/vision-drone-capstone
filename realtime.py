from ultralytics import YOLO
import cv2

# Muat model YOLOv8
model = YOLO('best-1.pt')  # Ganti dengan path ke model .pt kamu

# Fungsi untuk deteksi real-time
def detect_realtime():
    cap = cv2.VideoCapture(0)  # Menggunakan kamera default (biasanya kamera laptop)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Lakukan inferensi pada frame
        results = model(frame)  # YOLOv8 memakai cara ini untuk inferensi
        
        # Ambil hasil deteksi dan gambar dengan kotak pembatas
        annotated_frame = results[0].plot()  # Menggunakan plot untuk menambahkan kotak pembatas ke frame
        
        # Tampilkan hasil deteksi dengan OpenCV
        cv2.imshow("Detected Objects", annotated_frame)
        
        # Tekan 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Panggil fungsi deteksi real-time
detect_realtime()
