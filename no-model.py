import cv2

# Fungsi untuk merekam video tanpa model
def record_video():
    # Menggunakan kamera default (biasanya kamera laptop)
    cap = cv2.VideoCapture(0)
    
    # Tentukan codec dan buat VideoWriter untuk menyimpan video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_recorded.avi', fourcc, 20.0, (640, 480))  # Ganti dengan resolusi yang sesuai
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Simpan frame ke file
        out.write(frame)
        
        # Tampilkan video langsung (real-time)
        cv2.imshow('Recording Video', frame)
        
        # Tekan 'q' untuk berhenti merekam
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Panggil fungsi untuk merekam video
record_video()
