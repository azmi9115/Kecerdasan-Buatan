import cv2
import numpy as np

# Buka webcam (kamera default)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize untuk efisiensi
    frame = cv2.resize(frame, (640, 480))

    # Buat mask hitam seluruhnya, lalu putihkan bagian lantai (misal: bagian bawah 40% gambar)
    mask = np.zeros_like(frame[:, :, 0])  # mask 1 channel (grayscale)
    height = frame.shape[0]
    cv2.rectangle(mask, (0, int(height * 0.6)), (frame.shape[1], height), 255, -1)  # putihkan bagian lantai

    # Ubah ke grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Terapkan masking hanya pada lantai
    gray_floor = cv2.bitwise_and(gray, gray, mask=mask)

    # Blur & edge detection
    blur = cv2.GaussianBlur(gray_floor, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Deteksi garis lurus
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

    # Gambar garis pada frame
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Tampilkan hasil
    cv2.imshow('Deteksi Garis Lantai', frame)
    cv2.imshow('Edge (Hanya Lantai)', edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan
cap.release()
cv2.destroyAllWindows()
