import cv2
import numpy as np
import math
import winsound

def calculate_angle(x1, y1, x2, y2):
    return math.degrees(math.atan2(y2 - y1, x2 - x1))

def get_line_intersection(p1, p2, p3, p4):
    a1 = p2[1] - p1[1]
    b1 = p1[0] - p2[0]
    c1 = a1 * p1[0] + b1 * p1[1]

    a2 = p4[1] - p3[1]
    b2 = p3[0] - p4[0]
    c2 = a2 * p3[0] + b2 * p3[1]

    det = a1 * b2 - a2 * b1
    if det == 0:
        return None
    else:
        x = (b2 * c1 - b1 * c2) / det
        y = (a1 * c2 - a2 * c1) / det
        return int(x), int(y)

def angle_between_lines(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    v1 = np.array([x2 - x1, y2 - y1])
    v2 = np.array([x4 - x3, y4 - y3])

    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)

    dot = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    angle_rad = np.arccos(dot)
    return math.degrees(angle_rad)

cap = cv2.VideoCapture(0)
scale_cm_per_pixel = 100 / 480
batas_jarak_cm = 30

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)
    blur = cv2.GaussianBlur(contrast, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=50, maxLineGap=10)

    alarm_triggered = False
    valid_lines = []
    garis_terdekat_cm = None

    # 🚨 Jika tidak ada garis terdeteksi, aktifkan alarm
    if lines is None:
        alarm_triggered = True
        cv2.putText(frame, "WARNING: TIDAK ADA GARIS TERDETEKSI!", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        for i, line in enumerate(lines):
            x1, y1, x2, y2 = line[0]
            angle = abs(calculate_angle(x1, y1, x2, y2))

            roi = gray[min(y1, y2):max(y1, y2)+1, min(x1, x2):max(x1, x2)+1]
            if roi.size == 0:
                continue
            avg_brightness = np.mean(roi)

            if avg_brightness < 100:
                garis_y = (y1 + y2) // 2
                jarak_pixel = height - garis_y
                jarak_cm = jarak_pixel * scale_cm_per_pixel

                if garis_terdekat_cm is None or jarak_cm < garis_terdekat_cm:
                    garis_terdekat_cm = jarak_cm

                if jarak_cm < batas_jarak_cm:
                    alarm_triggered = True
                    cv2.putText(frame, "WARNING: GARIS DEKAT!", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

                color = (0, 255, 0)
                cv2.line(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{int(jarak_cm)} cm", (x1, y1 - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(frame, f"{int(angle)} deg", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                valid_lines.append(((x1, y1, x2, y2), angle))

        for i in range(len(valid_lines)):
            for j in range(i + 1, len(valid_lines)):
                line1 = valid_lines[i][0]
                line2 = valid_lines[j][0]
                angle_diff = angle_between_lines(line1, line2)
                if 80 <= angle_diff <= 100:
                    x1, y1, x2, y2 = line1
                    x3, y3, x4, y4 = line2
                    titik_potong = get_line_intersection((x1, y1), (x2, y2), (x3, y3), (x4, y4))
                    if titik_potong:
                        cv2.circle(frame, titik_potong, 6, (0, 0, 255), -1)
                        cv2.putText(frame, "Tegak Lurus", (titik_potong[0]+5, titik_potong[1]-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    if alarm_triggered:
        winsound.Beep(1000, 200)

    cv2.imshow('Deteksi Garis dan Tegak Lurus', frame)
    cv2.imshow('Edges', edges)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
