import cv2
import mediapipe as mp
import numpy as np
import os
import time  # <-- TAMBAHAN: Untuk menghitung waktu

# FOLDER & CONFIG
MEME_FOLDER = 'images'
DATA_FOLDER = 'pose_data' 
if not os.path.exists(DATA_FOLDER): os.makedirs(DATA_FOLDER)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=2)

meme_files = [f for f in os.listdir(MEME_FOLDER) if f.endswith(('.png', '.jpg', '.jpeg'))]

cap = cv2.VideoCapture(0)
idx = 0

# VARIABEL COUNTDOWN
countdown_mode = False
start_time = 0
countdown_duration = 3  # Detik
saved_message_timer = 0 # Untuk menampilkan teks "TERSIMPAN" sebentar

print("=== MODE PELATIHAN DENGAN TIMER ===")
print("[S] MULAI Hitung Mundur & Simpan")
print("[N] Lanjut Meme Berikutnya")
print("[Q] Keluar")

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Load Meme
    curr_file = meme_files[idx]
    meme_img = cv2.imread(os.path.join(MEME_FOLDER, curr_file))
    
    # Tampilkan Meme Kecil di Pojok
    meme_resized = cv2.resize(meme_img, (200, int(200 * meme_img.shape[0] / meme_img.shape[1])))
    mh, mw, _ = meme_resized.shape
    frame[0:mh, 0:mw] = meme_resized
    
    # Deteksi Pose Kamu
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = pose.process(img_rgb)
    
    # Gambar Skeleton
    if res.pose_landmarks:
        mp_drawing.draw_landmarks(frame, res.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
    # Info Text Standar
    cv2.putText(frame, f"FILE: {curr_file}", (220, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Cek status file
    npy_path = os.path.join(DATA_FOLDER, curr_file + '.npy')
    if os.path.exists(npy_path):
        cv2.putText(frame, "SUDAH ADA DATA", (220, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "BELUM ADA DATA", (220, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # === LOGIKA COUNTDOWN ===
    if countdown_mode:
        elapsed = time.time() - start_time
        remaining = countdown_duration - elapsed
        
        if remaining > 0:
            # Tampilkan Angka Besar (3... 2... 1...)
            text = str(int(remaining) + 1)
            font_scale = 5
            thickness = 5
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            text_x = (w - text_size[0]) // 2
            text_y = (h + text_size[1]) // 2
            
            # Efek shadow biar jelas
            cv2.putText(frame, text, (text_x+5, text_y+5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0,0,0), thickness+5)
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
        else:
            # WAKTU HABIS -> SAATNYA SIMPAN!
            if res.pose_landmarks:
                landmarks = [[lm.x, lm.y, lm.z, lm.visibility] for lm in res.pose_landmarks.landmark]
                np.save(npy_path, landmarks)
                print(f"Disimpan: {curr_file}")
                saved_message_timer = time.time() # Trigger pesan sukses
            else:
                print("Gagal: Badan tidak terlihat!")
            
            countdown_mode = False # Reset timer

    # Tampilkan pesan "TERSIMPAN!" selama 2 detik setelah sukses
    if time.time() - saved_message_timer < 2:
        cv2.putText(frame, "CEKREK! TERSIMPAN", (w//2 - 200, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow('Trainer', frame)
    
    key = cv2.waitKey(1)
    
    # TOMBOL KONTROL
    if key == ord('s'): # START COUNTDOWN
        if not countdown_mode: # Biar ga dobel klik
            countdown_mode = True
            start_time = time.time()
            print("Bersiap dalam 3 detik...")
            
    elif key == ord('n'): # NEXT
        idx = (idx + 1) % len(meme_files)
        countdown_mode = False # Reset timer jika ganti gambar
        
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()