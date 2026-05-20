from flask import Flask, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import numpy as np
import os
import time
import random # <--- TAMBAHAN UNTUK ACAR GAMBAR

app = Flask(__name__)

# --- CONFIG ---
MEME_FOLDER = 'images'
DATA_FOLDER = 'pose_data'

# --- CLUES (Pastikan nama file sama persis) ---
MEME_CLUES = {
    "meme1.jpg": "hol up, let him cook",
    "meme2.jpg": "mikir nyet?",
    "meme3.jpg": "centil ih monyet",
    "meme4.jpg": "MONYET! ngagetin",
    "meme5.jpg": "banyak ide si monyet",
    "meme6.jpg": "niche baby",
    "meme7.jpg": "ash baby",
    "meme8.jpg": "just ran over a h*mo xoxo beep beep b*tch",
    "meme9.jpg": "just ran over a f*ggot☺️ cant wait to k*ll all these g*ys off😘",
    "meme10.png": "the concept of not knowing this referece",
    "meme11.jpg": "i found your hat!",
    "meme12.jpg": "erm... actually",
    "meme13.jpg": "jangan dikasitau",
    "meme14.jpg": "12 hour after my shift",
    "meme15.jpg": "she's so crazzzzzzzzy! Love Her!!!",
}

class GameState:
    def __init__(self):
        # Ambil semua file gambar
        self.all_memes = [f for f in os.listdir(MEME_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        # ACAL URUTANNYA DI SINI
        random.shuffle(self.all_memes) 
        
        self.meme_queue = self.all_memes.copy() # Antrian game
        self.curr_file = self.meme_queue[0] if self.meme_queue else None
        self.idx = 0
        self.score = 0
        self.master_data = None
        self.latest_landmarks = None
        self.load_current_data()

    def load_current_data(self):
        if not self.curr_file: return
        
        npy_path = os.path.join(DATA_FOLDER, self.curr_file + '.npy')
        if os.path.exists(npy_path):
            self.master_data = np.load(npy_path)
        else:
            self.master_data = None

    def next_level(self):
        self.idx += 1
        # Jika antrian habis, acak ulang
        if self.idx >= len(self.meme_queue):
            random.shuffle(self.meme_queue)
            self.idx = 0
            
        self.curr_file = self.meme_queue[self.idx]
        self.load_current_data()
        self.score = 0

game = GameState()

# --- MEDIAPIPE SETUP ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)

def calculate_score(master_data, user_landmarks):
    if user_landmarks is None or master_data is None: return 0
    
    user_list = [[lm.x, lm.y, lm.z, lm.visibility] for lm in user_landmarks]
    
    def get_angle(a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        ang = np.abs(rad*180.0/np.pi)
        if ang > 180: ang = 360-ang
        return ang

    joints = [[11,13,15], [12,14,16], [13,11,23], [14,12,24], [11,23,25], [12,24,26], [23,25,27], [24,26,28]]
    total_score = 0
    valid_joints = 0
    
    for j in joints:
        m_a, m_b, m_c = master_data[j[0]], master_data[j[1]], master_data[j[2]]
        u_a, u_b, u_c = user_list[j[0]], user_list[j[1]], user_list[j[2]]
        
        if m_a[3] > 0.5 and u_a[3] > 0.5:
            ang_m = get_angle(m_a[:2], m_b[:2], m_c[:2])
            ang_u = get_angle(u_a[:2], u_b[:2], u_c[:2])
            diff = abs(ang_m - ang_u)
            total_score += max(0, 100 - (diff * 2.0))
            valid_joints += 1
            
    if valid_joints == 0: return 0
    return total_score / valid_joints

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1)
        
        # Proses Pose tanpa menggambar (Wajah Bersih)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(img_rgb)
        
        if results.pose_landmarks:
            game.latest_landmarks = results.pose_landmarks.landmark
            
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_state')
def get_state():
    clue = MEME_CLUES.get(game.curr_file, "Tebak Pose Misterius Ini!")
    return jsonify({
        'level': game.idx + 1,
        'clue': clue,
        'image_url': f"/image/{game.curr_file}"
    })

@app.route('/image/<filename>')
def get_image(filename):
    from flask import send_from_directory
    return send_from_directory(MEME_FOLDER, filename)

@app.route('/check_score', methods=['POST'])
def check_score():
    final_score = calculate_score(game.master_data, game.latest_landmarks)
    game.score = final_score
    return jsonify({'score': int(final_score)})

@app.route('/next_level', methods=['POST'])
def next_level_route():
    game.next_level()
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)