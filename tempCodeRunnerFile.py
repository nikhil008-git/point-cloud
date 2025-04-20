import cv2
import mediapipe as mp
import threading
from flask import Flask
from flask_socketio import SocketIO
from flask_cors import CORS

# Flask and SocketIO setup
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# MediaPipe hand setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# === Calculate pinch distance and convert it to zoom ===
def get_zoom_from_pinch(hand_landmarks):
    thumb_tip = hand_landmarks[4]
    index_tip = hand_landmarks[8]

    # Euclidean distance
    distance = ((thumb_tip[0] - index_tip[0]) ** 2 + (thumb_tip[1] - index_tip[1]) ** 2) ** 0.5

    # Convert distance to zoom level (smaller distance = zoom in)
    zoom_level = max(0.5, min(2.5, 1 / (distance * 10)))  # Clamp zoom between 0.5 to 2.5
    return zoom_level

# === Get hand landmarks ===
def process_hand(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    landmarks = []
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            lm = [(l.x, l.y) for l in hand.landmark]
            landmarks.append(lm)
            mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
    return frame, landmarks

# === Background loop to track hand and send zoom ===
def hand_tracking_loop():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame, hands_data = process_hand(frame)

        if hands_data:
            zoom = get_zoom_from_pinch(hands_data[0])
            socketio.emit('zoom', {'level': zoom})
            print(f"Zoom: {zoom:.2f}")

        socketio.emit('hand_data', {'positions': hands_data})
        cv2.imshow("Hand Gesture - Pinch to Zoom", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# === Main ===
if __name__ == "__main__":
    tracking_thread = threading.Thread(target=hand_tracking_loop)
    tracking_thread.start()
    socketio.run(app, port=5000)
