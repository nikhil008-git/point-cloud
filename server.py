import cv2
import mediapipe as mp
import threading
from flask import Flask
from flask_socketio import SocketIO
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

# Gesture tracking helpers
prev_hand_center = None

def get_zoom_from_pinch(hand_landmarks):
    thumb_tip = hand_landmarks[4]
    index_tip = hand_landmarks[8]
    distance = ((thumb_tip[0] - index_tip[0]) ** 2 + (thumb_tip[1] - index_tip[1]) ** 2) ** 0.5
    zoom_level = max(0.5, min(2.5, 1 / (distance * 10)))  # Adjust zoom level
    return zoom_level

def get_rotation_from_movement(current_center):
    global prev_hand_center
    if prev_hand_center is None:
        prev_hand_center = current_center
        return (0, 0)

    dx = current_center[0] - prev_hand_center[0]
    dy = current_center[1] - prev_hand_center[1]
    prev_hand_center = current_center

    # Scale movement to reasonable rotation values
    rot_x = dy * 5  # vertical movement → rotate X
    rot_y = dx * 5  # horizontal movement → rotate Y
    return rot_x, rot_y

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

def hand_tracking_loop():
    global prev_hand_center
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame, hands_data = process_hand(frame)

        if hands_data:
            hand = hands_data[0]
            zoom = get_zoom_from_pinch(hand)

            # Center point of palm for rotation
            palm_center = hand[9]  # approx palm center landmark
            rot_x, rot_y = get_rotation_from_movement(palm_center)

            # Emit gesture data via SocketIO
            socketio.emit('gesture_data', {
                'zoom': zoom,
                'rotateX': rot_x,
                'rotateY': rot_y
            })

            print(f"Zoom: {zoom:.2f}, RotateX: {rot_x:.2f}, RotateY: {rot_y:.2f}")

        cv2.imshow("Gesture Control (Pinch + Movement)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return "Gesture Tracking Server is Running"

if __name__ == "__main__":
    t = threading.Thread(target=hand_tracking_loop)
    t.start()
    socketio.run(app, port=5000)
