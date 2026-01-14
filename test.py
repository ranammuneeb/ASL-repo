import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
import gdown
from gtts import gTTS
import uuid
from datetime import datetime, timedelta
import glob

# -------------------------------------------------
# Global State
# -------------------------------------------------
predicted_sentence = ""
current_sign = ""
prediction_count = 0
threshold_frames = 15
last_prediction = ""

# -------------------------------------------------
# Audio Directory
# -------------------------------------------------
AUDIO_DIR = "audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

# -------------------------------------------------
# Model Paths
# -------------------------------------------------
MODEL_PATH = "model/asl_model.joblib"
ENCODER_PATH = "model/label_encoder.joblib"

os.makedirs("model", exist_ok=True)

MODEL_FILE_ID = "1oZeBgnRUqLYe5IaYG6NIokCEuqz07Ru2"
ENCODER_FILE_ID = "13oBSsI927KltAI7z0bpz3hgCTrUAQap-"

# -------------------------------------------------
# Download Model if Missing
# -------------------------------------------------
if not os.path.exists(MODEL_PATH):
    print("⬇ Downloading ASL model...")
    gdown.download(f"https://drive.google.com/uc?id={MODEL_FILE_ID}", MODEL_PATH, quiet=False)

if not os.path.exists(ENCODER_PATH):
    print("⬇ Downloading LabelEncoder...")
    gdown.download(f"https://drive.google.com/uc?id={ENCODER_FILE_ID}", ENCODER_PATH, quiet=False)

# -------------------------------------------------
# Load Model
# -------------------------------------------------
model = joblib.load(MODEL_PATH)
le = joblib.load(ENCODER_PATH)

print("✅ Model and Encoder loaded")

# -------------------------------------------------
# MediaPipe Hands
# -------------------------------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.9,
    min_tracking_confidence=0.9
)

# -------------------------------------------------
# Audio Helpers
# -------------------------------------------------
def cleanup_old_audio_files():
    cutoff = datetime.now() - timedelta(hours=1)
    for f in glob.glob(os.path.join(AUDIO_DIR, "*.mp3")):
        if datetime.fromtimestamp(os.path.getctime(f)) < cutoff:
            os.remove(f)

def generate_audio_file(text):
    if not text.strip():
        return None

    cleanup_old_audio_files()

    filename = f"speech_{uuid.uuid4().hex[:8]}.mp3"
    filepath = os.path.join(AUDIO_DIR, filename)

    tts = gTTS(text=text, lang="en")
    tts.save(filepath)

    return filepath

def play_audio(filepath):
    if os.name == 'nt':  # Windows
        os.system(f'start {filepath}')
    elif os.name == 'posix':  # Linux/Mac
        os.system(f'afplay {filepath}' if os.uname().sysname == 'Darwin' else f'mpg123 {filepath}')

# -------------------------------------------------
# Main Loop
# -------------------------------------------------
def main():
    global predicted_sentence, current_sign
    global prediction_count, last_prediction

    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n🚀 ASL Recognition Started")
    print("\nControls:")
    print("  'c' - Clear sentence")
    print("  's' - Speak sentence")
    print("  'q' - Quit")
    print("\n" + "="*50 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                # Extract features
                features = []
                for lm in hand_landmarks.landmark:
                    features.extend([lm.x, lm.y, lm.z])

                # Predict
                x_input = np.array(features).reshape(1, -1)
                y_pred = model.predict(x_input)
                label = le.inverse_transform(y_pred)[0]

                current_sign = label

                # Count consecutive predictions
                if label == last_prediction:
                    prediction_count += 1
                else:
                    prediction_count = 0
                    last_prediction = label

                # Add to sentence after threshold
                if prediction_count == threshold_frames:
                    if label == "space":
                        predicted_sentence += " "
                    elif label == "del":
                        predicted_sentence = predicted_sentence[:-1]
                    elif label != "nothing":
                        predicted_sentence += label
                    prediction_count = 0
        else:
            current_sign = "nothing"

        # Create overlay background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (630, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Display current sign
        cv2.putText(frame, f"Current Sign: {current_sign}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Display sentence
        cv2.putText(frame, f"Sentence: {predicted_sentence}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Display instructions
        cv2.putText(frame, "Press: 'c' Clear | 's' Speak | 'q' Quit", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Show frame
        cv2.imshow("ASL Recognition", frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\n👋 Exiting...")
            break
        elif key == ord('c'):
            predicted_sentence = ""
            print("🗑️  Sentence cleared")
        elif key == ord('s'):
            if predicted_sentence.strip():
                print(f"🔊 Speaking: {predicted_sentence}")
                audio_file = generate_audio_file(predicted_sentence)
                if audio_file:
                    play_audio(audio_file)
            else:
                print("⚠️  No text to speak")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

# -------------------------------------------------
# Run
# -------------------------------------------------
if __name__ == "__main__":
    main()