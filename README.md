
## 📂 Folder Structure

```
project/
├── app.py                    # Main Flask backend
├── requirements.txt          # Python dependencies
├── model/
│   ├── asl_model.joblib      # ML model (auto-downloaded)
│   └── label_encoder.joblib  # Label encoder (auto-downloaded)
├── static/
│   └── audio/                # Contains generated .mp3 files
├── templates/
│   └── index.html            # Frontend web page
```

---

## 🧠 How It Works

1. OpenCV captures video from the webcam.
2. MediaPipe extracts 21 hand landmarks (x, y, z = 63 features).
3. The trained RandomForestClassifier model predicts the ASL letter.
4. A letter is only added to the sentence if:
   - It's stable across multiple frames, and
   - The model's confidence is **85% or more**.
5. User can press:
   - ✅ **Speak** → converts the sentence to an audio file
   - ❌ **Clear** → resets the current sentence
6. Audio is saved and streamed via browser.

---


