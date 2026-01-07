import cv2
from deepface import DeepFace
import webbrowser
import random
import time
from collections import deque, Counter

# ---------------- EMOTION STABILITY SETTINGS ----------------
emotion_buffer = deque(maxlen=20)
stable_emotion = None
last_stable_emotion = None
last_play_time = 0
COOLDOWN_SECONDS = 20  # change if needed

# ---------------- EMOTION → YOUTUBE SONGS ----------------
emotion_music = {
    "happy": [
        "https://youtu.be/riUBONxLABg?si=Q8O-GUp_pVNlQgfL",
        "https://youtu.be/O-yRLOWSgEM?si=gtPbSEQ6_qgE7ssS"
    ],
    "sad": [
        "https://youtu.be/3M377ycIwDA?si=WvIffn4r7rzoV_VC"
        "https://youtu.be/9m3mT4KV3es?si=0X1ObPC8PjOrQSGc"
    ],
    "angry": [
        "https://youtu.be/CKpbdCciELk?si=-BET7jmRUWArWDqM"
    ],
    "neutral": [
        "https://youtu.be/f-KyCvE8AS0?si=b0zTv_tifsGIxdvL"
    ],
    "surprise": [
        "https://youtu.be/Fa4COn3sPDY?si=rUJfHhR0ynMS27Yr'"
    ],
    "disgust": [
        "https://youtu.be/ltLTwjjTehg?si=jxzU4xHP6Q-2qTg2'"
    ],
    "fear": [
        "https://youtu.be/FbXOsVByKmk?si=-pvcF2PVqRw1ISAX"
    ]
}
    

# ---------------- WEBCAM ----------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        result = DeepFace.analyze(
            frame,
            actions=['emotion'],
            enforce_detection=False
        )

        if isinstance(result, list):
            result = result[0]

        dominant_emotion = result['dominant_emotion']
        emotions = result['emotion']

        # ----------- DISPLAY EMOTION PERCENTAGES -----------
        y = 30
        for emotion, score in emotions.items():
            cv2.putText(
                frame,
                f"{emotion}: {score:.2f}%",
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            y += 25

        cv2.putText(
            frame,
            f"Dominant: {dominant_emotion}",
            (10, y + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        # ----------- STABLE EMOTION LOGIC -----------
        emotion_buffer.append(dominant_emotion)

        if len(emotion_buffer) == emotion_buffer.maxlen:
            stable_emotion = Counter(emotion_buffer).most_common(1)[0][0]
            current_time = time.time()

            if (
                stable_emotion in emotion_music
                and stable_emotion != last_stable_emotion
                and (current_time - last_play_time) > COOLDOWN_SECONDS
            ):
                webbrowser.open(
                    random.choice(emotion_music[stable_emotion]),
                    new=0
                )
                last_stable_emotion = stable_emotion
                last_play_time = current_time

    except Exception as e:
        cv2.putText(
            frame,
            "Detecting...",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    cv2.imshow("Emotion Based Music Recommendation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()