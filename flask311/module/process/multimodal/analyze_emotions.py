from deepface import DeepFace
from collections import Counter

def analyze_emotions(face_images):
    emotion_counts = Counter()
    total = 0

    for face in face_images:
        try:
            result = DeepFace.analyze(face, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
            emotion_counts[emotion] += 1
            total += 1
        except Exception as e:
            print(f"분석 오류: {e}")
            continue

    if total == 0:
        return {}

    # 감정 비율 계산
    emotion_percentages = {emotion: (count / total) * 100 for emotion, count in emotion_counts.items()}
    return emotion_percentages
