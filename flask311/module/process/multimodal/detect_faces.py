import cv2
import os
import math
import argparse
import numpy as np
import mediapipe as mp
from deepface import DeepFace
from scipy.spatial.distance import cosine

def cosine_distance(v1, v2):
    return cosine(v1, v2)

def track_faces_mediapipe(video_path, output_dir="output_faces", max_frames=300, threshold=0.5):
    print(f"[🚀 시작] '{video_path}' 에서 MediaPipe + ArcFace 얼굴 추적")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[❌ 오류] 비디오 열기 실패")
        return

    mp_face = mp.solutions.face_detection
    face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

    os.makedirs(output_dir, exist_ok=True)

    person_db = {}
    person_counter = 0
    frame_count = 0

    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        print(f"[📷 프레임 {frame_count}] 처리 중...")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb)

        if not results.detections:
            print("  └─ 얼굴 없음")
            continue

        for i, detection in enumerate(results.detections):
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            w_box = int(bbox.width * w)
            h_box = int(bbox.height * h)

            x, y = max(x, 0), max(y, 0)
            face_img = frame[y:y+h_box, x:x+w_box]

            if w_box < 60 or h_box < 60:
                print("    └─ 너무 작은 얼굴, 스킵")
                continue

            try:
                embedding = DeepFace.represent(face_img, model_name="ArcFace", enforce_detection=False)[0]['embedding']
            except Exception as e:
                print(f"    [!] 얼굴 아님 or 인식 실패: {e}")
                continue

            matched_id = None
            min_dist = 1.0

            for pid, data in person_db.items():
                avg_emb = np.mean(data['embeddings'], axis=0)
                dist = cosine_distance(embedding, avg_emb)
                if dist < threshold and dist < min_dist:
                    matched_id = pid
                    min_dist = dist

            if matched_id:
                person_id = matched_id
                print(f"    ✅ 기존 인물 인식됨: {person_id} (거리={min_dist:.4f})")
                person_db[person_id]['embeddings'].append(embedding)
                person_db[person_id]['last_seen'] = frame_count
            else:
                person_counter += 1
                person_id = f"P{person_counter}"
                print(f"    ➕ 새로운 인물 등록: {person_id}")
                person_db[person_id] = {
                    'embeddings': [embedding],
                    'last_seen': frame_count
                }

            # 저장
            person_folder = os.path.join(output_dir, person_id)
            os.makedirs(person_folder, exist_ok=True)
            save_path = os.path.join(person_folder, f"frame{frame_count}.jpg")
            cv2.imwrite(save_path, face_img)

            # 시각화
            cv2.rectangle(frame, (x, y), (x+w_box, y+h_box), (0,255,0), 2)
            cv2.putText(frame, person_id, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[🛑 사용자 종료]")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[✅ 완료] 얼굴 이미지 저장 완료")

# ✅ main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MediaPipe + ArcFace 얼굴 트래킹")
    parser.add_argument("video_path", type=str, help="비디오 경로")
    parser.add_argument("--output_dir", type=str, default="output_faces")
    parser.add_argument("--max_frames", type=int, default=300)
    parser.add_argument("--threshold", type=float, default=0.55)

    args = parser.parse_args()
    track_faces_mediapipe(
        args.video_path,
        args.output_dir,
        args.max_frames,
        args.threshold
    )
