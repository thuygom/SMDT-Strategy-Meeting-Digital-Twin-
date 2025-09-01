import argparse
from detect_faces import detect_faces_from_video
from analyze_emotions import analyze_emotions

def main():
    parser = argparse.ArgumentParser(description="비디오에서 감정 분석")
    parser.add_argument("video_file", type=str, help="비디오 파일 경로")
    args = parser.parse_args()

    print("얼굴 인식 중...")
    faces = detect_faces_from_video(args.video_file)
    print(f"{len(faces)}개의 얼굴이 감지되었습니다.")

    print("감정 분석 중...")
    results = analyze_emotions(faces)

    print("감정 분석 결과:")
    for emotion, percentage in results.items():
        print(f"{emotion}: {percentage:.2f}%")

if __name__ == "__main__":
    main()
