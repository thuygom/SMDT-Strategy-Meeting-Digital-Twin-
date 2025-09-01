import pandas as pd
import os

# 설정
dir_path = 'C:/Users/bandl/OneDrive/바탕 화면/youtube_data/youtube_data'
out_path = 'C:/Users/bandl/OneDrive/바탕 화면/youtube_data'

# ✅ 사용자에게 Video ID 열 인덱스 입력 받기
while True:
    try:
        URL_COL_INDEX = int(input("🔢 Video ID가 포함된 열의 인덱스를 입력하세요 (예: 첫 번째 열이면 0): "))
        break
    except ValueError:
        print("❌ 숫자를 입력해주세요.")

# 모든 파일의 URL을 담을 리스트
all_urls = []

for filename in os.listdir(dir_path):
    if filename.endswith('.xlsx'):
        file_path = os.path.join(dir_path, filename)
        print(f"\n📂 처리 중: {filename}")

        try:
            df = pd.read_excel(file_path).reset_index(drop=True)

            # 유효성 확인
            if URL_COL_INDEX >= df.shape[1]:
                print(f"⚠️ '{filename}'에 열 인덱스 {URL_COL_INDEX}가 없습니다. (총 열 수: {df.shape[1]})")
                continue

            # Video ID → YouTube URL 변환
            video_ids = df.iloc[:, URL_COL_INDEX].dropna().astype(str)
            full_urls = video_ids.apply(lambda vid: f"https://www.youtube.com/watch?v={vid}").tolist()
            all_urls.extend(full_urls)

        except Exception as e:
            print(f"❌ 오류 발생 ({filename}): {e}")

# 중복 제거
unique_urls = list(set(all_urls))

# DataFrame 생성
url_df = pd.DataFrame(unique_urls, columns=["YouTube URL"])

# 저장
output_path = os.path.join(out_path, "youtube_urls_from_ids.xlsx")
url_df.to_excel(output_path, index=False)
print(f"\n✅ 총 {len(unique_urls)}개의 URL을 저장했습니다: {output_path}")
