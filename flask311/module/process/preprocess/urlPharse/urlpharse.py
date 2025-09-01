import pandas as pd
import os

# 설정
dir_path = 'C:/Users/bandl/OneDrive/바탕 화면/youtube_data/youtube_data'
out_path = 'C:/Users/bandl/OneDrive/바탕 화면/youtube_data'

# ✅ 사용자에게 URL 컬럼 인덱스 입력 받기
while True:
    try:
        URL_COL_INDEX = int(input("🔢 URL이 포함된 열의 인덱스를 입력하세요 (예: 첫 번째 열이면 0): "))
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

            # ✅ 입력한 열 인덱스가 유효한지 확인
            if URL_COL_INDEX >= df.shape[1]:
                print(f"⚠️ '{filename}' 파일에 열 인덱스 {URL_COL_INDEX}가 존재하지 않습니다. (총 열 수: {df.shape[1]})")
                continue

            # 지정한 컬럼에서 URL 추출
            urls = df.iloc[:, URL_COL_INDEX].dropna().astype(str).tolist()
            all_urls.extend(urls)

        except Exception as e:
            print(f"❌ 오류 발생 ({filename}): {e}")

# URL 중복 제거
unique_urls = list(set(all_urls))

# DataFrame으로 저장
url_df = pd.DataFrame(unique_urls, columns=["URL"])

# 저장 경로
output_path = os.path.join(out_path, "unique_urls.xlsx")
url_df.to_excel(output_path, index=False)
print(f"\n✅ 중복 제거된 URL {len(unique_urls)}개를 저장했습니다: {output_path}")
