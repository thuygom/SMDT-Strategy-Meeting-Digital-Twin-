import os
import pandas as pd

# ▶ 남길 열 목록 (원본 이름 기준)
COLUMNS_TO_KEEP = ["게시물 URL", "댓글", "댓글 작성일", "감정", "주제", "군집"]

# ▶ 컬럼명 변경 매핑
RENAME_MAP = {
    "게시물 URL": "video_url",
    "댓글": "comment",
    "댓글 작성일": "date",
    "감정": "감정",
    "주제": "주제",
    "군집": "군집"
}

# ▶ 디렉토리 기반 일괄 처리 함수
def filter_and_rename_excel_columns(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".xlsx"):
            file_path = os.path.join(directory, filename)
            print(f"🔍 처리 중: {filename}")
            try:
                df = pd.read_excel(file_path)

                # 필요한 열만 추출
                filtered_cols = [col for col in COLUMNS_TO_KEEP if col in df.columns]
                filtered_df = df[filtered_cols]

                # ✅ 게시물 URL 비어있는 경우 위 값으로 채우기
                if "게시물 URL" in filtered_df.columns:
                    filtered_df["게시물 URL"] = filtered_df["게시물 URL"].fillna(method="ffill")

                # 컬럼명 영어로 변경
                rename_map_local = {col: RENAME_MAP[col] for col in filtered_cols if col in RENAME_MAP}
                filtered_df.rename(columns=rename_map_local, inplace=True)

                # ✅ 원본 파일에 덮어쓰기
                filtered_df.to_excel(file_path, index=False)
                print(f"✅ 덮어쓰기 완료: {file_path}")
            except Exception as e:
                print(f"❌ 오류 발생 ({filename}): {e}")

# ▶ 실행 예시
if __name__ == "__main__":
    target_dir = "C:/Users/bandl/OneDrive/바탕 화면/youtube_data/instagra"
    filter_and_rename_excel_columns(target_dir)
