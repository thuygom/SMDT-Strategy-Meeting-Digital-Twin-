import json

topic_affinity_labels = {
    "사건 / 논란": 0.75,
    "콘텐츠 평가": 0.88,
    "유튜버 개인": 0.98,
    "제품 / 아이템 리뷰": 0.90,
    "사회 / 시사 이슈": 0.78,
    "공감 / 감정 공유": 0.85,
    "정보 / 꿀팁": 0.92,
    "유머 / 드립": 0.85,
    "질문 / 피드백": 0.90,
    "기타 / 미분류": 0.60
}

with open("topic_affinity_labels.json", "w", encoding="utf-8") as f:
    json.dump(topic_affinity_labels, f, indent=2, ensure_ascii=False)

print("✅ 저장 완료: topic_affinity_labels.json")
