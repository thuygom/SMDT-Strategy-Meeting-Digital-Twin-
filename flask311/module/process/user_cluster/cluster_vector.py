import json

cluster_reliability = {
    "Aggressive": 0.62,
    "Supportive": 0.88,
    "Neutral Informative": 0.81,
    "Sarcastic/Playful": 0.67,
    "Analytical": 0.90,
    "Spam/Promotional": 0.48,
    "Empathetic": 0.85
}

with open("cluster_reliability.json", "w", encoding="utf-8") as f:
    json.dump(cluster_reliability, f, indent=2, ensure_ascii=False)

print("✅ 저장 완료: cluster_reliability.json")
