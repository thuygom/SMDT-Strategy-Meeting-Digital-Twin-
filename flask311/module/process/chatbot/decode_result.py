# decode_result.py
import sys
import json

data = sys.stdin.read().strip()

if not data:
    print("⚠️ 서버 응답 없음")
    sys.exit(1)

try:
    parsed = json.loads(data)
    print("📌 function_call:", parsed.get("function_call", "-"))
    print("📊 result:", parsed.get("result", "-"))
    print("🧠 gpt_summary:", parsed.get("gpt_summary", "(없음)"))
except json.JSONDecodeError:
    print("❌ JSON 파싱 실패. 응답 원문:")
    print(data)
except Exception as e:
    print("❌ 처리 중 오류 발생:", str(e))
