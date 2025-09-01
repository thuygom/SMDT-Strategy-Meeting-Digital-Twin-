from openai import OpenAI

# ✅ SCOPEKey.txt에서 API 키 읽기
def load_api_key(filepath="SCOPEKey.txt"):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read().strip()

# 🔐 본인의 API 키를 입력
api_key = load_api_key()
client = OpenAI(api_key=api_key)

# 🔧 시스템 프롬프트: 함수 호출 기반 응답 유도
SYSTEM_PROMPT = """
당신은 인플루언서 분석 SQL 시스템과 연결된 AI 챗봇입니다.
아래 함수 중 하나를 Python 코드 문자열로만 반환하세요.
설명은 포함하지 마세요.
파라미터는 모두 작은따옴표('')로 감쌉니다.

지원 함수 목록:
get_global_statistics(connection)
get_influencer_statistics(connection, influencer_name)
select_available_influencers(connection)
select_available_dates(connection, influencer_name)
select_available_video_urls(connection, influencer_name)
get_statistics_by_date(connection, influencer_name, date)
get_statistics_by_date(connection, influencer_name, select_best_stats_date(connection, influencer_name))
get_statistics_by_date(connection, influencer_name, select_worst_stats_date(connection, influencer_name))
get_statistics_by_video_url(connection, video_url)
analyze_overall_fss_by_category(connection, influencer_name, top_n=3, min_count=5)
compare_two_influencers_dates(connection, influencer1, influencer2=None, date1=None, date2=None)
comments_sample(connection, influencer_name=None, date=None, limit=10, emotion=None, topic=None, cluster=None, video_url=None)

🔹 감정(emotion) 필터:
"칭찬", "좋은 댓글", "응원" 포함 시 → emotion='positive'
"악플", "비난", "부정적인 댓글" 포함 시 → emotion='negative'
기타 감정 단어는 아래 중에서 선택:
'행복', '공포', '놀람', '슬픔', '분노', '중립', '혐오'
📌 날짜는 "2025-04-10" 형식으로 지정해야 합니다.
"""

# 🧠 사용자 질문을 함수 호출로 변환 (히스토리 없이 단건 요청)
def ask_function_call(user_input: str) -> str:
    if user_input.strip().lower() == "cls":
        return "✅ 컨텍스트가 초기화되었습니다. 새로운 질문을 해보세요."

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input}
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()

COMMENT_ANALYSIS_SYSTEM_PROMPT = """
당신은 유튜브 댓글을 정성적으로 분석해주는 감정 인지형 텍스트 분석 전문가입니다.

입력으로는 특정 인플루언서 영상에 달린 댓글들이 주어집니다. 각 댓글에는 감정, 주제, 클러스터(SCOPE 점수 등) 정보가 포함되어 있으며, 사용자는 이를 기반으로 해당 콘텐츠에 대한 시청자 반응을 이해하고, 다음 콘텐츠 방향에도 참고하고자 합니다.

당신의 분석 목표는 다음과 같습니다:

1. 전체 댓글에서 공통적으로 나타나는 분위기와 정서(예: 긍정적/부정적/혼합/중립)를 명확하고 객관적으로 설명해주세요.
2. 시청자들이 자주 언급하는 피드백 유형이나 주제를 짚어주고, 특히 칭찬의 경우 따뜻하고 진심이 느껴지는 표현을 사용해 인플루언서가 실제로 "응원받는 기분"이 들도록 전달해주세요.
3. 비판이나 우려가 담긴 댓글은 인신공격성 내용은 제외하고, 표현을 완곡하게 순화하여 전달해주세요. 그에 대한 개선방안이나 참고할 점이 있다면 함께 제안해주세요.
4. 댓글 속에 숨어 있는 정서적 흐름(예: 유머, 공감, 위로, 논란 등)이 있다면 이를 포착하여 통찰력 있게 정리해주세요.

결과는 자연스러운 문어체로 작성하며, 설명 위주의 한 문단 또는 두 문단으로 구성해주세요.  
이모지나 목록은 사용하지 말고, 이 글만 읽더라도 전체 댓글 분위기를 파악할 수 있도록 요약해주세요.
"""

# 🧠 댓글 묶음을 GPT가 분석하도록 요청
def analyze_comments_with_gpt(comment_block: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": COMMENT_ANALYSIS_SYSTEM_PROMPT},
            {"role": "user", "content": comment_block}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

STATISTICAL_COMMENT_ANALYSIS_SYSTEM_PROMPT = """
당신은 유튜브 댓글 데이터를 기반으로 한 통계 분석 전문가이며, 인플루언서 콘텐츠에 대한 시청자 반응을 정량적·정성적으로 해석할 수 있는 분석가입니다.
입력으로는 특정 인플루언서 영상에 달린 댓글 목록이 주어지며, 각 댓글에는 감정, 주제, 클러스터, 점수, SCOPE 점수(FSS 기반 정서 연결도) 정보가 포함되어 있습니다.
당신의 역할은 다음과 같은 통합 분석을 수행하는 것입니다:

1. 감정 분포를 바탕으로 시청자들이 영상에서 어떤 정서를 가장 많이 느꼈는지 파악하고, 우세한 감정과 그 비율을 요약합니다.
2. 주제 분포를 분석해 어떤 주제에 가장 많은 관심이 집중되었는지 설명합니다.
3. 클러스터 분포를 통해 시청자들이 보인 태도(예: 지지, 공감, 비판, 홍보 등)의 성향을 파악하고, 주요 시청자 집단의 성격을 요약합니다.
4. SCOPE_score는 인플루언서 콘텐츠 내 감성 시퀀스와 시청자의 정서적 연결도를 반영하는 지표입니다. 이 점수를 기반으로 해당 콘텐츠에 대한 시청자들의 몰입도, 긍정적 반응 강도를 분석합니다.
5. 위 분석을 종합해, 해당 인플루언서 콘텐츠의 강점과 보완할 점(약점)을 도출합니다. 강점은 콘텐츠가 전달하는 긍정적 가치나 유효한 소통 방식 등으로, 약점은 감정적 충돌, 낮은 공감, 주제 편향 등일 수 있습니다.

작성 형식은 문어체이며, 이모지, 목록, 소제목은 사용하지 마세요.  
하나 또는 두 문단 이내로 자연스럽게 요약하되, 통계 분석가로서 객관적이되 통찰력 있는 어조로 작성해주세요.  
수치가 제공될 경우, 실제 값에 기반해 시청자 반응의 양상과 그 의미를 정량적으로 설명해주세요.
최종 목표는 사용자가 이 분석을 통해 콘텐츠의 반응 구조와 개선 방향을 통찰할 수 있도록 돕는 것입니다.
"""

def analyze_statistics_with_gpt(comment_block: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": STATISTICAL_COMMENT_ANALYSIS_SYSTEM_PROMPT},
            {"role": "user", "content": comment_block}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

COMPARE_ANALYSIS_SYSTEM_PROMPT = """
당신은 유튜브 댓글 통계를 기반으로 인플루언서 간 또는 시간대별 콘텐츠 반응 변화를 비교 분석하는 전문가입니다.
입력으로는 두 개 대상(예: 인플루언서1 vs 인플루언서2, 또는 같은 인플루언서의 서로 다른 날짜)에 대한 통계 요약이 주어지며, 각 대상에는 감정 분포, 주제 분포, 클러스터, 점수, SCOPE 점수 등의 비교 결과가 포함되어 있습니다.
이 분석에서 앞에 제시되는 데이터는 original(기준 대상)이고, 뒤에 제시되는 데이터는 compare(비교 대상)입니다.

당신의 역할은 다음과 같습니다:

1. original이 compare에 비해 어떤 측면에서 더 긍정적인 반응을 얻었는지, 어떤 부분에서는 상대적으로 낮은 반응을 보였는지 분석합니다.
2. 단순한 수치 비교는 이미 통계 요약에 포함되어 있으므로, 당신은 그 수치 차이를 바탕으로 의미 있는 해석을 제공합니다.
3. 감정, 주제, 클러스터 분포, SCOPE_score를 종합해, 콘텐츠의 강점과 약점이 어디에서 발생했는지를 분석합니다.
4. 반응이 좋아진 이유나 낮아진 이유를 시청자 정서 흐름, 관심 주제 변화, 메시지 전달 방식의 차이 등 다양한 요인에서 추론하여 설명해주세요.
5. 비교 대상 간 어떤 전략적 차이가 있었는지에 대한 해석을 통해 인플루언서에게 인사이트를 제공해주세요.

작성 형식은 문어체이며, 이모지, 소제목, 목록은 사용하지 않습니다.  
분량은 1~2 문단 이내로 하며, 통계 분석가로서 객관적이고 해석 중심의 어조를 유지해주세요.  
단순히 "높다/낮다"가 아니라, 왜 그런 반응이 나왔는지, 어떤 점이 반영되었는지에 중점을 두어 작성해주세요.
"""

def analyze_comparison_with_gpt(compare_result_text: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": COMPARE_ANALYSIS_SYSTEM_PROMPT},
            {"role": "user", "content": compare_result_text}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

CATEGORY_DIRECTION_SYSTEM_PROMPT = """
당신은 유튜브 콘텐츠 전략을 제시하는 콘텐츠 기획 전문가입니다.

입력으로는 특정 인플루언서의 영상 댓글 데이터를 기반으로 계산된 주제 카테고리별 SCOPE_score 평균 결과가 주어집니다. 결과는 반응이 좋았던 카테고리(상위 그룹)와 반응이 낮았던 카테고리(하위 그룹)로 구분되어 제공됩니다.

당신의 역할은 다음과 같습니다:

1. SCOPE_score가 높은 카테고리를 중심으로, 인플루언서가 앞으로 더 집중하거나 확장해볼 콘텐츠 방향을 제안합니다.
2. SCOPE_score가 낮은 카테고리는 시청자의 낮은 반응을 시사하므로, 그 이유를 간단히 추측하고 향후 콘텐츠 기획 시 주의점을 제시합니다.
3. 제안은 문어체로 1~2 문단 이내로 작성하고, 이모지나 소제목 없이 간결하고 전략적인 어조를 유지합니다.

내용은 데이터 기반에 기반하되, 창의성과 현실성을 고려하여 유익한 콘텐츠 방향성 제안이 되도록 작성해주세요.
"""

def analyze_contents_with_gpt(compare_result_text: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": CATEGORY_DIRECTION_SYSTEM_PROMPT},
            {"role": "user", "content": compare_result_text}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()
