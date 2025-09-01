import pymysql
from collections import Counter
from chat_execute import ask_function_call
from chat_execute import analyze_comments_with_gpt
from chat_execute import analyze_statistics_with_gpt
from chat_execute import analyze_comparison_with_gpt
from chat_execute import analyze_contents_with_gpt
from rich import print as rprint
from collections import defaultdict
import random

# ✅ DB 연결
def connect_to_db():
    return pymysql.connect(
        host='localhost',
        user='root',
        password='Bandalgom!0927',
        database='my_new_db',
        charset='utf8mb4',
        cursorclass=pymysql.cursors.DictCursor
    )

# ✅ 유튜브 댓글+메타 데이터 가져오기
def get_youtube_comment_data(connection, influencer_name=None, date=None, video_url=None):
    """
    유튜브 댓글과 메타데이터를 조건에 따라 조회 (topic_categories 포함)
    :param connection: DB 연결 객체
    :param influencer_name: 인플루언서 이름
    :param date: 영상 날짜
    :param video_url: 영상 URL
    :return: 조회된 댓글 및 메타데이터 리스트
    """
    query = """
        SELECT 
            yc.video_url,
            y.title,
            yc.comment,
            yc.emotion,
            yc.topic,
            y.topic_categories,  -- ✅ 추가된 컬럼
            yc.cluster,
            yc.score,
            yc.fss as SCOPE_score,
            y.view_count,
            y.like_count,
            y.comment_count,
            y.subscriber_count,
            y.date
        FROM youtube_comment yc
        JOIN youtube y ON yc.video_url = y.video_url
    """
    conditions = []
    params = []

    if video_url:
        conditions.append("y.video_url = %s")
        params.append(video_url)
    else:
        if influencer_name:
            conditions.append("y.influencer_name = %s")
            params.append(influencer_name)
        if date:
            conditions.append("DATE(y.date) = %s")
            params.append(date)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    with connection.cursor() as cursor:
        cursor.execute(query, tuple(params))
        return cursor.fetchall()

def comments_sample(connection, influencer_name=None, date=None, limit=10,
                    emotion=None, topic=None, cluster=None, video_url=None):
    """
    댓글 샘플 데이터를 무작위로 문자열로 반환 (기본 10개)
    """
    data = get_youtube_comment_data(connection, influencer_name, date)
    if not data:
        return "⚠️ 데이터가 없습니다."

    # ✅ 감정 매핑
    emotion_map = {
        "positive": ["행복"],
        "negative": ["분노", "혐오", "슬픔"]
    }

    # ✅ 필터링
    filtered = []
    for row in data:
        if video_url and row["video_url"] != video_url:
            continue
        if emotion:
            if emotion in emotion_map:
                if row["emotion"] not in emotion_map[emotion]:
                    continue
            elif row["emotion"] != emotion:
                continue
        if topic and row["topic"] != topic:
            continue
        if cluster and row["cluster"] != cluster:
            continue
        filtered.append(row)

    if not filtered:
        return "⚠️ 조건에 맞는 댓글이 없습니다."

    # ✅ 무작위 샘플링
    sampled = random.sample(filtered, min(limit, len(filtered)))

    # ✅ 결과 포맷
    target_label = f"{influencer_name or '전체'} - {date or '전체'}"
    if video_url:
        target_label += f" - 영상: {video_url}"
    if emotion in ["positive", "negative"]:
        target_label += f" - 감정: {emotion}"

    lines = [f"\n📋 [{target_label}] 필터링된 댓글 샘플 {len(sampled)}개:"]
    for i, row in enumerate(sampled, 1):
        lines.append(
            f"{i}. ({row['date']}) [{row['video_url']}] {row['comment']}\n"
            f"   ▶ 감정: {row['emotion']} / 주제: {row['topic']} / 클러스터: {row['cluster']} / "
            f"점수: {row['score']} / SCOPE_score: {row['SCOPE_score']}"
        )
    return "\n".join(lines)

# ✅ 평균 계산
def calculate_average_stats(data):
    keys = ['SCOPE_score', 'view_count', 'like_count', 'comment_count', 'subscriber_count']
    avg_stats = {}
    for key in keys:
        values = [row[key] for row in data if row.get(key) is not None]
        avg_stats[f"avg_{key}"] = sum(values) / len(values) if values else 0
    return avg_stats

# ✅ 분포 계산
def calculate_distribution(data, key):
    counter = Counter([row[key] for row in data if row.get(key)])
    total = sum(counter.values())
    return {k: round((v / total) * 100, 2) for k, v in counter.items()} if total > 0 else {}

def format_stat_dict(title, stat_dict):
    """
    통계 딕셔너리를 문자열로 포맷팅
    :param title: 출력 제목
    :param stat_dict: {항목: 값} 형태의 평균 통계 딕셔너리
    :return: 포맷된 문자열
    """
    lines = [f"\n[{title}]"]
    for key, value in stat_dict.items():
        lines.append(f"{key}: {value:.2f}")
    return "\n".join(lines)

def format_distribution(title, dist_dict):
    """
    분포 딕셔너리를 문자열로 포맷팅
    :param title: 출력 제목
    :param dist_dict: {카테고리: 퍼센트} 형태의 분포 딕셔너리
    :return: 포맷된 문자열
    """
    lines = [f"\n[{title}]"]
    for key, pct in dist_dict.items():
        lines.append(f"- {key}: {pct}%")
    return "\n".join(lines)

# ✅ 날짜 선택 유도
def select_available_dates(connection, influencer_name):
    """
    특정 인플루언서의 유효한 날짜 목록을 문자열로 반환 (최고/최저 SCOPE_score 날짜 강조 포함)
    :param connection: DB 연결 객체
    :param influencer_name: 인플루언서 이름
    :return: 날짜 목록 문자열
    """
    # 유효 날짜 조회
    query = """
        SELECT DISTINCT y.date 
        FROM youtube y 
        WHERE y.influencer_name = %s 
        ORDER BY y.date
    """
    with connection.cursor() as cursor:
        cursor.execute(query, (influencer_name,))
        rows = cursor.fetchall()
        dates = [row['date'] for row in rows]

    if not dates:
        return f"⚠️ [{influencer_name}]의 유효한 날짜가 없습니다."

    # 최고 및 최저 날짜 가져오기
    best_date = select_best_stats_date(connection, influencer_name)
    worst_date = select_worst_stats_date(connection, influencer_name)

    result = [f"📅 [{influencer_name}]의 유효한 날짜 목록:"]
    for i, date in enumerate(dates):
        label = ""
        if date == best_date:
            label += "🌟 최고 성과일"
        if date == worst_date:
            if label: label += " / "
            label += "📉 최저 성과일"
        result.append(f"{i + 1}. {date}" + (f"  ← {label}" if label else ""))
    
    return "\n".join(result)

def select_available_influencers(connection):
    """
    유효한 인플루언서 이름 목록을 문자열로 반환
    :param connection: DB 연결 객체
    :return: 인플루언서 목록 문자열
    """
    query = """
        SELECT DISTINCT influencer_name 
        FROM youtube 
        ORDER BY influencer_name
    """
    with connection.cursor() as cursor:
        cursor.execute(query)
        rows = cursor.fetchall()
        names = [row['influencer_name'] for row in rows]

    if not names:
        return "⚠️ 등록된 인플루언서가 없습니다."

    result = ["🧑‍💻 유효한 인플루언서 목록:"]
    for i, name in enumerate(names):
        result.append(f"{i + 1}. {name}")

    return "\n".join(result)

def select_available_video_urls(connection, influencer_name):
    """
    특정 인플루언서의 유효한 영상 제목 및 YouTube ID 목록을 날짜별로 정리해 문자열로 반환
    :param connection: DB 연결 객체
    :param influencer_name: 인플루언서 이름
    :return: 영상 제목 목록 문자열
    """
    query = """
        SELECT date, title, video_url
        FROM youtube
        WHERE influencer_name = %s
        ORDER BY date
    """
    with connection.cursor() as cursor:
        cursor.execute(query, (influencer_name,))
        rows = cursor.fetchall()

    if not rows:
        return f"⚠️ [{influencer_name}]의 영상이 없습니다."

    result = [f"🎬 [{influencer_name}]의 영상 목록 (제목 + YouTube ID):"]
    for i, row in enumerate(rows):
        date = row['date']
        title = row['title']
        url = row['video_url']
        video_id = url.split("v=")[-1].split("&")[0] if "v=" in url else "❓ID 없음"
        result.append(f"{i + 1}. ({date}) 『{title}』 (🎞️ ID: {video_id})")

    return "\n".join(result)

# ✅ 최고 SCOPE_score 날짜 구하기
def select_best_stats_date(connection, influencer_name):
    query = """
        SELECT y.date
        FROM youtube_comment yc
        JOIN youtube y ON yc.video_url = y.video_url
        WHERE y.influencer_name = %s
        GROUP BY y.date
        ORDER BY AVG(fss) DESC
        LIMIT 1
    """
    with connection.cursor() as cursor:
        cursor.execute(query, (influencer_name,))
        row = cursor.fetchone()
        return row['date'] if row else None

# ✅ 최저 SCOPE_score 날짜 구하기
def select_worst_stats_date(connection, influencer_name):
    query = """
        SELECT y.date
        FROM youtube_comment yc
        JOIN youtube y ON yc.video_url = y.video_url
        WHERE y.influencer_name = %s
        GROUP BY y.date
        ORDER BY AVG(fss) ASC
        LIMIT 1
    """
    with connection.cursor() as cursor:
        cursor.execute(query, (influencer_name,))
        row = cursor.fetchone()
        return row['date'] if row else None

def compare_two_stats(original_stats, compare_stats):
    """
    두 개의 통계 dict를 비교하여 항목별 차이를 문자열로 반환 (절댓값 + 퍼센트 차이)
    """
    result = [f"\n📊 항목별 차이 (기준 - 비교대상)"]
    for key in original_stats:
        if key.startswith("avg_") and key in compare_stats:
            orig = original_stats[key]
            comp = compare_stats[key]
            diff = orig - comp

            # 퍼센트 차이 계산 (comp가 0이면 'N/A')
            if comp == 0:
                pct = "N/A"
            else:
                pct = f"{(diff / comp) * 100:+.2f}%"

            result.append(f"{key}: {orig:.2f} - {comp:.2f} = {diff:+.2f} ({pct})")
    return "\n".join(result)

def compare_distributions(dist1, dist2):
    """
    두 분포 딕셔너리를 비교하여 공통된 라벨에 대해 퍼센트 차이를 문자열로 반환
    """
    lines = [f"\n📊 공통 카테고리별 퍼센트 차이 (기준 - 비교대상)"]

    common_keys = set(dist1.keys()).intersection(dist2.keys())
    if not common_keys:
        lines.append("⚠️ 공통된 라벨이 없어 비교할 수 없습니다.")
        return "\n".join(lines)

    for key in sorted(common_keys):
        try:
            val1 = dist1[key]
            val2 = dist2[key]
            diff = val1 - val2
            lines.append(f"{key}: {val1:.2f}% - {val2:.2f}% = {diff:+.2f}pp")
        except Exception as e:
            lines.append(f"{key}: 비교 실패 ({e})")

    return "\n".join(lines)

#전체 인플루언서 평균 통계 보여줘
def get_global_statistics(connection):
    """
    전체 인플루언서의 평균 통계 및 감정/주제/클러스터 분포를 문자열로 반환
    """
    all_data = get_youtube_comment_data(connection)
    lines = [
        format_stat_dict("전체 인플루언서 평균 통계", calculate_average_stats(all_data)),
        format_distribution("전체 감정 분포", calculate_distribution(all_data, "emotion")),
        format_distribution("전체 주제 분포", calculate_distribution(all_data, "topic")),
        format_distribution("전체 클러스터 분포", calculate_distribution(all_data, "cluster")),
    ]
    return "\n".join(lines)

#특정 인플루언서 평균 통계 보여줘
def get_influencer_statistics(connection, influencer_name):
    """
    특정 인플루언서의 평균 통계 및 감정/주제/클러스터 분포를 문자열로 반환
    """
    inf_data = get_youtube_comment_data(connection, influencer_name)
    lines = [
        format_stat_dict(f"{influencer_name} 평균 통계", calculate_average_stats(inf_data)),
        format_distribution(f"{influencer_name} 감정 분포", calculate_distribution(inf_data, "emotion")),
        format_distribution(f"{influencer_name} 주제 분포", calculate_distribution(inf_data, "topic")),
        format_distribution(f"{influencer_name} 클러스터 분포", calculate_distribution(inf_data, "cluster")),
    ]
    return "\n".join(lines)

#특정 인플루언서의 특정 날짜 통계 보여줘
def get_statistics_by_date(connection, influencer_name, selected_date):
    """
    특정 인플루언서의 특정 날짜 기준 평균 통계 및 감정/주제/클러스터 분포를 문자열로 반환
    """
    date_data = get_youtube_comment_data(connection, influencer_name, selected_date)
    lines = [
        format_stat_dict(f"{influencer_name}의 {selected_date} 평균 통계", calculate_average_stats(date_data)),
        format_distribution(f"{influencer_name}의 {selected_date} 감정 분포", calculate_distribution(date_data, "emotion")),
        format_distribution(f"{influencer_name}의 {selected_date} 주제 분포", calculate_distribution(date_data, "topic")),
        format_distribution(f"{influencer_name}의 {selected_date} 클러스터 분포", calculate_distribution(date_data, "cluster")),
    ]
    return "\n".join(lines)

def get_statistics_by_video_url(connection, video_url):
    """
    특정 video_url(해시 or 전체 URL)에 대한 평균 통계 및 감정/주제/클러스터 분포를 문자열로 반환
    """
    # URL 해시값만 들어왔다면 전체 URL로 변환
    if not video_url.startswith("http"):
        video_url = f"https://www.youtube.com/watch?v={video_url}"

    video_data = get_youtube_comment_data(connection, video_url=video_url)

    if not video_data:
        return f"⚠️ 해당 영상({video_url})에 대한 데이터가 존재하지 않습니다."

    title = video_data[0]['title']
    date = video_data[0]['date']

    lines = [
        format_stat_dict(f"🎥 『{title}』 ({date}) 평균 통계", calculate_average_stats(video_data)),
        format_distribution(f"🎭 『{title}』 감정 분포", calculate_distribution(video_data, "emotion")),
        format_distribution(f"🧠 『{title}』 주제 분포", calculate_distribution(video_data, "topic")),
        format_distribution(f"👥 『{title}』 클러스터 분포", calculate_distribution(video_data, "cluster")),
    ]
    return "\n".join(lines)

def compare_two_influencers_dates(connection, influencer1,influencer2=None, date1=None, date2=None):
    """
    두 인플루언서의 특정 날짜 또는 전체 데이터를 비교하여 평균 통계 및 분포 차이를 문자열로 반환
    :param connection: DB 연결 객체
    :param influencer1: 첫 번째 인플루언서 이름
    :param date1: 첫 번째 인플루언서 날짜 (None이면 전체)
    :param influencer2: 두 번째 인플루언서 이름 (None이면 influencer1과 동일하게 비교)
    :param date2: 두 번째 인플루언서 날짜 (None이면 전체)
    :return: 비교 결과 문자열
    """
    if not influencer2:
        influencer2 = influencer1  # 같은 사람 비교

    result_lines = []

    # 🔹 데이터 가져오기
    data1 = get_youtube_comment_data(connection, influencer1, date1)
    data2 = get_youtube_comment_data(connection, influencer2, date2)

    # 🔹 타이틀 설정
    title = f"{influencer1}({date1 or '전체'}) vs {influencer2}({date2 or '전체'}) 평균 통계 비교"

    # 📊 평균 통계 비교
    avg_stats_1 = calculate_average_stats(data1)
    avg_stats_2 = calculate_average_stats(data2)
    result_lines.append(compare_two_stats(avg_stats_1, avg_stats_2))

    # 📊 분포 비교 (감정, 주제, 클러스터)
    for category in ["emotion", "topic", "cluster"]:
        dist_1 = calculate_distribution(data1, category)
        dist_2 = calculate_distribution(data2, category)
        result_lines.append(
            compare_distributions(
                dist_1, dist_2
                )
        )

    return "\n".join(result_lines)

def analyze_overall_fss_by_category(connection, influencer_name, top_n=3, min_count=5):
    """
    특정 인플루언서의 전체 콘텐츠에 대해 topic_categories별 평균 FSS 분석
    (Wikipedia URL은 항목 이름만 추출해 표시)
    """
    from collections import defaultdict

    video_data = get_youtube_comment_data(connection, influencer_name=influencer_name)
    
    if not video_data:
        return f"⚠️ [{influencer_name}]에 대한 댓글 데이터가 존재하지 않습니다."

    category_scores = defaultdict(list)

    for row in video_data:
        raw = row.get('topic_categories') or "미분류"
        score = row.get('SCOPE_score')
        if score is not None:
            category_scores[raw].append(score)

    filtered = {
        cat: scores for cat, scores in category_scores.items()
        if len(scores) >= min_count
    }

    if not filtered:
        return f"⚠️ 분석 가능한 주제 카테고리가 없습니다. (댓글 수 부족)"

    category_avg = {
        cat: sum(scores) / len(scores)
        for cat, scores in filtered.items()
    }

    sorted_avg = sorted(category_avg.items(), key=lambda x: x[1], reverse=True)
    top = sorted_avg[:top_n]
    bottom = sorted_avg[-top_n:]

    def simplify_category(cat):
        if cat == "미분류":
            return cat
        # 여러 개일 경우 분리 후 각각 처리
        parts = [part.strip() for part in cat.split(',')]
        simplified = []
        for p in parts:
            if "en.wikipedia.org/wiki/" in p:
                simplified.append(p.split("en.wikipedia.org/wiki/")[-1])
            else:
                simplified.append(p)
        return ", ".join(simplified)

    lines = [f"\n📈 [{influencer_name}]님의 콘텐츠 방향성 제안 (SCOPE_score 기반):"]

    lines.append("\n✅ 앞으로 더 활용해볼만한 주제 카테고리:")
    for cat, avg in top:
        lines.append(f"  - {simplify_category(cat)}: 평균 SCOPE_score {avg:.2f}")

    lines.append("\n⚠️ 반응이 낮았던 주제 카테고리:")
    for cat, avg in bottom:
        lines.append(f"  - {simplify_category(cat)}: 평균 SCOPE_score {avg:.2f}")

    return "\n".join(lines)

def ask_for_additional_analysis():
    user_input = input("\n🧠 추가 GPT 분석을 실행하시겠습니까? (y/n): ").strip().lower()
    return user_input == "y"

if __name__ == "__main__":
    connection = connect_to_db()

    try:
        rprint("[bold green]🧠 SCOPE 챗봇에 오신 것을 환영합니다! 'exit' 입력 시 종료됩니다.[/bold green]")

        while True:
            user_query = input("\n👤 사용자: ")

            if user_query.lower() in ["exit", "quit"]:
                rprint("[bold red]👋 챗봇을 종료합니다.[/bold red]")
                break

            elif user_query.strip().lower() == "cls":
                rprint("[bold yellow]✅ 컨텍스트가 초기화되었습니다. 새로운 질문을 해보세요.[/bold yellow]")
                continue

            try:
                # 함수 호출 추론
                call_string = ask_function_call(user_query)
                rprint(f"📌 호출할 함수: [cyan]{call_string}[/cyan]")

                # 함수 실행 (문자열 따옴표 제거 후 eval)
                true_call_string = call_string.strip('"')
                
                if true_call_string.startswith(("select_")):
                    result = eval(true_call_string)
                    rprint(f"📊 결과:\n{result}")

                elif true_call_string.startswith(("analyze_")):
                    result = eval(true_call_string)
                    rprint(f"📊 결과:\n{result}")

                    if ask_for_additional_analysis():
                        rprint("\n[bold magenta]GPT 분석 중...[/bold magenta]")
                        summary = analyze_contents_with_gpt(result)
                        rprint(f"\n🧾 GPT 분석 결과:\n{summary}")

                elif true_call_string.startswith(("get_")):
                    result = eval(true_call_string)
                    rprint(f"📊 결과:\n{result}")

                    if ask_for_additional_analysis():
                        rprint("\n[bold magenta]GPT 분석 중...[/bold magenta]")
                        summary = analyze_statistics_with_gpt(result)
                        rprint(f"\n🧾 GPT 분석 결과:\n{summary}")

                elif true_call_string.startswith(("compare_")):
                    result = eval(true_call_string)
                    rprint(f"📊 결과:\n{result}")

                    if ask_for_additional_analysis():
                        rprint("\n[bold magenta]GPT 분석 중...[/bold magenta]")
                        summary = analyze_comparison_with_gpt(user_query + result)
                        rprint(f"\n🧾 GPT 분석 결과:\n{summary}")

                elif true_call_string.startswith("comments_"):
                    comment_text = eval(true_call_string)
                    rprint(f"📊 결과:\n{comment_text}")

                    if ask_for_additional_analysis():
                        rprint("\n[bold magenta]GPT 분석 중...[/bold magenta]")
                        summary = analyze_comments_with_gpt(comment_text)
                        rprint(f"\n🧾 GPT 분석 결과:\n{summary}")

                else:
                    rprint(f"[dim]{true_call_string}[/dim]")

            except Exception as e:
                rprint(f"[bold red]⚠️ 오류 발생: {e}[/bold red]")

    finally:
        connection.close()