import React, { useState } from "react";
import { fetchChat } from "../api/chatApi";
import TypingLine from "../components/TypingLine";

const ChatPage = () => {
  const [query, setQuery] = useState("");
  const [chatLog, setChatLog] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  const exampleCommands = [
    { label: "전체 인플루언서 통계 보기", command: "전체 인플루언서 통계를 보여줘" },
    { label: "특정일자 통계 보기", command: "[특정 인플루언서]의 [n월 m일] 통계를 보여줘" },
    { label: "최고 반응일 통계 보기", command: "[특정 인플루언서]의 최고성과일 통계보여줘" },
    { label: "최근 평균 통계 보기", command: "[특정 인플루언서]의 최근 평균 통계보여줘" },
    { label: "인플루언서 비교", command: "[인플루언서1]과 [인플루언서2]의 통계를 비교해줘" },
    { label: "칭찬 댓글 보기", command: "[특정 인플루언서]의 칭찬 댓글 분석해줘" },
    { label: "콘텐츠 방향 추천", command: "[특정 인플루언서]의 컨텐츠 방향성 제시해줘" },
    { label: "영상 목록 보기", command: "[특정 인플루언서]의 영상 URL 목록 보여줘" },
    { label: "SCOPE란?", command: "SCOPE 설명해줘" },
    { label: "통계 가능한 날짜 보기", command: "[특정 인플루언서의] 통계자료 중 사용가능한 날짜 보여줘" },
  ];

  const handleSubmit = async () => {
    if (!query.trim()) return;

    setIsLoading(true);

    try {
      const response = await fetchChat(query, true);

      const newEntry = {
        question: query,
        functionCall: response.function_call,
        result: response.result,
        gptSummary: response.gpt_summary,
      };

      setChatLog((prev) => [...prev, newEntry]);
      setQuery("");
    } catch (error) {
      console.error("챗봇 응답 실패:", error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div style={{ padding: "20px" }}>
      <h2>SCOPE 챗봇</h2>
      {/* 대화 로그 */}  
      <div>
        {chatLog.map((entry, idx) => (
          <div key={idx} style={{ marginBottom: "20px", borderBottom: "1px solid #ccc", paddingBottom: "10px" }}>
            <p><strong>🙋 질문:</strong> {entry.question}</p>
            <p><strong>📌 호출된 함수:</strong> {entry.functionCall}</p>
            {entry.result && (
              <p><strong>📊 통계 결과:</strong><br />
                <TypingLine text={entry.result} />
              </p>
            )}
            {entry.gptSummary && (
              <p><strong>🧠 GPT 요약:</strong><br />
                <TypingLine text={entry.gptSummary} />
              </p>
            )}
          </div>
        ))}

        {/* 응답 대기 중이면 스피너 표시 */}
        {isLoading && (
          <div style={{ marginBottom: "20px", borderBottom: "1px solid #ccc", paddingBottom: "10px" }}>
            <p><strong>🙋 질문:</strong> {query}</p>
            <p><strong>📌 응답 대기 중...</strong></p>
            <TypingLine text="" />
          </div>
        )}
      </div>


      {/* 예시 명령어 버튼 */}
      <div style={{ marginBottom: "15px", display: "flex", flexWrap: "wrap", gap: "8px" }}>
        {exampleCommands.map((ex, idx) => (
          <button
            key={idx}
            onClick={() => setQuery(ex.command)}
            style={{
              padding: "6px 10px",
              fontSize: "13px",
              borderRadius: "4px",
              border: "1px solid #ccc",
              backgroundColor: "#f9f9f9",
              cursor: "pointer"
            }}
          >
            {ex.label}
          </button>
        ))}
      </div>

      {/* 입력창 */}
      <div style={{ marginBottom: "20px" }}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
          placeholder="질문을 입력하세요..."
          style={{ width: "70%", padding: "8px" }}
        />
        <button
          onClick={handleSubmit}
          disabled={isLoading}
          style={{ marginLeft: "10px", padding: "8px 16px" }}
        >
          {isLoading ? "분석 중..." : "전송"}
        </button>
      </div>

      
    </div>
  );
};

export default ChatPage;
