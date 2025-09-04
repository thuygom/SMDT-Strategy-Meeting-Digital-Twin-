import React from "react";
import { Typewriter } from "react-simple-typewriter";
import Spinner from "./Spinner"; // 스피너 가져오기

const TypingLine = ({ text }) => {
  if (!text || text.trim() === "") {
    return (
      <div style={{ display: "flex", alignItems: "center", color: "#666", fontStyle: "italic" }}>
        <Spinner />
        GPT가 생각 중입니다...
      </div>
    );
  }

  return (
    <span style={{ whiteSpace: "pre-wrap" }}>
      <Typewriter
        words={[text]}
        loop={1}
        cursor
        cursorStyle="|"
        typeSpeed={15}
        deleteSpeed={0}
        delaySpeed={800}
      />
    </span>
  );
};

export default TypingLine;
