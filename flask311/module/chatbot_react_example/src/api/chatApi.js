// api/chatApi.js
import axios from 'axios';

export const fetchChat = async (query, useGpt = false) => {
  try {
    const response = await axios.post("http://3.34.90.217:5000/chat", {
      query: query,
      gpt: useGpt
    });

    return response.data; // { function_call, result, gpt_summary }
  } catch (error) {
    console.error("챗봇 요청 실패:", error);
    throw error;
  }
};
