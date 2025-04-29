# chatbot_server.py - MCP 서버: SSE 방식으로 웹 검색 및 답변 생성 툴 정의
import os
from dotenv import load_dotenv, find_dotenv

# .env 파일에서 OPENAI_API_KEY, SERPAPI_API_KEY 등을 로드
load_dotenv(find_dotenv())
openai_api_key = os.getenv('OPENAI_API_KEY')
serpapi_api_key = os.getenv('SERPAPI_API_KEY')

# OpenAI API 키 설정
import openai
openai.api_key = openai_api_key

# FastMCP 및 SerpAPIWrapper 임포트
from mcp.server.fastmcp import FastMCP
from langchain_community.utilities import SerpAPIWrapper

# SerpAPI 검색 래퍼 초기화
search_tool_api = SerpAPIWrapper()

# MCP 서버 인스턴스 생성 (서비스 이름: ChatbotService)
mcp = FastMCP("ChatbotService")

# [1] 웹 검색 도구 등록
@mcp.tool()
def search_web(query: str) -> str:
    """주어진 질의에 대해 웹 검색을 수행하고 결과를 문자열로 반환한다."""
    try:
        result = search_tool_api.run(query)
    except Exception as e:
        result = f"(검색 중 오류 발생: {e})"
    return result

# [2] 답변 생성 도구 등록
@mcp.tool()
def generate_answer(query: str, search_results: str) -> str:
    """사용자 질문과 검색 정보를 활용하여 답변을 생성한다."""
    if search_results and search_results.strip():
        user_content = (
            f"사용자 질문: {query}\n\n"
            f"다음은 웹 검색 정보입니다:\n{search_results}\n\n"
            "이 정보를 참고하여 질문에 답변해 주세요."
        )
    else:
        user_content = f"사용자 질문: {query}"
    system_content = (
        "당신은 다중 단계를 거쳐 질문에 답하는 지능형 비서입니다. "
        "가능한 정확하고 간결하게 답변하세요."
    )
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]
    try:
        completion = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        answer_text = completion.choices[0].message.content
    except Exception as e:
        answer_text = f"답변 생성 중 오류가 발생했습니다: {e}"
    return answer_text

# --- SSE 서버 구성 (FastAPI 사용) ---
from fastapi import FastAPI, Request, HTTPException
from sse_starlette.sse import EventSourceResponse
import asyncio

app = FastAPI()

@app.post("/sse")
async def sse_invoke(request: Request):
    """
    클라이언트로부터 tool 이름과 파라미터를 포함한 JSON을 POST로 받으면
    해당 MCP 도구를 호출하고 결과를 SSE 이벤트로 전송한다.
    """
    payload = await request.json()
    tool_name = payload.get("tool")
    params = payload.get("params", {})
    if not tool_name:
        raise HTTPException(status_code=400, detail="tool 파라미터 필요")
    
    # tool_name에 따라 MCP 도구 호출 (필요시 추가 도구 등록 가능)
    if tool_name == "search_web":
        result = search_web(**params)
    elif tool_name == "generate_answer":
        result = generate_answer(**params)
    else:
        result = f"알 수 없는 도구: {tool_name}"
    
    async def event_generator():
        # SSE 프로토콜에 따라 data: <내용> 형식으로 한 줄 전송 후 종료
        yield f"data: {result}\n\n"
        await asyncio.sleep(0.1)
    
    return EventSourceResponse(event_generator())

# FastAPI 서버 실행 (uvicorn 사용)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
