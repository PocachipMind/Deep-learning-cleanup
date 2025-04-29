# chatbot_server.py - MCP 서버: 웹 검색 및 답변 생성 툴 정의
import os
from dotenv import load_dotenv, find_dotenv

# 환경 변수(.env 파일) 로드 - OPENAI_API_KEY, SERPAPI_API_KEY 등이 설정되어 있어야 함
load_dotenv(find_dotenv())
openai_api_key = os.getenv('OPENAI_API_KEY')
serpapi_api_key = os.getenv('SERPAPI_API_KEY')

# OpenAI API 키 설정
import openai
openai.api_key = openai_api_key

# MCP 서버 및 툴 데코레이터 임포트
from mcp.server.fastmcp import FastMCP

# SerpAPI 검색 래퍼 임포트 (LangChain 사용)
from langchain_community.utilities import SerpAPIWrapper

# SerpAPI 검색 툴 초기화 (환경 변수 SERPAPI_API_KEY 자동 사용)
search_tool_api = SerpAPIWrapper()

# MCP 서버 인스턴스 생성 (서비스 이름은 "ChatbotService")
mcp = FastMCP("ChatbotService")

# 툴 정의: 웹 검색 기능
@mcp.tool()
def search_web(query: str) -> str:
    """주어진 질의에 대해 웹 검색을 수행하고 결과를 문자열로 반환한다."""
    try:
        result = search_tool_api.run(query)
    except Exception as e:
        result = f"(검색 중 오류 발생: {e})"
    return result

# 툴 정의: 답변 생성 기능
@mcp.tool()
def generate_answer(query: str, search_results: str) -> str:
    """사용자 질문과 검색 정보를 활용하여 답변을 생성한다."""
    # 사용자 프롬프트 구성
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
        # OpenAI ChatCompletion API 호출하여 답변 생성 (기본 모델: gpt-3.5-turbo)
        completion = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        answer_text = completion.choices[0].message.content
    except Exception as e:
        answer_text = f"답변 생성 중 오류가 발생했습니다: {e}"
    return answer_text

if __name__ == "__main__":
    # MCP 서버 실행 (표준입출력 transport 사용)
    mcp.run(transport="stdio")
