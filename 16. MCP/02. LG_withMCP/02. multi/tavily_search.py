# tavily_search.py
from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient
import os
from dotenv import load_dotenv, find_dotenv

# .env 파일에서 TAVILY_API_KEY, SERPAPI_API_KEY 등을 로드
load_dotenv(find_dotenv())
# Tavily API 키 설정 (환경변수 또는 .env 활용)
api_key = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=api_key)

# FastMCP 서버 인스턴스 생성 (서비스 이름: "WebSearchService")
mcp = FastMCP("WebSearchService")

# 웹 검색 툴 정의 및 등록
@mcp.tool()
def search_web(query: str) -> str:
    """질문에 대한 최신 웹 검색 결과를 요약하여 반환합니다."""
    try:
        response = tavily_client.search(query)  # Tavily API로 웹 검색 실행
        # Tavily 응답에서 직접 답변이 있으면 사용, 없으면 결과 내용 사용
        if isinstance(response, dict):
            if "answer" in response and response["answer"]:
                return response["answer"]
            if "results" in response and response["results"]:
                top = response["results"][0]  # 가장 첫 번째 결과 활용
                return f"{top.get('title')}: {top.get('content')}"
        # 응답이 사전 구조가 아니거나 처리할 내용이 없으면 문자열로 변환
        return str(response)
    except Exception as e:
        return f"(검색 오류: {e})"

if __name__ == "__main__":
    mcp.run(transport="stdio")
