# chatbot_client.py - LangGraph 클라이언트: MCP 툴 연동 대화 에이전트
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List, Annotated
import operator

# 상태 데이터 구조 정의
class State(TypedDict):
    user_input: str                                      # 사용자의 질문
    search_results: Annotated[List[str], operator.add]   # 검색 결과 목록 (누적 가능)
    final_answer: str                                    # 최종 답변

def decide_next_step(state: State) -> dict:
    """사용자 질문을 분석해 'search' 또는 'answer' 중 다음 스텝 결정"""
    query = state["user_input"].lower()
    # 특정 키워드가 질문에 포함되면 검색 단계를 우선 수행
    NEED_SEARCH_KEYWORDS = ["날씨", "오늘", "내일", "어제", "현재", "검색"]
    if any(keyword in query for keyword in NEED_SEARCH_KEYWORDS):
        return {"route": "search"}   # 검색 노드로 분기
    else:
        return {"route": "answer"}   # 바로 답변 생성 노드로 분기

import os

if os.environ.get("EXEC_ENV") == "vscode":
    # VSCode에서 실행 중일 경우 (src 폴더 기준)
    server_script = "./src_langgraph_mcp/chatbot_server.py"
    python_command="C:\\Users\\user\\anaconda3\\envs\\langgraph-mcp\\python.exe"
else:
    # DOS 창에서 실행 중일 경우 (루트 디렉토리 기준)
    server_script = "./chatbot_server.py"
    python_command="python"

async def main():
    # 1. MCP 서버 프로세스를 STDIO 모드로 실행 (서버 스크립트 경로 지정)
    server_params = StdioServerParameters(command=python_command, args=[server_script])
    # 2. MCP 서버에 연결하여 세션 시작
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            # 3. 서버로부터 사용 가능한 툴 목록 가져오기
            tools = await load_mcp_tools(session)
            #    불러온 툴 중에서 필요한 툴 선택
            search_tool = next(t for t in tools if t.name == "search_web")
            answer_tool = next(t for t in tools if t.name == "generate_answer")
            # 4. 그래프 노드 함수 정의 (원격 MCP 툴 호출 구현)
            async def search_node(state: State) -> dict:
                query = state["user_input"]
                try:
                    result_str = await search_tool.ainvoke({"query": query})
                except Exception as e:
                    result_str = f"(검색 중 오류 발생: {e})"
                return {"search_results": [result_str]}
            
            async def answer_node(state: State) -> dict:
                query = state["user_input"]
                # 이전 검색 결과들을 하나의 문자열로 결합
                search_info = "\n".join(state.get("search_results", []))
                try:
                    answer_text = await answer_tool.ainvoke({
                        "query": query,
                        "search_results": search_info
                    })
                except Exception as e:
                    answer_text = f"답변 생성 중 오류가 발생했습니다: {e}"
                return {"final_answer": answer_text}
            
            # 5. 상태 그래프 생성 및 노드 추가
            graph = StateGraph(State)
            graph.add_node("router", decide_next_step)   # 라우터 노드
            graph.add_node("search_node", search_node)   # 검색 노드
            graph.add_node("answer_node", answer_node)   # 답변 생성 노드
            # 6. 노드 간 엣지 설정 (조건부 분기 및 순차 흐름)
            graph.add_edge(START, "router")
            graph.add_conditional_edges(
                "router",
                lambda state: state["route"],
                {"search": "search_node", "answer": "answer_node"}
            )
            graph.add_edge("search_node", "answer_node")
            graph.add_edge("answer_node", END)
            graph = graph.compile()
            # 7. 그래프 실행 예시
            user_question = "서울의 내일 날씨는?"
            initial_state = {"user_input": user_question}
            result_state = await graph.ainvoke(initial_state)
            # 8. 결과 출력
            print("질문:", user_question)
            print("검색 결과 요약:", result_state.get("search_results"))
            print("답변:", result_state.get("final_answer"))

# 비동기 메인 함수 실행
if __name__ == "__main__":
    asyncio.run(main())
