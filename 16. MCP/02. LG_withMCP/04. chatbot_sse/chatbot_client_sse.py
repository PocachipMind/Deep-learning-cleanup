# chatbot_client.py - LangGraph 클라이언트: SSE 방식 MCP 툴 연동 대화 에이전트
import asyncio
import aiohttp  # aiohttp를 이용하여 SSE 스트림 수신
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List
import operator

# 상태 데이터 구조 정의
class State(TypedDict):
    user_input: str                # 사용자의 질문
    search_results: List[str]      # 검색 결과 목록 (누적 가능)
    final_answer: str              # 최종 답변

def decide_next_step(state: State) -> dict:
    """사용자 질문을 분석해 'search' 또는 'answer' 중 다음 스텝 결정"""
    query = state["user_input"].lower()
    NEED_SEARCH_KEYWORDS = ["날씨", "오늘", "내일", "어제", "현재", "검색"]
    if any(keyword in query for keyword in NEED_SEARCH_KEYWORDS):
        return {"route": "search"}
    else:
        return {"route": "answer"}

async def sse_invoke(tool_name: str, params: dict) -> str:
    """
    aiohttp를 통해 SSE 엔드포인트에 POST 요청을 보내고, 
    SSE 스트림으로 전달된 결과 문자열을 수신한다.
    """
    url = "http://localhost:8000/sse"
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json={"tool": tool_name, "params": params}) as resp:
            if resp.status != 200:
                return f"오류 발생: 상태 코드 {resp.status}"
            result = ""
            # SSE 스트림은 text/event-stream 형식의 데이터를 보낸다.
            async for line in resp.content:
                decoded_line = line.decode('utf-8').strip()
                # "data:"로 시작하는 줄이 SSE 이벤트 데이터
                if decoded_line.startswith("data:"):
                    result = decoded_line[len("data:"):].strip()
                    break
            return result

async def main():
    # [1] 검색 노드: 사용자 질문을 대상으로 웹 검색 도구 호출 (search_web)
    async def search_node(state: State) -> dict:
        query = state["user_input"]
        try:
            result_str = await sse_invoke("search_web", {"query": query})
        except Exception as e:
            result_str = f"(검색 중 오류 발생: {e})"
        return {"search_results": [result_str]}
    
    # [2] 답변 생성 노드: 사용자 질문과 검색 결과를 이용해 답변 생성 도구 호출 (generate_answer)
    async def answer_node(state: State) -> dict:
        query = state["user_input"]
        search_info = "\n".join(state.get("search_results", []))
        try:
            answer_text = await sse_invoke("generate_answer", {
                "query": query,
                "search_results": search_info
            })
        except Exception as e:
            answer_text = f"답변 생성 중 오류가 발생했습니다: {e}"
        return {"final_answer": answer_text}
    
    # [3] 상태 그래프 구성 (router → search_node → answer_node)
    graph = StateGraph(State)
    graph.add_node("router", decide_next_step)   # 라우터 노드: 검색 필요 여부 결정
    graph.add_node("search_node", search_node)     # 검색 노드
    graph.add_node("answer_node", answer_node)     # 답변 생성 노드
    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        lambda state: state["route"],
        {"search": "search_node", "answer": "answer_node"}
    )
    graph.add_edge("search_node", "answer_node")
    graph.add_edge("answer_node", END)
    graph = graph.compile()
    
    # [4] 그래프 실행 (예제: "서울의 내일 날씨는?")
    user_question = "서울의 내일 날씨는?"
    initial_state = {"user_input": user_question}
    result_state = await graph.ainvoke(initial_state)
    
    # [5] 결과 출력
    print("질문:", user_question)
    print("검색 결과 요약:", result_state.get("search_results"))
    print("답변:", result_state.get("final_answer"))

# 비동기 메인 함수 실행
if __name__ == "__main__":
    asyncio.run(main())
