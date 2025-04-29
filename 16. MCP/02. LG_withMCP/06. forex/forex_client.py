from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph import StateGraph, START, END

import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools

import os
# 실행 환경에 따라 경로 설정
if os.environ.get("EXEC_ENV") == "vscode":
    # VSCode에서 실행 중일 경우 (src 폴더 기준)
    server_script = "./src_langgraph_mcp/forex_server.py"
    python_command="C:\\Users\\user\\anaconda3\\envs\\langgraph-mcp\\python.exe"
else:
    # DOS 창에서 실행 중일 경우 (루트 디렉토리 기준)
    server_script = "./forex_server.py"
    python_command="python"

# 1. 상태 구조 정의
class ForexState(TypedDict):
    pair: str
    prices: list[float]
    short_window: int
    long_window: int
    short_ma: float
    long_ma: float
    signal: str  # "BUY", "SELL", "HOLD"

# 2. 노드 함수 구현
# 환율 수집 노드 – MCP 툴 호출로 현재 환율 획득
# def fetch_rate_node(state: ForexState) -> ForexState:
#     # MCP 툴 호출 (모의 구현: 실제론 MCP client 통해 서버의 get_rate 호출)
#     pair = state["pair"]
#     # 여기서는 예제를 위해 랜덤 값으로 시뮬레이션하거나 MCP 없이 임의 생성
#     import random
#     latest_price = round(random.uniform(0.98, 1.02), 4)  # 예: 0.98~1.02 범위의 임의 환율
#     state["prices"].append(latest_price)
#     # long_window 길이 초과 시 가장 오래된 데이터 제거 (롤링 윈도우 유지)
#     if len(state["prices"]) > state["long_window"]:
#         state["prices"].pop(0)
#     return state
def fetch_rate_node(state: ForexState) -> ForexState:
    pair = state["pair"]
    try:
        # MCP 클라이언트 세션에서 get_rate 호출 (동기적 사용 예시)
        result = asyncio.run(fetch_rate_via_mcp(pair))
        state["prices"].append(result)
        # 윈도우 관리
        if len(state["prices"]) > state["long_window"]:
            state["prices"].pop(0)
    except Exception as e:
        print(f"MCP fetch error: {e}")
        
    return state

# 이동평균 계산 노드 – 단기/장기 이동평균 산출
def compute_ma_node(state: ForexState) -> ForexState:
    prices = state["prices"]
    sw, lw = state["short_window"], state["long_window"]
    # 단기 이동평균 계산
    if len(prices) >= sw:
        state["short_ma"] = sum(map(float, prices[-sw:])) / sw
    else:
        state["short_ma"] = sum(map(float, prices)) / len(prices)  # 데이터가 부족하면 이용 가능한 전체 평균
    # 장기 이동평균 계산
    if len(prices) >= lw:
        state["long_ma"] = sum(map(float, prices[-lw:])) / lw
    else:
        state["long_ma"] = sum(map(float, prices)) / len(prices)
    return state

# 신호 판단 노드 – 이동평균 비교하여 매수/매도/관망 신호 결정
def signal_check_node(state: ForexState) -> ForexState:
    short_ma = state["short_ma"]
    long_ma = state["long_ma"]
    # 단순 교차 비교로 신호 결정
    if short_ma > long_ma:
        state["signal"] = "BUY"
    elif short_ma < long_ma:
        state["signal"] = "SELL"
    else:
        state["signal"] = "HOLD"
    return state

# 이메일 알림 노드 – BUY 또는 SELL 신호 시 Gmail SMTP 이메일 전송
def send_email_node(state: ForexState) -> ForexState:
    import os, smtplib
    from email.message import EmailMessage
    # 환경변수 (.env)에서 SMTP 인증 정보 로드
    smtp_user = os.getenv("GMAIL_USER")
    smtp_pass = os.getenv("GMAIL_PASS")
    alert_to  = os.getenv("ALERT_EMAIL", smtp_user)  # 받는 사람 이메일 (지정 없으면 자기 자신)
    if not smtp_user or not smtp_pass:
        print("⚠️ 이메일 발송을 위한 SMTP 계정 정보가 설정되지 않았습니다.")
        return state
    # 이메일 메시지 구성
    msg = EmailMessage()
    signal = state["signal"]
    pair = state["pair"]
    price = state["prices"][-1] if state["prices"] else None
    msg["Subject"] = f"[Forex Signal] {pair} - {signal}"
    msg["From"] = smtp_user
    msg["To"] = alert_to
    body = f"Trading signal for {pair}: {signal}\nLatest price = {price}\nShort MA = {state['short_ma']}, Long MA = {state['long_ma']}"
    msg.set_content(body)
    # Gmail SMTP 서버에 연결하여 이메일 보내기
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()  # TLS 암호화 시작
        server.login(smtp_user, smtp_pass)
        server.send_message(msg)
        server.quit()
        print(f"✅ Email alert sent: {signal} @ price {price}")
    except Exception as e:
        print(f"❌ Email send failed: {e}")
    return state

# 3. 그래프 빌드 및 엣지 구성
workflow = StateGraph(ForexState)
# 노드 등록
workflow.add_node("fetch_rate", fetch_rate_node)
workflow.add_node("compute_ma", compute_ma_node)
workflow.add_node("signal_check", signal_check_node)
workflow.add_node("send_email", send_email_node)
# 엣지 연결: START -> fetch_rate -> compute_ma -> signal_check
workflow.add_edge(START, "fetch_rate")
workflow.add_edge("fetch_rate", "compute_ma")
workflow.add_edge("compute_ma", "signal_check")
# 조건부 엣지: signal_check 후 신호에 따라 send_email 또는 종료
def route_signal_edge(state: ForexState) -> str:
    # "ALERT" if buy/sell signal, otherwise "NO_ALERT"
    return "ALERT" if state["signal"] != "HOLD" else "NO_ALERT"
workflow.add_conditional_edges(
    "signal_check",            # 분기 출발 노드
    route_signal_edge,         # 분기 결정 함수
    {"ALERT": "send_email",    # ALERT이면 send_email 노드로
     "NO_ALERT": END}          # NO_ALERT이면 그래프 종료
)
workflow.add_edge("send_email", END)

# 4. 그래프 컴파일 (실행 가능한 상태로 변환)
agent = workflow.compile()

# MCP 서버 접속 설정: forex_server.py를 서브프로세스로 실행
server_params = StdioServerParameters(
    command=python_command,
    args=[server_script],  # MCP 서버 스크립트 경로
)
# 비동기 컨텍스트에서 MCP 세션 열기 및 툴 호출
async def fetch_rate_via_mcp(pair: str):
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # MCP 연결 초기화
            await session.initialize()
            #서버로부터 사용 가능한 툴 목록 가져오기
            tools = await load_mcp_tools(session)
            get_rate = next(t for t in tools if t.name == "get_rate")
            # ForexServer의 get_rate 툴 호출
            result = await get_rate.ainvoke({"pair": pair})
            return result

# 예시 실행
pair = "EUR/USD"
rate = asyncio.run(fetch_rate_via_mcp(pair))
print(f"{pair} 현재 환율: {rate}")

import random
random.seed(42)  # 예측 가능하도록 랜덤 시드 고정 (재현용)

# 에이전트 초기 상태 정의
state = {
    "pair": "EUR/USD",
    "prices": [],
    "short_window": 3,
    "long_window": 5,
    "short_ma": 0.0,
    "long_ma": 0.0,
    "signal": "HOLD"
}

# 10회 반복 실행 시나리오
for i in range(1, 11):
    print(f"\n=== Iteration {i} ===")
    state = agent.invoke(state)  # 그래프 실행
    latest_price = state["prices"][-1]
    short_ma = state["short_ma"]
    long_ma = state["long_ma"]
    signal = state["signal"]
    print(f"New price = {latest_price}")
    print(f"Short MA (window={state['short_window']}) = {short_ma}")
    print(f"Long  MA (window={state['long_window']}) = {long_ma}")
    print(f"Signal = {signal}")
