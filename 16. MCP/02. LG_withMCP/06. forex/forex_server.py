# forex_server.py - FastMCP를 이용한 모의 Forex MCP 서버
from mcp.server.fastmcp import FastMCP

# MCP 서버 인스턴스 생성
mcp = FastMCP("ForexServer")

# 환율 조회 MCP 툴 정의 (모의 구현)
@mcp.tool()
def get_rate(pair: str = "EUR/USD") -> float:
    """주어진 통화쌍의 현재 환율을 반환 (모의 데이터)"""
    import random
    # 예시로 EUR/USD 환율을 0.95~1.05 사이의 랜덤값으로 생성
    rate = round(random.uniform(0.95, 1.05), 4)
    print(f"[MCP] {pair} rate = {rate}")
    return rate

if __name__ == "__main__":
    # MCP 서버 실행 (표준 I/O transport 사용)
    mcp.run(transport="stdio")
