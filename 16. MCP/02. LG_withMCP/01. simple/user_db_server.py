from mcp.server.fastmcp import FastMCP

# "UserService" 이름의 MCP 서버 생성
mcp = FastMCP("UserDBService")

# 사용자 ID -> 이름 조회 툴 등록
@mcp.tool()
def get_user_name(user_id: int) -> str:
    """주어진 사용자 ID에 해당하는 이름을 반환한다."""
    users = {1: "Alice", 2: "Bob"}
    return users.get(user_id, f"Unknown user (ID: {user_id})")

if __name__ == "__main__":
    # 표준입출력(stdio) 기반 MCP 서버 실행 (동일 머신 내 프로세스 통신)
    mcp.run(transport="stdio")
