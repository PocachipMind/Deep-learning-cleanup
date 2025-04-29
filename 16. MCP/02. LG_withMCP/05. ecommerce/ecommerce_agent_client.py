# ecommerce_agent_client.py - MCP 클라이언트 및 LangGraph 에이전트 설정
import os
from dotenv import load_dotenv, find_dotenv

# .env 파일에서 OPENAI_API_KEY로드
load_dotenv(find_dotenv())
openai_api_key = os.getenv('OPENAI_API_KEY')

# OpenAI API 키 설정
import openai
openai.api_key = openai_api_key

import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI  # OpenAI GPT-4 모델 (LangChain OpenAI wrapper)
from src_langgraph.utils import show_graph

# 실행 환경에 따라 경로 설정
if os.environ.get("EXEC_ENV") == "vscode":
    # VSCode에서 실행 중일 경우 (src 폴더 기준)
    server_script = "./src_langgraph_mcp/ecommerce_service_server.py"
    python_command="C:\\Users\\user\\anaconda3\\envs\\langgraph-mcp\\python.exe"
else:
    # DOS 창에서 실행 중일 경우 (루트 디렉토리 기준)
    server_script = "./ecommerce_service_server.py"
    python_command="python"

async def main():
    # 1. MCP 서버 프로세스를 STDIO 모드로 실행하기 위한 파라미터 설정
    server_params = StdioServerParameters(
        command=python_command,
        args=[server_script]  # MCP 서버 스크립트 경로
    )
    # 2. MCP 서버에 STDIO 클라이언트로 접속하여 세션 시작 
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 3. MCP 세션 초기화 (서버와 초기 메타데이터 교환)
            await session.initialize()
            # 4. 서버로부터 사용 가능한 툴 목록 불러오기 
            tools = await load_mcp_tools(session)
            # 5. LLM 모델과 툴을 포함한 LangGraph 에이전트 생성
            model = ChatOpenAI(model="gpt-4")  # GPT-4 모델 인스턴스 (API 키 필요)
            agent = create_react_agent(model, tools)
            show_graph(agent)
            # 6. 에이전트를 이용하여 사용자 질의 처리 
            query = {"messages": "재고 있는 저렴한 노트북 추천해줘"}
            result = await agent.ainvoke(query)
            # 7. 결과 출력 
            for message in result["messages"]:
                print(message.content)

# 비동기 함수 실행
if __name__ == "__main__":
    asyncio.run(main())
