import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
from src_langgraph.utils import show_graph

load_dotenv()
if os.environ.get("EXEC_ENV") == "vscode":
    # VSCode에서 실행 중일 경우 (src 폴더 기준)
    server_script_path = "./src_langgraph_mcp/"
    python_command="C:\\Users\\user\\anaconda3\\envs\\langgraph-mcp\\python.exe"
else:
    # DOS 창에서 실행 중일 경우 (루트 디렉토리 기준)
    server_script_path = "./"
    python_command="python"
# Example query
# "What is weather in newyork"
# "What is FastMCP?"
# "summarize this youtube video in 50 words, here is a video link: https://www.youtube.com/watch?v=2f3K43FHRKo"
questions = [
    "서울의 내일 날씨 알려줘",
    "FastMCP가 무엇인가?",
    "다음 유튜브 영상 요약해줘: https://youtu.be/abcd1234"
]


# Define llm
model = ChatOpenAI(model="gpt-4o")

# Define MCP servers
async def run_agent():
    async with MultiServerMCPClient(
        {
            "tavily": {
                "command": python_command,
                "args": [server_script_path+"tavily_search.py"],
                "transport": "stdio",
            },
            "youtube_transcript": {
                "command": python_command,
                "args": [server_script_path+"yt_transcript.py"],
                "transport": "stdio",
            }, 
            # "weather": {
            # "url": "http://localhost:8000/sse", # start your weather server on port 8000
            # "transport": "sse",
            # }
        }
    ) as client:
        # Load available tools
        tools = client.get_tools()
        agent = create_react_agent(model, tools)
        show_graph(agent)
        # Add system message
        system_message = SystemMessage(content=(
                "You have access to multiple tools that can help answer queries. "
                "Use them dynamically and efficiently based on the user's request. "
        ))

        for q in questions:
            result = await agent.ainvoke({"messages": q})
            answer = result["messages"][-1].content  # 최종 Assistant 답변 메시지
            print(f"Q: {q}\nA: {answer}\n{'-'*40}")

# Run the agent
if __name__ == "__main__":
    asyncio.run(run_agent())
