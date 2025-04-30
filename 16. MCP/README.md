# MCP 공부 여러 자료 정보

1. Base :

   https://github.com/weniv/mcp_book_source

2. LangGraph 활용 :

   https://github.com/netschool-kr/langgraph-mcp

## MCP
### 1. mcp.tool

tool은 LLM이 사용할 수 있는 도구 설정. ```@mcp.tool```

### 2. mcp.resource

resource는 통일된 패턴으로 자원에 안정적으로 접근 할 수 있도록 함. API의 경우 GET 요청과 유사. ```@mcp.resource```

### 3. mcp.prompt

prompt는 프롬프트를 확장하거나 프롬프트로 템플릿을 통해 정해진 패턴으로 질문할 때 사용. ```@mcp.prompt```

### 4. from mcp.server.fastmcp import Image

Image 객체는 응답을 이미지로 하기 위한 객체. 이미지처리에 자주 사용하는 pillow 라이브러리 사용.

### 5. from mcp.server.fastmcp import Context

Context는 다른 함수에서 resource에 접근할 때 사용.
