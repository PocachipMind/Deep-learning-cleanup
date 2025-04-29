# ecommerce_service_server.py - MCP 서버 설정 (제품 추천 & 재고 확인 서비스)
from mcp.server.fastmcp import FastMCP

# "ECommerceService" 이름의 MCP 서버 생성
mcp = FastMCP("ECommerceService")

# 간단한 제품 데이터 (ID: {상품명, 카테고리, 가격, 재고수량})
PRODUCT_DB = {
    1: {"name": "게이밍 노트북 X", "category": "노트북", "price": 950, "stock": 5},
    2: {"name": "비즈니스 노트북 Y", "category": "노트북", "price": 850, "stock": 0},
    3: {"name": "스마트폰 A", "category": "스마트폰", "price": 700, "stock": 10},
    4: {"name": "게이밍 마우스 Z", "category": "주변기기", "price": 50, "stock": 100}
}

@mcp.tool()
def recommend_products(query: str) -> list:
    """사용자 질의에 맞는 상품 추천 리스트를 반환한다."""
    # 간단한 키워드 매칭을 통해 추천 (실제 시나리오에서는 복잡한 추천 알고리즘 사용)
    results = []
    query_lower = query.lower()
    # 예시: "노트북", "저렴한" 등의 키워드로 필터링
    for pid, info in PRODUCT_DB.items():
        name = info["name"]
        price = info["price"]
        category = info["category"]
        if ("노트북" in query and category == "노트북") or \
           ("스마트폰" in query and category == "스마트폰"):
            # 가격 조건 예시: "저렴" 또는 숫자 가격 범위가 언급되면 필터링
            if "저렴" in query or "싼" in query or "이하" in query:
                # 편의상 1000 (만원 기준) 또는 1000달러 이하로 해석
                if price <= 1000:
                    results.append({"id": pid, "name": name, "price": price})
            else:
                results.append({"id": pid, "name": name, "price": price})
    # 추천 결과를 3개까지 잘라서 반환
    return results[:3]

@mcp.tool()
def check_inventory(product_id: int) -> str:
    """주어진 상품 ID의 재고 상태를 반환한다."""
    product = PRODUCT_DB.get(product_id)
    if product is None:
        return f"상품 ID {product_id}를 찾을 수 없습니다."
    stock = product["stock"]
    if stock > 0:
        return f"{stock}개 재고 있음"
    else:
        return "품절됨"

if __name__ == "__main__":
    # 표준입출력 기반 MCP 서버 실행 (클라이언트와 동일 프로세스 통신 준비)
    mcp.run(transport="stdio")
