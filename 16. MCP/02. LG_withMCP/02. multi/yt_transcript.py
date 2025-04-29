# yt_transcript.py
from mcp.server.fastmcp import FastMCP
from youtube_transcript_api import YouTubeTranscriptApi
import re

# FastMCP 서버 인스턴스 생성 (서비스 이름: "YouTubeService")
mcp = FastMCP("YouTubeService")

# 유튜브 자막 툴 정의 및 등록
@mcp.tool()
def get_transcript(video_url: str) -> str:
    """지정한 YouTube 영상의 전체 자막을 하나의 문자열로 반환합니다."""
    try:
        # URL에서 비디오 ID 추출 (youtu.be 단축 URL 혹은 v= 파라미터)
        match = re.search(r"(?:v=|youtu\.be/)([^&/\n?]+)", video_url)
        video_id = match.group(1) if match else video_url  # 매치되지 않으면 입력을 ID로 간주
        # YouTubeTranscriptApi를 통해 자막 리스트 가져오기 (한국어 우선, 없으면 영어)
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['ko', 'en'])
        # 각 자막 조각의 'text' 필드 추출 후 공백으로 연결
        transcript_text = " ".join(seg['text'] for seg in transcript_list)
        return transcript_text
    except Exception as e:
        return f"(자막 추출 오류: {e})"

if __name__ == "__main__":
    mcp.run(transport="stdio")
