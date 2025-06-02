"""
YouTube 연동 콘텐츠 추천 시스템 사용 예시
MODIFIED 2024-01-20: YouTube API 연동 기능 테스트 및 사용법 안내
"""
import asyncio
import requests
import json

# API 서버 기본 URL
BASE_URL = "http://localhost:8000"

async def test_youtube_recommendations():
    """YouTube 연동 추천 시스템 테스트"""
    
    print("=" * 60)
    print("YouTube 연동 콘텐츠 추천 시스템 테스트")
    print("=" * 60)
    
    # 1. 기본 추천 요청 (YouTube 포함)
    print("\n1. 기본 추천 요청 (YouTube 포함)")
    print("-" * 40)
    
    recommend_data = {
        "keywords": ["파이썬", "머신러닝"],
        "content_types": ["book", "movie", "video", "youtube_video"],
        "max_items": 10,
        "include_youtube": True,
        "youtube_max_per_keyword": 3
    }
    
    try:
        response = requests.post(f"{BASE_URL}/recommend", json=recommend_data)
        result = response.json()
        
        print(f"총 추천 수: {result['total_count']}")
        print(f"YouTube 포함 여부: {result['youtube_included']}")
        print(f"소스별 요약: {result['sources_summary']}")
        
        print("\n추천 목록:")
        for i, item in enumerate(result["recommendations"], 1):
            print(f"{i}. {item['title']}")
            print(f"   타입: {item['content_type']}")
            print(f"   소스: {item['source']}")
            print(f"   키워드: {item['keyword']}")
            if item['content_type'] == 'youtube_video':
                metadata = item['metadata']
                print(f"   조회수: {metadata.get('view_count', 0):,}")
                print(f"   채널: {metadata.get('channel_title', 'N/A')}")
                print(f"   길이: {metadata.get('duration', 'N/A')}")
                print(f"   URL: {metadata.get('video_url', 'N/A')}")
            print()
            
    except Exception as e:
        print(f"추천 요청 실패: {e}")
    
    # 2. YouTube 인기 동영상 추천
    print("\n2. YouTube 인기 동영상 추천")
    print("-" * 40)
    
    trending_data = {
        "category_id": "0",  # 전체 카테고리
        "max_results": 5
    }
    
    try:
        response = requests.post(f"{BASE_URL}/recommend/youtube/trending", json=trending_data)
        result = response.json()
        
        print(f"인기 동영상 수: {result['total_count']}")
        
        for i, item in enumerate(result["recommendations"], 1):
            metadata = item['metadata']
            print(f"{i}. {item['title']}")
            print(f"   채널: {metadata.get('channel_title', 'N/A')}")
            print(f"   조회수: {metadata.get('view_count', 0):,}")
            print(f"   URL: {metadata.get('video_url', 'N/A')}")
            print()
            
    except Exception as e:
        print(f"인기 동영상 조회 실패: {e}")
    
    # 3. YouTube 카테고리 조회
    print("\n3. YouTube 카테고리 목록")
    print("-" * 40)
    
    try:
        response = requests.get(f"{BASE_URL}/recommend/youtube/categories")
        result = response.json()
        
        print(f"총 카테고리 수: {result['total_count']}")
        print("\n카테고리 목록:")
        for category_id, category_name in list(result["categories"].items())[:10]:
            print(f"  {category_id}: {category_name}")
        print("  ...")
            
    except Exception as e:
        print(f"카테고리 조회 실패: {e}")
    
    # 4. YouTube 영상 수동 저장 (관리자 기능)
    print("\n4. YouTube 동영상 수동 저장 예시")
    print("-" * 40)
    print("※ 이 기능은 관리자가 좋은 동영상을 수동으로 추천 DB에 저장할 때 사용합니다")
    print("예시: /recommend/youtube/save?keyword=파이썬&video_id=dQw4w9WgXcQ")
    
def curl_examples():
    """curl 명령어 예시"""
    
    print("\n" + "=" * 60)
    print("curl 명령어 예시")
    print("=" * 60)
    
    examples = [
        {
            "title": "1. 기본 추천 요청",
            "command": '''curl -X POST "http://localhost:8000/recommend" \\
  -H "Content-Type: application/json" \\
  -d '{
    "keywords": ["파이썬", "AI"],
    "content_types": ["book", "movie", "video", "youtube_video"],
    "max_items": 10,
    "include_youtube": true,
    "youtube_max_per_keyword": 3
  }' '''
        },
        {
            "title": "2. YouTube 인기 동영상 조회",
            "command": '''curl -X POST "http://localhost:8000/recommend/youtube/trending" \\
  -H "Content-Type: application/json" \\
  -d '{
    "category_id": "0",
    "max_results": 5
  }' '''
        },
        {
            "title": "3. YouTube 카테고리 목록 조회",
            "command": '''curl -X GET "http://localhost:8000/recommend/youtube/categories"'''
        },
        {
            "title": "4. YouTube 동영상 수동 저장",
            "command": '''curl -X POST "http://localhost:8000/recommend/youtube/save?keyword=파이썬&video_id=dQw4w9WgXcQ"'''
        }
    ]
    
    for example in examples:
        print(f"\n{example['title']}")
        print("-" * 40)
        print(example['command'])

def setup_instructions():
    """설정 안내"""
    
    print("\n" + "=" * 60)
    print("YouTube API 설정 안내")
    print("=" * 60)
    
    instructions = """
1. Google Cloud Console 설정:
   - https://console.cloud.google.com 접속
   - 새 프로젝트 생성 또는 기존 프로젝트 선택
   - "API 및 서비스" > "라이브러리"에서 "YouTube Data API v3" 활성화
   - "API 및 서비스" > "사용자 인증 정보"에서 API 키 생성

2. 환경변수 설정:
   - .env 파일에 YOUTUBE_API_KEY=your_api_key_here 추가
   - API 키는 절대 공개하지 마세요

3. 의존성 설치:
   pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib httpx

4. 서버 실행:
   uvicorn main:app --reload

5. 테스트:
   python youtube_integration_example.py

주의사항:
- YouTube API 무료 할당량: 하루 10,000 단위
- 검색 1회당 100 단위 소모 (하루 약 100회 검색 가능)
- API 키에 HTTP 리퍼러 제한 설정 권장
"""
    
    print(instructions)

if __name__ == "__main__":
    print("YouTube 연동 콘텐츠 추천 시스템")
    print("서버가 실행 중인지 확인하세요: uvicorn main:app --reload")
    
    choice = input("\n실행할 기능을 선택하세요 (1: 테스트, 2: curl 예시, 3: 설정 안내): ")
    
    if choice == "1":
        # 서버 실행 확인
        try:
            response = requests.get(f"{BASE_URL}/docs")
            if response.status_code == 200:
                asyncio.run(test_youtube_recommendations())
            else:
                print("서버가 실행되지 않았거나 접근할 수 없습니다.")
        except:
            print("서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.")
    elif choice == "2":
        curl_examples()
    elif choice == "3":
        setup_instructions()
    else:
        print("잘못된 선택입니다.")
        curl_examples()
        setup_instructions() 