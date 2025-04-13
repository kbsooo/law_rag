#!/bin/bash

# 색상 정의
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}===== 형법 RAG + LLM 평가 시스템 실행 스크립트 =====${NC}"

# OpenAI API 키 확인
if [ -z "$OPENAI_API_KEY" ]; then
    if [ -f ".env" ]; then
        echo -e "${YELLOW}환경 변수에서 OpenAI API 키를 찾을 수 없습니다. .env 파일에서 로드합니다.${NC}"
        export $(grep -v '^#' .env | xargs)
    else
        echo -e "${RED}오류: OpenAI API 키가 설정되지 않았습니다.${NC}"
        echo "다음 방법 중 하나로 API 키를 설정하세요:"
        echo "1. export OPENAI_API_KEY=your-api-key"
        echo "2. .env 파일을 생성하고 OPENAI_API_KEY=your-api-key 추가"
        exit 1
    fi
fi

# 데이터셋 디렉토리 확인
if [ ! -d "./dataset" ]; then
    echo -e "${RED}오류: dataset 디렉토리를 찾을 수 없습니다.${NC}"
    echo "필요한 데이터 파일을 포함하는 dataset 디렉토리가 필요합니다."
    exit 1
fi

# results 디렉토리 생성 (없는 경우)
mkdir -p ./results

echo -e "${GREEN}1. Docker Compose로 시스템 구축 및 실행 중...${NC}"
docker-compose up --build

echo -e "${GREEN}실행 완료! 결과는 results 디렉토리에서 확인할 수 있습니다.${NC}"