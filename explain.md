### 과제 설명

Agent System을 만들고 이 agent의 성능을 KMMLU: https://huggingface.co/datasets/HAERAE-HUB/KMMLU 중 Criminal-Law 카테고리의 test셋을 활용해 평가하는 코드를 작성합니다.

### 요구사항

- **제출 관련**
    - 전체 파일 크기는 2GB가 넘지 않게 작업해주세요.
        - 위 벤치마크 점수도 반드시 함께 제출해주세요.
    - 최종 평가에 활용된 batch api의 input, output jsonl 파일을 모두 제출해주세요.

- 코드 요구 사항
    - 모든 코드는 도커 환경 내에서 돌아갈 수 있어야 합니다.
    - 파이썬 Dependency manager를 사용해주세요.
    - Poetry, Pyproject를 사용해주세요.
    - Agent system의 구축부터 KMMLU 평가까지 한번에 실행할 수 있는 실행 스크립트를 작성해주세요.
    - 위 스크립트와 함께 컨테이너 셋업, 의존성 설치 등을 모두 포함한 docker compose 파일을 작성해주세요.
    - 위 내용들을 설명하는 README를 상세하게 작성해주세요.
        - README에 패키지 설치, 컨테이너 셋업 및 agent system 구축 및 평가를 실행할 수 있는 커맨드 라인 첨부 필수
    - RAG에 사용될 데이터의 raw data와 정제하는 과정도 모두 코드에 포함되어 있어야 합니다.
    - 구축과 평가를 모두 합쳐서 1시간 이내로 모두 돌아갈 수 있게 작성해주세요.
    (batch api 응답시간은 제외)
    - 최종 KMMLU 평가는 OpenAI batch api를 사용해주세요.

- 모델 제한 사항
    - LLM은 GPT-4o-mini, Embedding은 text-embedding-small 만 사용합니다.

### 과제 사례비

- 과제물을 제출해주신 분들께 감사의 마음을 담아 사례비가 지급됩니다.
    - LLM API를 활용하여 과제를 진행해주시기에 사용비 5만원, 과제비 5만원으로 총 10만원이 지급됩니다.
- 과제비는 과제를 완료한 경우에만 지급됩니다.