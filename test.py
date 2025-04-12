# -*- coding: utf-8 -*-
import os
import json
import re
import time
import random
from datetime import datetime
from typing import List, Dict, Any, Optional

import pandas as pd
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from neo4j import GraphDatabase, Driver
from openai import OpenAI
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import display, HTML
import plotly.io as pio
from collections import Counter

# --- Configuration and Initialization ---

def load_config_and_initialize():
    """환경 변수 로드 및 주요 객체 초기화"""
    load_dotenv()

    config = {
        "neo4j_uri": os.getenv("NEO4J_URI"),
        "neo4j_username": os.getenv("NEO4J_USERNAME"),
        "neo4j_password": os.getenv("NEO4J_PASSWORD"),
        "openai_api_key": os.getenv("OPENAI_API_KEY"),
        "embedding_model_name": 'text-embedding-3-small',
        "embedding_dimension": 1536,
        "pdf_path": './dataset/criminal-law.pdf',
        "precedent_dir": './dataset/precedent_label/',
        "test_csv_path": './dataset/Criminal-Law-test.csv',
        "results_dir": "results",
        "llm_model": "gpt-4o-mini", # Batch API에서 사용할 모델
    }

    # 결과 디렉토리 생성
    os.makedirs(config["results_dir"], exist_ok=True)

    # Embedding 모델 설정
    embedding_model = OpenAIEmbeddings(
        model=config["embedding_model_name"],
        api_key=config["openai_api_key"]
    )

    # OpenAI 클라이언트 초기화
    openai_client = OpenAI(api_key=config["openai_api_key"])

    print("Configuration loaded and models initialized.")
    return config, embedding_model, openai_client

# --- Data Loading ---

def load_articles_from_pdf(pdf_path: str) -> Dict[str, str]:
    """PDF에서 법 조항 텍스트를 로드하고 추출"""
    print(f"Loading articles from PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    full_text = "\n".join(page.page_content for page in pages)

    # 조항 패턴 수정 (괄호 안 내용 포함, 공백 유연하게 처리)
    article_pattern = r'(제\s*\d+\s*조(?:의\s*\d+)?(?:\s*\(.*?\))?)'
    matches = list(re.finditer(article_pattern, full_text))

    articles = {}
    for i in range(len(matches)):
        current_match = matches[i]
        # 조항 ID 정규화 (공백 제거)
        current_article_id = re.sub(r'\s+', '', current_match.group(1)).strip()

        start_pos = current_match.start()
        end_pos = matches[i+1].start() if i < len(matches)-1 else len(full_text)
        article_text = full_text[start_pos:end_pos].strip()

        # 내용이 너무 짧으면 건너뛰기 (예: 목차 등에서 잘못 추출된 경우)
        if len(article_text) > 50: # 최소 길이 기준 설정
             articles[current_article_id] = article_text

    print(f"Processed {len(articles)} articles from PDF.")
    # 예시 출력 (선택적)
    # if articles:
    #     article_ids = list(articles.keys())
    #     print("\n--- First 5 Articles ---")
    #     for article_id in article_ids[:5]:
    #         print(f"\n--- Article: {article_id} ---")
    #         print(articles[article_id][:200] + "...")
    return articles

def load_precedents_from_json(precedent_dir: str, sample_size: Optional[int] = 1000) -> List[Dict[str, Any]]:
    """JSON 파일에서 판례 정보를 로드하고 정제"""
    print(f"Loading precedents from directory: {precedent_dir}")
    precedents = []
    rule_pattern = re.compile(r'제\s*\d+\s*조(?:의\s*\d+)?') # 참조 법조항 추출 패턴 (공백 유연)

    for filename in os.listdir(precedent_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(precedent_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    info = data.get("info", {})
                    summary_list = data.get("Summary", [])
                    keyword_list = data.get("keyword_tagg", [])
                    ref_info = data.get("Reference_info", {})

                    precedent_info = {
                        "case_id": info.get("caseNoID", filename.replace(".json", "")),
                        "case_name": info.get("caseNm"),
                        "judgment_summary": data.get("jdgmn", ""),
                        "full_summary": " ".join([s.get("summ_contxt", "") for s in summary_list]).strip(),
                        "keywords": [kw.get("keyword") for kw in keyword_list if kw.get("keyword")],
                        "referenced_rules_raw": ref_info.get("reference_rules", ""),
                        "referenced_cases_raw": ref_info.get("reference_court_case", ""),
                    }

                    # 참조 법조항 정제 (조항 번호만, 공백 제거)
                    raw_rules = precedent_info["referenced_rules_raw"].split(',') if precedent_info["referenced_rules_raw"] else []
                    cleaned_rules = set()
                    for rule in raw_rules:
                        matches = rule_pattern.findall(rule.strip())
                        for match in matches:
                            cleaned_rules.add(re.sub(r'\s+', '', match)) # 공백 제거 후 추가
                    precedent_info["referenced_rules"] = list(cleaned_rules)

                    # 참조 판례 정제 (간단히 공백 제거)
                    raw_cases = precedent_info["referenced_cases_raw"].split(',') if precedent_info["referenced_cases_raw"] else []
                    precedent_info["referenced_cases"] = [case.strip() for case in raw_cases if case.strip()]

                    # 임베딩할 텍스트 준비 (full_summary 우선, 없으면 judgment_summary)
                    precedent_info["text_for_embedding"] = precedent_info["full_summary"] or precedent_info["judgment_summary"]

                    # 유효한 데이터만 추가 (임베딩할 텍스트가 있어야 함)
                    if precedent_info["text_for_embedding"]:
                        precedents.append(precedent_info)

            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from {filename}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    print(f"Loaded {len(precedents)} valid precedents.")

    # 샘플링
    if sample_size is not None and len(precedents) > sample_size:
        print(f"Sampling {sample_size} precedents from {len(precedents)}...")
        random.seed(42)
        precedents = random.sample(precedents, sample_size)
        print(f"Selected {len(precedents)} precedents.")

    # 예시 출력 (선택적)
    # if precedents:
    #     print("\n--- Example Precedent ---")
    #     print(json.dumps(precedents[0], indent=2, ensure_ascii=False))

    return precedents

# --- Neo4j Interaction ---

def connect_neo4j(uri: str, auth: tuple) -> Optional[Driver]:
    """Neo4j 데이터베이스에 연결"""
    try:
        driver = GraphDatabase.driver(uri, auth=auth)
        driver.verify_connectivity()
        print("Successfully connected to Neo4j.")
        return driver
    except Exception as e:
        print(f"Failed to connect to Neo4j: {e}")
        return None

def close_neo4j(driver: Optional[Driver]):
    """Neo4j 드라이버 연결 종료"""
    if driver:
        driver.close()
        print("Neo4j driver connection closed.")

def setup_neo4j_constraints_and_indexes(driver: Driver, dimension: int):
    """Neo4j 제약조건 및 벡터 인덱스 설정"""
    print("Setting up Neo4j constraints and indexes...")
    with driver.session(database="neo4j") as session:
        # 제약조건
        session.run("CREATE CONSTRAINT article_id IF NOT EXISTS FOR (a:Article) REQUIRE a.id IS UNIQUE")
        session.run("CREATE CONSTRAINT precedent_id IF NOT EXISTS FOR (p:Precedent) REQUIRE p.id IS UNIQUE")
        session.run("CREATE CONSTRAINT keyword_text IF NOT EXISTS FOR (k:Keyword) REQUIRE k.text IS UNIQUE")
        print("Constraints created or already exist.")

        # 벡터 인덱스 (오류 발생 가능성 고려)
        try:
            session.run(
                "CREATE VECTOR INDEX article_embedding IF NOT EXISTS "
                "FOR (a:Article) ON (a.embedding) "
                f"OPTIONS {{indexConfig: {{`vector.dimensions`: {dimension}, `vector.similarity_function`: 'cosine'}}}}"
            )
            print("Article vector index created or already exists.")
        except Exception as e:
            print(f"Warning: Could not create Article vector index (may require Neo4j 5.11+): {e}")

        try:
            session.run(
                "CREATE VECTOR INDEX precedent_embedding IF NOT EXISTS "
                "FOR (p:Precedent) ON (p.embedding) "
                f"OPTIONS {{indexConfig: {{`vector.dimensions`: {dimension}, `vector.similarity_function`: 'cosine'}}}}"
            )
            print("Precedent vector index created or already exists.")
        except Exception as e:
            print(f"Warning: Could not create Precedent vector index (may require Neo4j 5.11+): {e}")

        # 인덱스 활성화 대기
        print("Waiting for indexes to populate (up to 300 seconds)...")
        try:
            session.run("CALL db.awaitIndexes(300)")
            print("Indexes should be online.")
        except Exception as e:
             print(f"Warning: Could not explicitly wait for indexes (may proceed anyway): {e}")


def create_graph_nodes_and_relationships(driver: Driver, articles: Dict[str, str], precedents: List[Dict[str, Any]], embed_model):
    """법 조항, 판례, 키워드 노드 및 관계를 배치 처리로 생성"""
    print("Creating graph nodes and relationships...")
    start_time = time.time()

    # 1. 법 조항 노드 생성 (Batch)
    print(f"Processing {len(articles)} articles...")
    articles_batch = []
    for article_id, content in articles.items():
        if content:
            try:
                embedding = embed_model.embed_query(content)
                articles_batch.append({
                    "id": article_id,
                    "text": content,
                    "embedding": embedding
                })
            except Exception as e:
                print(f"Error embedding article {article_id}: {e}")
        else:
            print(f"Skipping article {article_id} due to empty content.")

    if articles_batch:
        with driver.session(database="neo4j") as session:
            session.run(
                """
                UNWIND $batch as article_data
                MERGE (a:Article {id: article_data.id})
                SET a.text = article_data.text,
                    a.embedding = article_data.embedding
                """,
                batch=articles_batch
            )
        print(f"Created/updated {len(articles_batch)} Article nodes.")

    # 2. 판례, 키워드 노드 및 관계 생성 (Batch)
    print(f"Processing {len(precedents)} precedents...")
    precedents_batch = []
    relationships_batch = {"HAS_KEYWORD": [], "REFERENCES_ARTICLE": []}
    all_keywords = set()

    for precedent in precedents:
        text_to_embed = precedent.get("text_for_embedding")
        case_id = precedent.get("case_id")
        if text_to_embed and case_id:
            try:
                embedding = embed_model.embed_query(text_to_embed)
                precedents_batch.append({
                    "id": case_id,
                    "name": precedent.get("case_name"),
                    "judgment_summary": precedent.get("judgment_summary"),
                    "full_summary": precedent.get("full_summary"),
                    "embedding": embedding
                })

                # 키워드 관계 준비
                for keyword in precedent.get("keywords", []):
                    if keyword:
                        all_keywords.add(keyword)
                        relationships_batch["HAS_KEYWORD"].append({"case_id": case_id, "keyword_text": keyword})

                # 참조 법조항 관계 준비
                for article_ref in precedent.get("referenced_rules", []):
                    if article_ref:
                        relationships_batch["REFERENCES_ARTICLE"].append({"case_id": case_id, "article_ref": article_ref})

            except Exception as e:
                print(f"Error embedding or preparing precedent {case_id}: {e}")
        else:
             print(f"Skipping precedent {case_id} due to missing ID or text for embedding.")

    if precedents_batch:
        with driver.session(database="neo4j") as session:
            # 판례 노드 생성/업데이트
            session.run(
                """
                UNWIND $batch as p_data
                MERGE (p:Precedent {id: p_data.id})
                SET p.name = p_data.name,
                    p.judgment_summary = p_data.judgment_summary,
                    p.full_summary = p_data.full_summary,
                    p.embedding = p_data.embedding
                """,
                batch=precedents_batch
            )
            print(f"Created/updated {len(precedents_batch)} Precedent nodes.")

            # 키워드 노드 생성 (존재하지 않는 경우)
            if all_keywords:
                session.run(
                    """
                    UNWIND $keywords as keyword_text
                    MERGE (k:Keyword {text: keyword_text})
                    """,
                    keywords=list(all_keywords)
                )
                print(f"Ensured {len(all_keywords)} Keyword nodes exist.")

            # 관계 생성 (HAS_KEYWORD)
            if relationships_batch["HAS_KEYWORD"]:
                session.run(
                    """
                    UNWIND $rels as rel
                    MATCH (p:Precedent {id: rel.case_id})
                    MATCH (k:Keyword {text: rel.keyword_text})
                    MERGE (p)-[:HAS_KEYWORD]->(k)
                    """,
                    rels=relationships_batch["HAS_KEYWORD"]
                )
                print(f"Created/updated {len(relationships_batch['HAS_KEYWORD'])} HAS_KEYWORD relationships.")

            # 관계 생성 (REFERENCES_ARTICLE)
            if relationships_batch["REFERENCES_ARTICLE"]:
                session.run(
                    """
                    UNWIND $rels as rel
                    MATCH (p:Precedent {id: rel.case_id})
                    MATCH (a:Article) WHERE a.id STARTS WITH rel.article_ref // ID 시작 부분 일치
                    MERGE (p)-[:REFERENCES_ARTICLE]->(a)
                    """,
                    rels=relationships_batch["REFERENCES_ARTICLE"]
                )
                print(f"Created/updated {len(relationships_batch['REFERENCES_ARTICLE'])} REFERENCES_ARTICLE relationships.")

    end_time = time.time()
    print(f"Finished creating graph nodes and relationships in {end_time - start_time:.2f} seconds.")


# --- RAG and Context Processing ---

def retrieve_context_from_graph(driver: Driver, query_text: str, embed_model, top_k: int = 8) -> List[Dict[str, Any]]:
    """그래프 데이터베이스에서 관련 법 조항 및 판례 검색"""
    # print(f"\n--- Retrieving context for query: '{query_text}' ---")
    start_time = time.time()
    results = []

    try:
        query_embedding = embed_model.embed_query(query_text)
        query_keywords = extract_query_keywords(query_text) # 키워드 추출 함수 사용

        with driver.session(database="neo4j") as session:
            # 기본 벡터 검색 (Article + Precedent 동시 검색 및 점수 기반 정렬)
            cypher_query = """
            // 1. Article 벡터 검색
            CALL db.index.vector.queryNodes('article_embedding', $limit, $embedding) YIELD node AS article, score AS article_score
            WITH article, article_score
            // 관련 판례 수 계산 (선택적 성능 고려: 필요 없으면 제거 가능)
            OPTIONAL MATCH (p:Precedent)-[:REFERENCES_ARTICLE]->(article)
            WITH article, article_score, count(p) AS precedent_count
            // Article 결과 구성
            WITH {
                id: article.id,
                type: 'Article',
                text: article.text,
                score: article_score + (precedent_count * 0.01), // 판례 수 기반 보너스
                precedent_count: precedent_count
            } AS article_result

            UNION // 결과 합치기

            // 2. Precedent 벡터 검색
            CALL db.index.vector.queryNodes('precedent_embedding', $limit, $embedding) YIELD node AS precedent, score AS precedent_score
            WITH precedent, precedent_score
            // 관련 키워드 및 참조 조항 가져오기
            OPTIONAL MATCH (precedent)-[:HAS_KEYWORD]->(k:Keyword)
            OPTIONAL MATCH (precedent)-[:REFERENCES_ARTICLE]->(ref_a:Article)
            WITH precedent, precedent_score,
                 collect(DISTINCT k.text) AS keywords,
                 collect(DISTINCT ref_a.id) AS referenced_articles,
                 // 쿼리 키워드 매칭 보너스
                 sum(CASE WHEN size($query_keywords) > 0 AND any(kw IN $query_keywords WHERE k.text CONTAINS kw) THEN 0.05 ELSE 0 END) as keyword_bonus
            // Precedent 결과 구성
            WITH {
                id: precedent.id,
                type: 'Precedent',
                name: precedent.name,
                text: coalesce(precedent.full_summary, precedent.judgment_summary), // 요약 텍스트
                score: precedent_score + keyword_bonus,
                keywords: keywords,
                referenced_articles: referenced_articles
            } AS precedent_result

            // 3. 최종 결과 반환 (Article + Precedent)
            RETURN article_result AS result WHERE article_result IS NOT NULL
            UNION
            RETURN precedent_result AS result WHERE precedent_result IS NOT NULL

            // 4. 점수 기준으로 정렬 및 제한
            ORDER BY result.score DESC
            LIMIT $limit
            """

            search_results = session.run(
                cypher_query,
                embedding=query_embedding,
                query_keywords=query_keywords,
                limit=top_k * 2 # 충분한 후보군 확보 후 코드에서 최종 top_k 선택
            )

            processed_ids = set()
            for record in search_results:
                res = record["result"]
                if res and res.get("id") not in processed_ids:
                     # 텍스트 미리보기 생성
                    text_preview = res.get("text", "")
                    res["text_preview"] = text_preview[:300] + "..." if len(text_preview) > 300 else text_preview
                    results.append(res)
                    processed_ids.add(res["id"])
                    if len(results) >= top_k:
                        break # 최종 top_k 개수만큼만 선택

    except Exception as e:
        print(f"Error during graph retrieval for query '{query_text}': {e}")
        # 오류 발생 시 빈 리스트 반환 또는 다른 예외 처리

    end_time = time.time()
    # print(f"Context retrieval completed in {end_time - start_time:.2f} seconds. Found {len(results)} items.")
    return results


def extract_query_keywords(text: str) -> List[str]:
    """텍스트에서 간단한 키워드 추출 (불용어 제거 및 기본 정제)"""
    stopwords = [ # 간단한 불용어 목록
        "무엇", "어떤", "어떻게", "언제", "누구", "왜", "어디", "경우", "관하여", "대하여",
        "은", "는", "이", "가", "을", "를", "에", "의", "와", "과", "로", "으로",
        "있다", "없다", "때", "것", "등", "수", "그", "이", "저", "하는", "다음", "또는", "그리고"
    ]
    words = re.findall(r'\b\w{2,}\b', text) # 2글자 이상 단어 추출
    keywords = [w for w in words if w.lower() not in stopwords and not w.isdigit()]
    # 간단히 빈도수 상위 키워드 선택 (예: 상위 5개)
    counter = Counter(keywords)
    return [k for k, _ in counter.most_common(5)]


def process_text_for_context(text: str, keywords: List[str], max_len: int = 500, is_article: bool = False) -> str:
    """주어진 텍스트를 키워드 기반으로 요약/하이라이트 (간소화된 버전)"""
    if len(text) <= max_len:
        return text

    sentences = re.split(r'(?<=[.!?])\s+|(?<=\n)', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return text[:max_len] + "..."

    scored_sentences = []
    for i, sentence in enumerate(sentences):
        score = 0
        # 키워드 점수
        for keyword in keywords:
            if keyword in sentence:
                score += 2
        # 법 조항 언급 점수 (법 조항 텍스트인 경우)
        if is_article and re.search(r'제\s*\d+\s*조', sentence):
            score += 3
        # 문장 위치 점수 (시작/끝 강조)
        if i == 0: score += 2
        if i == len(sentences) - 1: score += 1

        scored_sentences.append((sentence, score, i))

    # 점수 높은 순 정렬 후 상위 문장 선택 (예: 상위 5개 또는 길이 제한까지)
    scored_sentences.sort(key=lambda x: x[1], reverse=True)

    selected_sentences = []
    current_len = 0
    # 최소 3문장, 최대 7문장 선택 시도
    min_sentences = 3
    max_sentences = 7

    # 점수 높은 순으로 추가하되, 원래 순서 고려 위해 인덱스 저장
    candidates = sorted(scored_sentences[:max_sentences], key=lambda x: x[2]) # 원래 순서대로 정렬

    for sentence, score, index in candidates:
         if current_len + len(sentence) < max_len or len(selected_sentences) < min_sentences :
              selected_sentences.append(sentence)
              current_len += len(sentence) + 1 # 공백 고려
         elif len(selected_sentences) >= min_sentences:
              break # 최소 문장 수 넘었고 길이 초과 시 중단

    # 첫 문장은 가능하면 포함 (제목 등)
    if sentences[0] not in selected_sentences and len(sentences[0]) < 100: # 너무 길지 않은 첫 문장
         selected_sentences.insert(0, sentences[0])

    result = " ".join(selected_sentences)
    return result[:max_len] + "..." if len(result) > max_len else result


def build_optimized_context(search_results: List[Dict[str, Any]], question: str) -> str:
    """검색 결과와 질문을 바탕으로 LLM에 제공할 최적화된 컨텍스트 구성"""
    if not search_results:
        return "관련 문맥 정보를 찾지 못했습니다."

    keywords = extract_query_keywords(question)
    context_parts = ["### 관련 참고 자료 ###"]

    # 법 조항 처리
    article_contexts = []
    for result in search_results:
        if result["type"] == "Article":
            processed_text = process_text_for_context(result["text"], keywords, max_len=400, is_article=True)
            context = f"【법조항: {result['id']} (Score: {result['score']:.2f})】\n{processed_text}"
            if result.get('precedent_count', 0) > 0:
                 context += f"\n[관련 판례 수: {result['precedent_count']}]"
            article_contexts.append(context)

    if article_contexts:
        context_parts.append("## 관련 법조항:")
        context_parts.extend(article_contexts[:3]) # 상위 3개 법조항 포함

    # 판례 처리
    precedent_contexts = []
    for result in search_results:
        if result["type"] == "Precedent":
            processed_text = process_text_for_context(result["text"], keywords, max_len=500, is_article=False)
            name_str = f" - {result.get('name', '')}" if result.get('name') else ""
            context = f"【판례: {result['id']}{name_str} (Score: {result['score']:.2f})】\n{processed_text}"
            if result.get("referenced_articles"):
                refs = ", ".join(result["referenced_articles"][:3])
                context += f"\n[참조 법조항: {refs}]"
            if result.get("keywords"):
                kws = ", ".join(result["keywords"][:5])
                context += f"\n[주요 키워드: {kws}]"
            precedent_contexts.append(context)

    if precedent_contexts:
        context_parts.append("## 관련 판례:")
        context_parts.extend(precedent_contexts[:2]) # 상위 2개 판례 포함

    context_parts.append("### 참고: 제공된 법조항과 판례를 종합적으로 고려하여 질문에 답하십시오. ###")

    return "\n\n".join(context_parts)

# --- Batch Processing and Evaluation ---

def prepare_batch_requests(df: pd.DataFrame, retrieved_contexts: Dict[int, List[Dict[str, Any]]], config: Dict) -> List[Dict[str, Any]]:
    """Batch API 요청 목록 생성"""
    batch_requests = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preparing batch requests"):
        question = row['question']
        options = { 'A': row['A'], 'B': row['B'], 'C': row['C'], 'D': row['D'] }
        contexts = retrieved_contexts.get(idx, [])

        # 최적화된 컨텍스트 구성
        context_str = build_optimized_context(contexts, question)

        # 프롬프트 작성
        prompt = f"""다음은 한국 형법에 관한 객관식 문제입니다. 제공된 문맥 정보를 참고하여 가장 적절한 답변을 선택하세요.

질문: {question}

선택지:
A. {options['A']}
B. {options['B']}
C. {options['C']}
D. {options['D']}

관련 문맥 정보:
{context_str}

답변 단계 (참고용):
1. 문제의 핵심 쟁점 파악 (구성요건/위법성/책임)
2. 관련 법조항 및 판례 적용 분석
3. 각 선택지의 법적 타당성 검토
4. 가장 정확한 선택지 결정

최종 답변은 반드시 A, B, C, D 중 하나만 명확하게 제시하세요. 예: "정답: A" 또는 "A"
"""

        request = {
            "custom_id": f"q_{idx}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": config["llm_model"],
                "messages": [
                    {"role": "system", "content": "당신은 한국 형법 전문가입니다. 주어진 질문과 문맥을 바탕으로 객관식 문제의 가장 적절한 답을 A, B, C, D 중 하나로 선택하여 명확하게 제시하세요."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 350, # 답변 및 약간의 설명을 포함할 수 있도록 조정
                "temperature": 0.1 # 일관성 있는 답변 유도
            }
        }
        batch_requests.append(request)
    return batch_requests

def run_batch_job(client: OpenAI, batch_requests: List[Dict[str, Any]], config: Dict, timestamp: str) -> Optional[str]:
    """Batch API 작업을 생성하고 완료될 때까지 모니터링"""
    # JSONL 파일로 저장
    batch_file_path = os.path.join(config["results_dir"], f"criminal_law_batch_input_{timestamp}.jsonl")
    try:
        with open(batch_file_path, 'w', encoding='utf-8') as f:
            for request in batch_requests:
                f.write(json.dumps(request, ensure_ascii=False) + '\n')
        print(f"Saved {len(batch_requests)} batch requests to {batch_file_path}")
    except IOError as e:
        print(f"Error saving batch input file: {e}")
        return None

    # 배치 파일 업로드
    try:
        with open(batch_file_path, "rb") as f:
            batch_input_file = client.files.create(file=f, purpose="batch")
        batch_input_file_id = batch_input_file.id
        print(f"Uploaded batch file with ID: {batch_input_file_id}")
    except Exception as e:
        print(f"Error uploading batch file: {e}")
        return None

    # 배치 작업 생성
    try:
        batch_job = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": f"Criminal Law benchmark evaluation {timestamp}"}
        )
        batch_id = batch_job.id
        print(f"Created batch job with ID: {batch_id}")
    except Exception as e:
        print(f"Error creating batch job: {e}")
        return None

    # 배치 작업 상태 확인 및 대기
    print("Waiting for batch job to complete...")
    start_time = time.time()
    status = None
    while True:
        try:
            status = client.batches.retrieve(batch_id)
            elapsed_time = time.time() - start_time
            print(f"Current status: {status.status} (Elapsed: {elapsed_time:.2f}s)")

            if status.status in ['completed', 'failed', 'cancelled', 'expired']:
                break

            # 폴링 간격 조절
            sleep_time = 30 if elapsed_time < 600 else 120
            time.sleep(sleep_time)
        except Exception as e:
            print(f"Error checking batch status: {e}. Retrying in 60s...")
            time.sleep(60)


    end_time = time.time()
    total_time = end_time - start_time
    print(f"Batch job finished with status: {status.status} in {total_time:.2f} seconds")

    if status.status == 'completed':
        return batch_id # 성공 시 배치 ID 반환
    else:
        print(f"Batch job failed or was cancelled. Final status: {status.status}")
        if hasattr(status, 'errors') and status.errors:
            print("Errors:")
            for error in status.errors.get('data', []):
                print(f"  - Code: {error.get('code')}, Message: {error.get('message')}, Line: {error.get('line')}")
        return None


def extract_choice_from_response(text: str) -> Optional[str]:
    """LLM 응답 텍스트에서 최종 선택지(A, B, C, D) 추출 (개선된 로직)"""
    if not text:
        return None

    # 1. 가장 명확한 패턴 우선 검색 (예: "정답: A", "최종 선택: B")
    # 대소문자 구분 없이, 공백 유연하게 처리
    clear_patterns = [
        r'(?:정답|답변|선택|결론)\s*:\s*([A-D])\b',
        r'\b([A-D])\s*(?:입니다|입니다\.|이다|이다\.)\s*$',
        r'최종(?:적)?\s*(?:선택|답변|정답)(?:은|은:)?\s*([A-D])\b',
        r'\b([A-D])(?:가|이)\s*가장\s*(?:적절|정확|타당)',
        r'따라서\s*(?:정답은|답은)?\s*([A-D])\b'
    ]
    for pattern in clear_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()

    # 2. 문장 시작 또는 끝에 단독으로 나오는 경우 (약간 덜 명확)
    # 예: "A", "B."
    edge_patterns = [
        r'^\s*([A-D])\b(?:\.|\s|$)', # 문장 시작
        r'\b([A-D])\b(?:\.|\s*)?$', # 문장 끝
    ]
    # 마지막 줄에서 먼저 확인
    lines = text.strip().split('\n')
    if lines:
        last_line = lines[-1].strip()
        for pattern in edge_patterns:
             match = re.match(pattern, last_line, re.IGNORECASE) # match 사용 (시작 부분 일치)
             if match: return match.group(1).upper()
             match = re.search(pattern, last_line, re.IGNORECASE) # search 사용 (끝 부분 일치)
             if match: return match.group(1).upper()


    # 3. 텍스트 전체에서 "A", "B", "C", "D" 언급 빈도 (최후의 수단, 정확도 낮을 수 있음)
    # "A가 정답", "B는 틀림" 등 긍정/부정 문맥 고려 시도
    counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    positive_indicators = ["정답", "맞는", "옳은", "적절", "타당", "선택"]
    negative_indicators = ["틀린", "아닌", "오답", "부적절", "잘못된"]

    sentences = re.split(r'[.!?]\s*', text)
    for sentence in sentences:
        for choice in counts.keys():
            if f" {choice} " in f" {sentence} " or sentence.startswith(f"{choice} ") or sentence.endswith(f" {choice}"):
                 # 긍정/부정 지표 확인
                 pos_score = sum(1 for indicator in positive_indicators if indicator in sentence)
                 neg_score = sum(1 for indicator in negative_indicators if indicator in sentence)
                 if pos_score > neg_score:
                     counts[choice] += 1
                 elif neg_score > pos_score:
                     counts[choice] -= 1
                 # 단순 언급도 약간의 가점
                 counts[choice] += 0.5

    # 가장 높은 점수를 가진 선택지 반환 (동점 시 None 반환 가능성 있음)
    if any(c > 0 for c in counts.values()):
        max_score = max(counts.values())
        best_choices = [choice for choice, score in counts.items() if score == max_score]
        if len(best_choices) == 1:
            return best_choices[0]

    # 모든 방법 실패 시 None 반환
    return None


def process_batch_results_and_evaluate(client: OpenAI, batch_id: str, df: pd.DataFrame, config: Dict, timestamp: str) -> Optional[str]:
    """배치 작업 결과를 다운로드, 처리하고 정확도 평가"""
    try:
        batch_job = client.batches.retrieve(batch_id)
        if batch_job.status != 'completed':
            print(f"Batch job {batch_id} did not complete successfully. Status: {batch_job.status}")
            return None

        output_file_id = batch_job.output_file_id
        error_file_id = batch_job.error_file_id
        print(f"Batch job completed. Output file ID: {output_file_id}, Error file ID: {error_file_id}")

        # 결과 파일 다운로드 및 처리
        output_file_path = os.path.join(config["results_dir"], f"criminal_law_batch_output_{timestamp}.jsonl")
        batch_results = []
        try:
            file_response = client.files.content(output_file_id)
            with open(output_file_path, 'w', encoding='utf-8') as f:
                for line in file_response.text.strip().split('\n'):
                    if line.strip():
                        f.write(line + '\n') # 원본 저장
                        batch_results.append(json.loads(line))
            print(f"Downloaded and saved {len(batch_results)} results to {output_file_path}")
        except Exception as e:
            print(f"Error downloading or saving output file {output_file_id}: {e}")
            return None # 결과 파일 없으면 평가 불가

        # 오류 파일 처리 (선택적)
        if error_file_id:
            try:
                error_response = client.files.content(error_file_id)
                error_file_path = os.path.join(config["results_dir"], f"criminal_law_batch_errors_{timestamp}.jsonl")
                with open(error_file_path, 'w', encoding='utf-8') as f:
                    f.write(error_response.text)
                print(f"Downloaded and saved error file to {error_file_path}")
            except Exception as e:
                print(f"Warning: Could not download or save error file {error_file_id}: {e}")


        # 정확도 평가
        correct_count = 0
        processed_count = 0
        results_data = []

        for result in batch_results:
            custom_id = result.get('custom_id')
            response = result.get('response')
            error = result.get('error')

            if not custom_id or (not response and not error):
                print(f"Skipping invalid result entry: {result}")
                continue

            try:
                idx = int(custom_id.split('_')[1])
                processed_count += 1
                predicted_answer = None
                response_text = ""
                is_correct = False
                actual_answer = chr(64 + df.iloc[idx]['answer']) # 1->A, 2->B, ...

                if error:
                    print(f"Error in result for question {idx}: {error}")
                    response_text = f"Error: {error.get('message', 'Unknown error')}"
                elif response and response.get('body') and response['body'].get('choices'):
                    response_text = response['body']['choices'][0]['message']['content'].strip()
                    predicted_answer = extract_choice_from_response(response_text)
                    if predicted_answer:
                        is_correct = (predicted_answer == actual_answer)
                        if is_correct:
                            correct_count += 1
                    else:
                        print(f"Warning: Could not extract answer for question {idx}. Response: {response_text[:100]}...")
                else:
                     print(f"Warning: Empty response body for question {idx}")
                     response_text = "Empty response"


                results_data.append({
                    'question_id': idx,
                    'question': df.iloc[idx]['question'],
                    'predicted': predicted_answer,
                    'actual': actual_answer,
                    'is_correct': is_correct,
                    'response': response_text
                })
            except Exception as e:
                print(f"Error processing result for custom_id {custom_id}: {e}")

        if processed_count == 0:
            print("No results were processed successfully.")
            return None

        accuracy = correct_count / processed_count
        print(f"\n--- Evaluation Summary ---")
        print(f"Total results processed: {processed_count}")
        print(f"Correct answers: {correct_count}")
        print(f"Accuracy: {accuracy:.4f}")

        # 상세 결과 저장
        results_df = pd.DataFrame(results_data)
        results_file_path = os.path.join(config["results_dir"], f"criminal_law_evaluation_results_{timestamp}.csv")
        results_df.to_csv(results_file_path, index=False, encoding='utf-8-sig')
        print(f"Saved detailed evaluation results to {results_file_path}")

        return results_file_path # 평가 결과 파일 경로 반환

    except Exception as e:
        print(f"An error occurred during batch result processing and evaluation: {e}")
        return None

# --- Result Analysis and Visualization ---

def analyze_and_visualize_results(results_file_path: str, config: Dict, timestamp: str):
    """평가 결과를 분석하고 시각화"""
    print(f"\n--- Analyzing results from: {results_file_path} ---")
    try:
        results_df = pd.read_csv(results_file_path)
    except Exception as e:
        print(f"Error reading results file: {e}")
        return

    if results_df.empty:
        print("Results data is empty. Skipping analysis.")
        return

    # 기본 테마 설정
    pio.templates.default = "plotly_white"

    # 요약 통계
    total_questions = len(results_df)
    # 'is_correct'가 boolean이 아닐 수 있으므로 안전하게 처리
    correct_answers = results_df['is_correct'].astype(bool).sum()
    accuracy = correct_answers / total_questions if total_questions > 0 else 0

    print("\n===== Benchmark Result Summary =====")
    print(f"Total Questions Processed: {total_questions}")
    print(f"Correct Answers: {correct_answers}")
    print(f"Accuracy: {accuracy:.4f}")

    # 예측/실제 분포
    options = ['A', 'B', 'C', 'D'] # 가능한 모든 옵션
    prediction_counts = results_df['predicted'].value_counts().reindex(options, fill_value=0)
    actual_counts = results_df['actual'].value_counts().reindex(options, fill_value=0)

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Bar(x=options, y=prediction_counts.values, name='Model Prediction', marker_color='rgb(55, 83, 109)'))
    fig_dist.add_trace(go.Bar(x=options, y=actual_counts.values, name='Actual Answer', marker_color='rgb(26, 118, 255)'))
    fig_dist.update_layout(title='Choice Distribution (Prediction vs Actual)', barmode='group', font=dict(size=12))
    fig_dist.show()

    # 혼동 행렬
    # NaN 값 처리 및 문자열로 변환 후 crosstab 계산
    results_df['predicted_fill'] = results_df['predicted'].fillna('None').astype(str)
    results_df['actual_fill'] = results_df['actual'].fillna('None').astype(str)
    all_labels = sorted(list(set(results_df['predicted_fill']) | set(results_df['actual_fill'])))

    conf_matrix = pd.crosstab(results_df['actual_fill'], results_df['predicted_fill'], rownames=['Actual'], colnames=['Predicted'])
    # 모든 라벨 포함하도록 reindex
    conf_matrix = conf_matrix.reindex(index=all_labels, columns=all_labels, fill_value=0)

    # 비율 계산 (0으로 나누는 경우 방지)
    conf_matrix_norm = conf_matrix.astype('float').div(conf_matrix.sum(axis=1).replace(0, 1), axis=0) # 행 기준 정규화

    fig_cm = px.imshow(conf_matrix_norm, text_auto='.2f', aspect="auto",
                       labels=dict(x="Predicted", y="Actual", color="Proportion"),
                       x=all_labels, y=all_labels, color_continuous_scale="Blues")
    fig_cm.update_layout(title='Confusion Matrix (Normalized by Actual)', font=dict(size=12))
    fig_cm.show()


    # 정답률 파이 차트
    labels = ['Correct', 'Incorrect']
    values = [correct_answers, total_questions - correct_answers]
    colors = ['rgb(46, 204, 113)', 'rgb(231, 76, 60)']
    fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4, marker_colors=colors)])
    fig_pie.update_layout(title='Overall Accuracy', font=dict(size=12))
    fig_pie.show()

    # 오답 예시 출력
    incorrect_examples = results_df[results_df['is_correct'] == False].head(5)
    print("\n===== Incorrect Answer Examples (Top 5) =====")
    for _, row in incorrect_examples.iterrows():
        print(f"Q_ID: {row['question_id']}, Predicted: {row['predicted']}, Actual: {row['actual']}")
        print(f"Question: {row['question'][:100]}...")
        # print(f"Response: {row['response'][:100]}...") # 필요시 응답 출력
        print("-" * 20)

    # 요약 정보 HTML 표시
    summary_html = f"""
    <div style="background-color: #f0f0f0; padding: 15px; border-radius: 5px; border-left: 5px solid #007bff;">
        <h4 style="margin-top: 0;">Benchmark Summary ({timestamp})</h4>
        <p><b>Accuracy:</b> {accuracy:.2%} ({correct_answers}/{total_questions})</p>
        <p><b>Model:</b> {config.get('llm_model', 'N/A')}</p>
        <p><b>RAG Type:</b> Graph-based (Neo4j)</p>
        <p><b>Results File:</b> {os.path.basename(results_file_path)}</p>
    </div>
    """
    display(HTML(summary_html))

    # (선택적) 카테고리별 분석 등 추가 분석 가능


# --- Main Execution ---

if __name__ == "__main__":
    # 1. 설정 로드 및 초기화
    config, embedding_model, openai_client = load_config_and_initialize()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 2. 데이터 로드
    articles = load_articles_from_pdf(config["pdf_path"])
    precedents = load_precedents_from_json(config["precedent_dir"], sample_size=1000) # 필요시 샘플 사이즈 조절

    # 3. Neo4j 연결 및 설정
    neo4j_driver = connect_neo4j(config["neo4j_uri"], (config["neo4j_username"], config["neo4j_password"]))

    if neo4j_driver:
        try:
            # 4. 그래프 생성 (제약조건/인덱스 설정 포함)
            setup_neo4j_constraints_and_indexes(neo4j_driver, config["embedding_dimension"])
            # 노드 및 관계 생성 (배치 처리) - 이미 생성된 경우 건너뛸 수 있도록 로직 추가 가능
            create_graph_nodes_and_relationships(neo4j_driver, articles, precedents, embedding_model)

            # 5. 평가 데이터 로드
            try:
                eval_df = pd.read_csv(config["test_csv_path"])
                print(f"\nLoaded {len(eval_df)} questions for evaluation from {config['test_csv_path']}")
            except FileNotFoundError:
                print(f"Error: Evaluation CSV file not found at {config['test_csv_path']}")
                eval_df = pd.DataFrame() # 빈 데이터프레임으로 계속 진행 방지

            if not eval_df.empty:
                # 6. 모든 질문에 대해 RAG 검색 실행
                print("\nPerforming RAG search for all evaluation questions...")
                retrieved_contexts = {}
                for idx, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Retrieving contexts"):
                    question = row['question']
                    try:
                        contexts = retrieve_context_from_graph(neo4j_driver, question, embedding_model, top_k=8)
                        retrieved_contexts[idx] = contexts
                    except Exception as e:
                        print(f"Error during RAG search for question index {idx}: {e}")
                        retrieved_contexts[idx] = [] # 오류 발생 시 빈 컨텍스트
                print(f"Completed RAG search for {len(retrieved_contexts)} questions.")

                # 7. Batch API 요청 준비
                batch_requests = prepare_batch_requests(eval_df, retrieved_contexts, config)

                # 8. Batch API 작업 실행 및 모니터링
                if batch_requests:
                    batch_id = run_batch_job(openai_client, batch_requests, config, timestamp)

                    # 9. 결과 처리 및 평가
                    if batch_id:
                        results_file = process_batch_results_and_evaluate(openai_client, batch_id, eval_df, config, timestamp)

                        # 10. 결과 분석 및 시각화 (선택적)
                        if results_file:
                             analyze_and_visualize_results(results_file, config, timestamp)
                    else:
                        print("Batch job failed to start or complete. Skipping evaluation.")
                else:
                    print("No batch requests were prepared. Skipping batch job execution.")
            else:
                print("Evaluation data is empty. Skipping RAG search and batch processing.")

        finally:
            # 11. Neo4j 연결 종료
            close_neo4j(neo4j_driver)
    else:
        print("Could not connect to Neo4j. Aborting.")

    print("\nScript execution finished.")