#!/usr/bin/env python3
# Criminal Law RAG + LLM Evaluation System

import os
import json
import re
import time
import random
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import Counter

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from neo4j import GraphDatabase, Driver
from openai import OpenAI
from tqdm import tqdm

def load_config_and_initialize():
    """환경 변수 로드 및 주요 객체 초기화"""
    # .env 파일 로드
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
        "llm_model": "gpt-4o-mini",
    }

    # 결과 디렉토리 생성
    os.makedirs(config["results_dir"], exist_ok=True)

    # OpenAI API 키 확인
    if not config["openai_api_key"]:
        print("Warning: OpenAI API Key not set. Please set the OPENAI_API_KEY environment variable.")
        return None, None, None

    print("OpenAI API Key loaded.")

    # Embedding 모델 설정
    try:
        embedding_model = OpenAIEmbeddings(
            model=config["embedding_model_name"],
            api_key=config["openai_api_key"]
        )
        print(f"Embedding model '{config['embedding_model_name']}' initialized.")
    except Exception as e:
        print(f"Error initializing embedding model: {e}")
        embedding_model = None

    # OpenAI 클라이언트 초기화
    try:
        openai_client = OpenAI(api_key=config["openai_api_key"])
        print("OpenAI client initialized.")
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        openai_client = None

    print("Configuration loaded and objects initialized.")
    return config, embedding_model, openai_client

def load_articles_from_pdf(pdf_path: str) -> Dict[str, str]:
    """PDF에서 법 조항 텍스트를 로드하고 추출"""
    print(f"Loading articles from PDF: {pdf_path}")
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return {}

    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        full_text = "\n".join(page.page_content for page in pages)
        print(f"Loaded {len(pages)} pages from PDF.")

        # 조항 패턴 수정 (괄호 안 내용 포함, 공백 유연하게 처리)
        article_pattern = r'(제\s*\d+\s*조(?:의\s*\d+)?(?:\s*\(.*?\))?)'
        matches = list(re.finditer(article_pattern, full_text))
        print(f"Found {len(matches)} potential article markers.")

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
        # 예시 출력 (3개만)
        if articles:
            article_ids = list(articles.keys())
            print("\n--- First 3 Articles (Preview) ---")
            for article_id in article_ids[:3]:
                print(f"\n--- Article: {article_id} ---")
                print(articles[article_id][:150] + "...")
        return articles

    except Exception as e:
        print(f"An error occurred while loading/processing the PDF: {e}")
        return {}

def load_precedents_from_json(precedent_dir: str, sample_size: Optional[int] = 1000) -> List[Dict[str, Any]]:
    """JSON 파일에서 판례 정보를 로드하고 정제"""
    print(f"Loading precedents from directory: {precedent_dir}")
    if not os.path.isdir(precedent_dir):
        print(f"Error: Precedent directory not found at {precedent_dir}")
        return []

    precedents = []
    rule_pattern = re.compile(r'제\s*\d+\s*조(?:의\s*\d+)?') # 참조 법조항 추출 패턴 (공백 유연)
    files_processed = 0
    files_skipped = 0

    json_files = [f for f in os.listdir(precedent_dir) if f.endswith(".json")]
    print(f"Found {len(json_files)} JSON files.")

    for filename in tqdm(json_files, desc="Loading precedents"):
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
                    files_processed += 1
                else:
                    files_skipped += 1

        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {filename}")
            files_skipped += 1
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            files_skipped += 1

    print(f"Loaded {len(precedents)} valid precedents. Skipped {files_skipped} files.")

    # 샘플링
    if sample_size is not None and len(precedents) > sample_size:
        print(f"Sampling {sample_size} precedents from {len(precedents)}...")
        random.seed(42) # 재현성을 위한 시드 고정
        precedents = random.sample(precedents, sample_size)
        print(f"Selected {len(precedents)} precedents after sampling.")

    return precedents

def connect_neo4j(uri: str, auth: tuple) -> Optional[Driver]:
    """Neo4j 데이터베이스에 연결"""
    print(f"Attempting to connect to Neo4j at {uri}...")
    try:
        driver = GraphDatabase.driver(uri, auth=auth)
        driver.verify_connectivity()
        print("Successfully connected to Neo4j.")
        return driver
    except Exception as e:
        print(f"Failed to connect to Neo4j: {e}")
        print("Please ensure Neo4j is running and the connection details are correct.")
        return None

def close_neo4j(driver: Optional[Driver]):
    """Neo4j 드라이버 연결 종료"""
    if driver:
        driver.close()
        print("Neo4j driver connection closed.")

def setup_neo4j_constraints_and_indexes(driver: Driver, dimension: int):
    """Neo4j 제약조건 및 벡터 인덱스 설정"""
    print("Setting up Neo4j constraints and indexes...")
    try:
        with driver.session(database="neo4j") as session:
            # 제약조건 (Idempotent: 이미 존재하면 오류 없이 넘어감)
            session.run("CREATE CONSTRAINT article_id IF NOT EXISTS FOR (a:Article) REQUIRE a.id IS UNIQUE")
            session.run("CREATE CONSTRAINT precedent_id IF NOT EXISTS FOR (p:Precedent) REQUIRE p.id IS UNIQUE")
            session.run("CREATE CONSTRAINT keyword_text IF NOT EXISTS FOR (k:Keyword) REQUIRE k.text IS UNIQUE")
            print("Constraints created or verified.")

            # 벡터 인덱스 (Idempotent: 이미 존재하면 오류 없이 넘어감)
            # Neo4j 버전 5.11 이상 필요
            index_commands = [
                (f"CREATE VECTOR INDEX article_embedding IF NOT EXISTS "
                 f"FOR (a:Article) ON (a.embedding) "
                 f"OPTIONS {{indexConfig: {{`vector.dimensions`: {dimension}, `vector.similarity_function`: 'cosine'}}}}"),
                (f"CREATE VECTOR INDEX precedent_embedding IF NOT EXISTS "
                 f"FOR (p:Precedent) ON (p.embedding) "
                 f"OPTIONS {{indexConfig: {{`vector.dimensions`: {dimension}, `vector.similarity_function`: 'cosine'}}}}")
            ]
            for command in index_commands:
                try:
                    session.run(command)
                    index_name = command.split(" ")[3] # 간단히 인덱스 이름 추출
                    print(f"Vector index '{index_name}' created or verified.")
                except Exception as e:
                    # 벡터 인덱스 생성 실패는 경고로 처리 (하위 버전 호환성)
                    print(f"Warning: Could not create or verify vector index (requires Neo4j 5.11+ with Vector Search plugin): {e}")
                    print("Vector search functionality might not be available.")

            # 인덱스 활성화 대기 (Best-effort)
            print("Attempting to wait for indexes to populate (up to 60 seconds)...")
            try:
                # awaitIndexes는 오래 걸릴 수 있으므로 타임아웃 설정
                session.run("CALL db.awaitIndexes(60000)") # 60초 타임아웃 (밀리초 단위)
                print("Indexes should be online.")
            except Exception as e:
                 print(f"Warning: Could not explicitly wait for indexes: {e}")

    except Exception as e:
        print(f"An error occurred during Neo4j setup: {e}")

def create_graph_nodes_and_relationships(driver: Driver, articles: Dict[str, str], precedents: List[Dict[str, Any]], embed_model):
    """법 조항, 판례, 키워드 노드 및 관계를 배치 처리로 생성"""
    if not driver:
        print("Neo4j driver is not available. Skipping graph creation.")
        return
    if not embed_model:
        print("Embedding model is not available. Skipping graph creation.")
        return
    if not articles and not precedents:
        print("No articles or precedents data to process. Skipping graph creation.")
        return

    print("Starting graph data creation process...")
    start_time = time.time()

    # --- 1. 법 조항 노드 생성 (Batch) ---
    print(f"Processing {len(articles)} articles...")
    articles_batch = []
    articles_embedded = 0
    articles_skipped = 0
    for article_id, content in tqdm(articles.items(), desc="Embedding Articles"):
        if content:
            try:
                # 임베딩 모델 재시도 로직 추가 (선택적)
                embedding = embed_model.embed_query(content)
                articles_batch.append({
                    "id": article_id,
                    "text": content,
                    "embedding": embedding
                })
                articles_embedded += 1
            except Exception as e:
                print(f"Error embedding article {article_id}: {e}. Skipping.")
                articles_skipped += 1
        else:
            articles_skipped += 1

    if articles_batch:
        print(f"Embedding complete for {articles_embedded} articles (skipped {articles_skipped}). Writing to Neo4j...")
        try:
            with driver.session(database="neo4j") as session:
                # UNWIND + MERGE/SET 사용 (효율적)
                session.run(
                    """
                    UNWIND $batch as article_data
                    MERGE (a:Article {id: article_data.id})
                    SET a.text = article_data.text,
                        a.embedding = article_data.embedding,
                        a.last_updated = timestamp() // 마지막 업데이트 시간 기록 (선택적)
                    """,
                    batch=articles_batch
                )
            print(f"Successfully created/updated {len(articles_batch)} Article nodes.")
        except Exception as e:
            print(f"Error writing Article nodes to Neo4j: {e}")
    else:
        print("No valid articles to write to Neo4j.")


    # --- 2. 판례, 키워드 노드 및 관계 생성 (Batch) ---
    print(f"\nProcessing {len(precedents)} precedents...")
    precedents_batch = []
    relationships_batch = {"HAS_KEYWORD": [], "REFERENCES_ARTICLE": []}
    all_keywords = set()
    precedents_embedded = 0
    precedents_skipped = 0

    for precedent in tqdm(precedents, desc="Embedding Precedents"):
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
                precedents_embedded += 1

                # 키워드 관계 준비
                for keyword in precedent.get("keywords", []):
                    if keyword:
                        all_keywords.add(keyword)
                        relationships_batch["HAS_KEYWORD"].append({"case_id": case_id, "keyword_text": keyword})

                # 참조 법조항 관계 준비 (정제된 ID 사용)
                for article_ref in precedent.get("referenced_rules", []):
                    if article_ref:
                        # 여기서 article_ref는 '제100조'와 같이 정제된 형태여야 함
                        relationships_batch["REFERENCES_ARTICLE"].append({"case_id": case_id, "article_ref": article_ref})

            except Exception as e:
                print(f"Error embedding or preparing precedent {case_id}: {e}. Skipping.")
                precedents_skipped += 1
        else:
             precedents_skipped += 1

    if precedents_batch:
        print(f"Embedding complete for {precedents_embedded} precedents (skipped {precedents_skipped}). Writing to Neo4j...")
        try:
            with driver.session(database="neo4j") as session:
                # --- 판례 노드 생성/업데이트 ---
                session.run(
                    """
                    UNWIND $batch as p_data
                    MERGE (p:Precedent {id: p_data.id})
                    SET p.name = p_data.name,
                        p.judgment_summary = p_data.judgment_summary,
                        p.full_summary = p_data.full_summary,
                        p.embedding = p_data.embedding,
                        p.last_updated = timestamp()
                    """,
                    batch=precedents_batch
                )
                print(f"Successfully created/updated {len(precedents_batch)} Precedent nodes.")

                # --- 키워드 노드 생성 (존재하지 않는 경우) ---
                if all_keywords:
                    # 키워드를 배치로 처리
                    keyword_list = list(all_keywords)
                    kw_batch_size = 1000 # 배치 크기 조절 가능
                    for i in range(0, len(keyword_list), kw_batch_size):
                         batch = keyword_list[i:i+kw_batch_size]
                         session.run(
                             """
                             UNWIND $keywords as keyword_text
                             MERGE (k:Keyword {text: keyword_text})
                             """,
                             keywords=batch
                         )
                    print(f"Ensured {len(all_keywords)} Keyword nodes exist.")

                # --- 관계 생성 (HAS_KEYWORD) ---
                if relationships_batch["HAS_KEYWORD"]:
                    rel_batch_size = 5000 # 배치 크기 조절 가능
                    rels = relationships_batch["HAS_KEYWORD"]
                    for i in range(0, len(rels), rel_batch_size):
                        batch = rels[i:i+rel_batch_size]
                        session.run(
                            """
                            UNWIND $rels as rel
                            MATCH (p:Precedent {id: rel.case_id})
                            MATCH (k:Keyword {text: rel.keyword_text})
                            MERGE (p)-[r:HAS_KEYWORD]->(k)
                            SET r.last_updated = timestamp() // 관계 업데이트 시간 (선택적)
                            """,
                            rels=batch
                        )
                    print(f"Created/updated {len(rels)} HAS_KEYWORD relationships.")

                # --- 관계 생성 (REFERENCES_ARTICLE) ---
                if relationships_batch["REFERENCES_ARTICLE"]:
                    rel_batch_size = 5000 # 배치 크기 조절 가능
                    rels = relationships_batch["REFERENCES_ARTICLE"]
                    for i in range(0, len(rels), rel_batch_size):
                        batch = rels[i:i+rel_batch_size]
                        session.run(
                            """
                            UNWIND $rels as rel
                            MATCH (p:Precedent {id: rel.case_id})
                            // 법 조항 ID가 정확히 일치하거나, '제100조의2' 같은 경우 '제100조'로 시작하는 것을 찾음
                            MATCH (a:Article) WHERE a.id = rel.article_ref OR a.id STARTS WITH rel.article_ref
                            MERGE (p)-[r:REFERENCES_ARTICLE]->(a)
                            SET r.last_updated = timestamp()
                            """,
                            rels=batch
                        )
                    print(f"Created/updated {len(rels)} REFERENCES_ARTICLE relationships.")

        except Exception as e:
            print(f"Error writing Precedent nodes or relationships to Neo4j: {e}")

    else:
        print("No valid precedents to write to Neo4j.")

    end_time = time.time()
    print(f"Finished graph data creation process in {end_time - start_time:.2f} seconds.")

def extract_query_keywords(text: str, max_keywords: int = 5) -> List[str]:
    """텍스트에서 간단한 키워드 추출 (개선된 버전: 불용어, 길이, 빈도 고려)"""
    # 간단한 한국어 불용어 목록 (필요시 확장)
    stopwords = set([
        "의", "가", "이", "은", "들", "는", "좀", "잘", "걍", "과", "도", "를", "으로", "자", "에", "와", "한", "하다",
        "것", "그", "저", "수", "때", "등", "및", "제", "조", "항", "관련", "대한", "대해", "위한", "있는", "하는",
        "그리고", "그러나", "그래서", "하지만", "또는", "다른", "모든", "어떤", "누구", "무엇", "언제", "어디서", "어떻게", "왜",
        "입니다", "습니다", "합니다", "에서", "에게", "부터", "까지", "보다", "만", "같이", "처럼", "따라", "통해",
        "경우", "문제", "질문", "답변", "선택지", "다음", "중", "가장", "적절한", "옳은", "틀린"
    ])
    # 명사형 단어 위주 추출 (정규식 개선)
    words = re.findall(r'\b[가-힣]{2,}\b', text) # 2글자 이상 한글 단어
    keywords = [w for w in words if w not in stopwords and not w.isdigit()]

    # 빈도수 계산
    counter = Counter(keywords)

    # 빈도수 상위 키워드 선택
    common_keywords = [k for k, _ in counter.most_common(max_keywords)]
    return common_keywords

def retrieve_context_from_graph(driver: Driver, query_text: str, embed_model, top_k: int = 12) -> List[Dict[str, Any]]:
    """그래프 데이터베이스에서 관련 법 조항 및 판례 검색 (스코어링 단순화)"""
    if not driver:
        print("Neo4j driver is not available. Cannot retrieve context.")
        return []
    if not embed_model:
        print("Embedding model is not available. Cannot retrieve context.")
        return []

    results = []

    try:
        query_embedding = embed_model.embed_query(query_text)
        # 키워드 추출은 유지 (향후 다른 용도로 사용 가능)
        query_keywords = extract_query_keywords(query_text, max_keywords=5)

        with driver.session(database="neo4j") as session:
            # 1. Article 검색 쿼리 - 스코어링 단순화
            article_query = """
            CALL db.index.vector.queryNodes('article_embedding', $limit, $embedding)
            YIELD node AS article, score AS article_score
            // 참조 판례 수는 정보 제공용으로 유지
            OPTIONAL MATCH (p:Precedent)-[:REFERENCES_ARTICLE]->(article)
            WITH article, article_score, count(p) AS precedent_count
            RETURN
                article.id AS id,
                'Article' AS type,
                article.text AS text,
                article_score AS score, // 벡터 점수만 사용
                precedent_count
            ORDER BY score DESC
            LIMIT $limit
            """

            article_results = session.run(
                article_query,
                embedding=query_embedding,
                limit=top_k # top_k 값 실험 가능
            )

            # 2. Precedent 검색 쿼리 - 스코어링 단순화
            precedent_query = """
            CALL db.index.vector.queryNodes('precedent_embedding', $limit, $embedding)
            YIELD node AS precedent, score AS precedent_score
            // 키워드 및 참조 조항은 정보 제공용으로 유지
            OPTIONAL MATCH (precedent)-[:HAS_KEYWORD]->(k:Keyword)
            WITH precedent, precedent_score, collect(DISTINCT k.text) AS keywords
            OPTIONAL MATCH (precedent)-[:REFERENCES_ARTICLE]->(ref_a:Article)
            WITH
                precedent,
                precedent_score,
                keywords,
                collect(DISTINCT ref_a.id) AS referenced_articles
            // 키워드 보너스 제거
            RETURN
                precedent.id AS id,
                'Precedent' AS type,
                precedent.name AS name,
                coalesce(precedent.full_summary, precedent.judgment_summary) AS text,
                precedent_score AS score, // 벡터 점수만 사용
                keywords,
                referenced_articles
            ORDER BY score DESC
            LIMIT $limit
            """

            precedent_results = session.run(
                precedent_query,
                embedding=query_embedding,
                limit=top_k # top_k 값 실험 가능
            )

            # 두 결과를 합치고 점수 기반으로 정렬
            all_results = []

            # Article 결과 처리
            for record in article_results:
                result = {
                    "id": record["id"],
                    "type": record["type"],
                    "text": record["text"],
                    "score": record["score"],
                    "precedent_count": record["precedent_count"]
                }
                all_results.append(result)

            # Precedent 결과 처리
            for record in precedent_results:
                result = {
                    "id": record["id"],
                    "type": record["type"],
                    "name": record["name"],
                    "text": record["text"],
                    "score": record["score"],
                    "keywords": record["keywords"],
                    "referenced_articles": record["referenced_articles"]
                }
                all_results.append(result)

            # 점수로 정렬 및 상위 결과 선택
            all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
            results = all_results[:top_k] # 최종 top_k 개수 선택

            # 텍스트 미리보기 추가 (기존 로직 유지)
            for res in results:
                text_preview = res.get("text", "")
                res["text_preview"] = text_preview[:250] + "..." if len(text_preview) > 250 else text_preview

    except Exception as e:
        if "IndexNotFoundException" in str(e):
            print(f"Warning: Vector index not found. Falling back to basic retrieval (if implemented) or skipping. Error: {e}")
        else:
            print(f"Error during graph retrieval for query '{query_text[:50]}...': {e}")

    return results

def process_text_for_context(text: str, max_len: int = 600) -> str:
    """주어진 텍스트를 단순 길이 제한으로 처리"""
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    else:
        # 단순 자르기 + 말줄임표
        return text[:max_len] + "..."

def build_optimized_context(search_results: List[Dict[str, Any]], question: str) -> str:
    """검색 결과와 질문을 바탕으로 LLM에 제공할 최적화된 컨텍스트 구성 (단순화된 처리)"""
    if not search_results:
        return "관련된 법 조항이나 판례 정보를 찾지 못했습니다. 질문 내용을 바탕으로 직접 답변해주세요."

    # 키워드 추출 제거 (process_text_for_context에서 사용 안 함)
    context_parts = ["### 참고 자료 (질문과 관련성이 높은 순서대로 정렬됨) ###"]
    processed_texts = [] # (타입, ID, 점수, 처리된 텍스트) 저장

    # 1. 결과 처리 및 단순 자르기
    for result in search_results:
        result_type = result.get("type", "Unknown")
        result_id = result.get("id", "N/A")
        score = result.get("score", 0.0)
        original_text = result.get("text", "")

        # 타입별 최대 길이 설정 (예: Article 500자, Precedent 700자) - 실험 필요
        max_len = 600 if result_type == "Article" else 800

        # 단순 텍스트 처리 함수 호출
        processed_text = process_text_for_context(original_text, max_len=max_len)

        if processed_text: # 처리된 텍스트가 있는 경우에만 추가
            header = f"【{result_type}: {result_id} (관련성 점수: {score:.3f})】" # 점수 소수점 자리 늘림
            details = []
            # 부가 정보는 유지 (LLM에게 유용할 수 있음)
            if result_type == "Article" and result.get('precedent_count', 0) > 0:
                 details.append(f"[관련 판례 수: {result['precedent_count']}]")
            elif result_type == "Precedent":
                 if result.get("referenced_articles"):
                     refs = ", ".join(result["referenced_articles"][:3]) # 최대 3개
                     details.append(f"[참조 법조항: {refs}]")
                 if result.get("keywords"):
                     kws = ", ".join(result["keywords"][:5]) # 최대 5개
                     details.append(f"[주요 키워드: {kws}]")

            full_context = f"{header}\n{processed_text}"
            if details:
                 full_context += "\n" + " ".join(details)

            processed_texts.append((result_type, result_id, score, full_context))

    # 2. 컨텍스트 조합
    if not processed_texts:
         return "관련 정보를 찾았으나 내용을 처리하는 데 실패했습니다. 질문 내용을 바탕으로 직접 답변해주세요."

    # 최대 컨텍스트 길이 제한 (기존 값 유지 또는 조정 필요)
    max_total_context_len = 3000
    current_total_len = 0
    final_context_parts = context_parts # 시작 부분 추가

    for _, _, _, context_str in processed_texts:
        # 길이 계산 시 개행 문자 고려 (+2)
        if current_total_len + len(context_str) + 2 < max_total_context_len:
            final_context_parts.append(context_str)
            current_total_len += len(context_str) + 2
        else:
            break # 길이 제한 도달 시 중단

    final_context_parts.append("\n### 지침: 위의 참고 자료를 바탕으로 질문에 가장 적합한 답변을 선택하세요. ###")

    return "\n\n".join(final_context_parts)

def prepare_batch_requests(df: pd.DataFrame, retrieved_contexts: Dict[int, List[Dict[str, Any]]], config: Dict) -> List[Dict[str, Any]]:
    """Batch API 요청 목록 생성 (개선된 프롬프트)"""
    batch_requests = []
    if df.empty:
        print("Evaluation data frame is empty. Cannot prepare batch requests.")
        return []
    if not config or not config.get("llm_model"):
         print("LLM model configuration is missing. Cannot prepare batch requests.")
         return []

    print(f"Preparing batch requests using LLM: {config['llm_model']}...")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preparing batch requests"):
        question = row['question']
        options = { 'A': row['A'], 'B': row['B'], 'C': row['C'], 'D': row['D'] }
        # 인덱스 idx에 해당하는 컨텍스트 가져오기
        contexts = retrieved_contexts.get(idx, [])

        # 최적화된 컨텍스트 구성
        context_str = build_optimized_context(contexts, question)

        # --- 개선된 프롬프트 ---
        prompt = f"""**문제:**
다음은 한국 형법에 관한 객관식 문제입니다. 제시된 참고 자료를 바탕으로 가장 적절한 답을 선택하고, 그 이유를 간략히 설명해주세요.

**질문:** {question}

**선택지:**
(A) {options['A']}
(B) {options['B']}
(C) {options['C']}
(D) {options['D']}

**참고 자료:**
{context_str}

**답변 형식:**
1.  **분석:** 문제의 핵심 쟁점과 관련된 법 조항/판례를 간략히 언급하고, 각 선택지를 검토합니다. (2-3 문장 내외)
2.  **최종 답변:** "정답: [A/B/C/D]" 형식으로 명확하게 제시합니다.

---
**예시:**
분석: 이 문제는 [핵심 쟁점]에 관한 것으로, 형법 제XX조 및 관련 판례 YYY에 따라 판단해야 합니다. 선택지 (A)는 ... 이유로 타당하고, (B)는 ... 이유로 틀립니다.
정답: A
---

이제 위 형식에 맞춰 답변해주세요.
"""

        request = {
            # custom_id는 결과 매칭에 사용됨 (Dataframe 인덱스 사용)
            "custom_id": f"q_{idx}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": config["llm_model"],
                "messages": [
                    {"role": "system", "content": "당신은 한국 형법 지식을 갖춘 AI 법률 전문가입니다. 주어진 문제, 선택지, 참고 자료를 바탕으로 가장 정확한 답을 논리적으로 분석하고, 지정된 형식에 맞춰 답변을 생성합니다."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 450, # 분석 내용 포함 위해 약간 늘림
                "temperature": 0.0, # 최대한 일관되고 결정적인 답변 유도
                "top_p": 0.1, # Temperature 0과 함께 사용 시 더 결정적
            }
        }
        batch_requests.append(request)

    print(f"Prepared {len(batch_requests)} batch requests.")
    return batch_requests

def run_batch_job(client: OpenAI, batch_requests: List[Dict[str, Any]], config: Dict, timestamp: str) -> Optional[str]:
    """Batch API 작업을 생성하고 완료될 때까지 모니터링"""
    if not client:
        print("OpenAI client is not available. Cannot run batch job.")
        return None
    if not batch_requests:
        print("No batch requests to process. Skipping batch job.")
        return None
    if not config or not config.get("results_dir"):
        print("Results directory configuration is missing. Cannot run batch job.")
        return None

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
    batch_input_file_id = None
    try:
        print("Uploading batch input file to OpenAI...")
        with open(batch_file_path, "rb") as f:
            batch_input_file = client.files.create(file=f, purpose="batch")
        batch_input_file_id = batch_input_file.id
        print(f"Successfully uploaded batch file. File ID: {batch_input_file_id}")
    except Exception as e:
        print(f"Error uploading batch file: {e}")
        return None

    # 배치 작업 생성
    batch_id = None
    try:
        print("Creating batch job...")
        batch_job = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h", # 24시간 내 완료 요청
            metadata={"description": f"KMMLU Criminal Law benchmark {timestamp}"}
        )
        batch_id = batch_job.id
        print(f"Successfully created batch job. Job ID: {batch_id}")
        print(f"Batch job status: {batch_job.status}")
    except Exception as e:
        print(f"Error creating batch job: {e}")
        return None

    # 배치 작업 상태 확인 및 대기
    print("\nMonitoring batch job progress...")
    start_time = time.time()
    status = None
    wait_interval = 30 # 초기 대기 시간 (초)
    max_wait_time = 3600 * 2 # 최대 대기 시간 (예: 2시간)

    while True:
        try:
            current_time = time.time()
            elapsed_time = current_time - start_time
            if elapsed_time > max_wait_time:
                 print(f"Maximum wait time ({max_wait_time}s) exceeded. Stopping monitoring.")
                 return batch_id

            status = client.batches.retrieve(batch_id)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Batch job status: {status.status} (Elapsed: {elapsed_time:.0f}s)")

            if status.status == 'completed':
                print("Batch job completed successfully.")
                break
            elif status.status in ['failed', 'cancelled', 'expired']:
                print(f"Batch job ended with status: {status.status}. Check OpenAI dashboard for details.")
                # 실패/취소 시 관련 정보 출력
                if hasattr(status, 'errors') and status.errors:
                    print("Errors reported:")
                    try:
                        # 오류가 data 필드 안에 있을 수 있음
                        error_data = status.errors.get('data', [])
                        for error in error_data[:5]: # 최대 5개 오류 표시
                            print(f"  - Code: {error.get('code')}, Message: {error.get('message')}, Line: {error.get('line')}")
                        if len(error_data) > 5:
                            print(f"  ... and {len(error_data) - 5} more errors.")
                    except Exception as e_parse:
                         print(f"Could not parse error details: {e_parse}")
                return None # 실패/취소 시 None 반환
            elif status.status in ['validating', 'in_progress', 'queued', 'cancelling']:
                # 진행 중 상태: 대기 시간 점진적 증가
                time.sleep(wait_interval)
                if wait_interval < 120: # 최대 대기 간격 설정
                    wait_interval = min(wait_interval * 1.5, 120)
            else:
                 print(f"Unknown batch status: {status.status}. Stopping monitoring.")
                 return None # 알 수 없는 상태

        except Exception as e:
            print(f"Error checking batch status: {e}. Retrying in {wait_interval}s...")
            time.sleep(wait_interval)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Batch job monitoring finished in {total_time:.2f} seconds.")

    # 완료 시 파일 정보 출력
    if status and status.status == 'completed':
        print(f"Output file ID: {status.output_file_id}")
        print(f"Error file ID: {status.error_file_id}")
        return batch_id # 성공 시 배치 ID 반환
    else:
        print("Batch job did not complete successfully.")
        return None

def extract_choice_from_response(text: str) -> Optional[str]:
    """LLM 응답 텍스트에서 최종 선택지(A, B, C, D) 추출 (정확도 향상)"""
    if not text:
        return None

    text_lower = text.lower() # 소문자 변환

    # 1. 가장 명확한 패턴 우선 검색 ("정답: A", "최종 선택: B" 등)
    #    - 대소문자 구분 없이, 콜론/공백 유연하게 처리
    #    - 문장 끝 마침표 고려
    clear_patterns = [
        r'(?:정답|최종\s*답변|최종\s*선택)\s*[:\s]\s*([a-d])\b\.?',
        r'\b([a-d])\s*(?:입니다|이다)\b\.?\s*$', # 문장 끝 "... A입니다."
        r'^\s*([a-d])\b\.?\s*$', # 문장 시작 "A." 또는 "A"
        r'따라서\s*(?:정답은|답은)?\s*([a-d])\b'
    ]
    for pattern in clear_patterns:
        # 전체 텍스트에서 검색
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()
        # 마지막 줄에서 검색 (중요도 높음)
        lines = text.strip().split('\n')
        if lines:
             last_line = lines[-1].strip()
             match = re.search(pattern, last_line, re.IGNORECASE)
             if match:
                 return match.group(1).upper()

    # 2. 괄호 안의 선택지 패턴 (예: "(A)", "[B]")
    bracket_patterns = [
        r'\( *([a-d]) *\)',
        r'\[ *([a-d]) *\]'
    ]
    # 마지막 문장에서 우선 검색
    lines = text.strip().split('\n')
    if lines:
        last_line = lines[-1].strip()
        for pattern in bracket_patterns:
            match = re.search(pattern, last_line, re.IGNORECASE)
            if match:
                return match.group(1).upper()
    # 전체 텍스트에서도 검색
    for pattern in bracket_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            # 여러 개 발견 시 마지막 것을 선택 (보통 결론 부분)
            return matches[-1].upper()

    # 3. 단순 언급 빈도 (최후의 수단, 정확도 낮음)
    #    "A", "B", "C", "D" 가 텍스트 내에 몇 번 나오는지 카운트
    #    긍정/부정 맥락 고려 시도 (간단 버전)
    counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    positive_keywords = ["정답", "옳은", "맞는", "타당", "적절", "선택"]
    negative_keywords = ["틀린", "아닌", "오답", "부적절"]

    # 문장 단위로 분리하여 분석
    sentences = re.split(r'[.!?]\s+', text)
    for sentence in sentences:
        sentence_lower = sentence.lower()
        for choice in counts.keys():
            choice_lower = choice.lower()
            # 선택지가 명확히 언급된 경우 (단독 또는 특정 구문과 함께)
            if re.search(r'\b' + choice_lower + r'\b', sentence_lower):
                score = 1
                # 긍정 키워드 있으면 가점
                if any(pos in sentence_lower for pos in positive_keywords):
                    score += 1
                # 부정 키워드 있으면 감점
                if any(neg in sentence_lower for neg in negative_keywords):
                    score -= 1
                counts[choice] += score

    # 가장 높은 점수를 가진 선택지 반환 (동점 제외)
    # 점수가 0 이하인 경우는 제외 (부정적이거나 언급 없는 경우)
    positive_counts = {k: v for k, v in counts.items() if v > 0}
    if positive_counts:
        max_score = max(positive_counts.values())
        best_choices = [choice for choice, score in positive_counts.items() if score == max_score]
        if len(best_choices) == 1:
            return best_choices[0]

    # 모든 방법 실패 시 None 반환
    print(f"Could not reliably extract choice from response: {text[:100]}...")
    return None

def process_batch_results_and_evaluate(client: OpenAI, batch_id: str, df: pd.DataFrame, config: Dict, timestamp: str) -> Optional[str]:
    """배치 작업 결과를 다운로드, 처리하고 정확도 평가"""
    if not client:
        print("OpenAI client is not available. Cannot process results.")
        return None
    if not batch_id:
        print("Batch ID is missing. Cannot process results.")
        return None
    if df.empty:
        print("Evaluation dataframe is empty. Cannot evaluate.")
        return None
    if not config or not config.get("results_dir"):
        print("Results directory configuration is missing. Cannot save results.")
        return None

    print(f"\nProcessing results for Batch Job ID: {batch_id}")

    try:
        # 1. 배치 작업 정보 가져오기
        batch_job = client.batches.retrieve(batch_id)
        if batch_job.status != 'completed':
            print(f"Batch job {batch_id} did not complete successfully. Status: {batch_job.status}")
            return None

        output_file_id = batch_job.output_file_id
        error_file_id = batch_job.error_file_id # 오류 파일 ID도 확인
        print(f"Batch job completed. Output file ID: {output_file_id}, Error file ID: {error_file_id}")

        if not output_file_id:
             print("Error: Batch job completed but no output file ID found.")
             return None

        # 2. 결과 파일 다운로드 및 처리
        output_file_path = os.path.join(config["results_dir"], f"criminal_law_batch_output_{timestamp}.jsonl")
        batch_results_raw = []
        print(f"Downloading output file {output_file_id}...")
        try:
            file_response = client.files.content(output_file_id)
            # 응답 내용을 줄 단위로 처리
            raw_content = file_response.text # content 대신 text 사용 (최신 openai 라이브러리)
            with open(output_file_path, 'w', encoding='utf-8') as f:
                for line in raw_content.strip().split('\n'):
                    if line.strip():
                        try:
                            batch_results_raw.append(json.loads(line))
                            f.write(line + '\n') # 원본 저장
                        except json.JSONDecodeError:
                            print(f"Warning: Could not decode JSON line: {line[:100]}...")
            print(f"Successfully downloaded and saved {len(batch_results_raw)} raw results to {output_file_path}")
        except Exception as e:
            print(f"Error downloading or saving output file {output_file_id}: {e}")
            return None # 결과 파일 없으면 평가 불가

        # 3. 오류 파일 처리 (선택적)
        if error_file_id:
            error_file_path = os.path.join(config["results_dir"], f"criminal_law_batch_errors_{timestamp}.jsonl")
            print(f"Downloading error file {error_file_id}...")
            try:
                error_response = client.files.content(error_file_id)
                with open(error_file_path, 'w', encoding='utf-8') as f:
                    f.write(error_response.text)
                print(f"Downloaded and saved error file to {error_file_path}")
            except Exception as e:
                print(f"Warning: Could not download or save error file {error_file_id}: {e}")

        # 4. 정확도 평가
        correct_count = 0
        processed_count = 0
        results_data = [] # 평가 결과를 저장할 리스트

        print("Evaluating predictions...")
        for result_entry in tqdm(batch_results_raw, desc="Evaluating results"):
            custom_id = result_entry.get('custom_id')
            response_body = result_entry.get('response', {}).get('body') if result_entry.get('response') else None
            error_body = result_entry.get('error')

            if not custom_id:
                print(f"Skipping result entry with missing custom_id: {result_entry}")
                continue

            try:
                # custom_id에서 원래 DataFrame 인덱스 추출 (q_ 제거)
                if custom_id.startswith('q_'):
                     idx = int(custom_id[2:])
                else:
                     print(f"Warning: Unexpected custom_id format: {custom_id}. Skipping.")
                     continue

                # 원본 데이터 가져오기 (존재하는지 확인)
                if idx not in df.index:
                     print(f"Warning: Index {idx} from custom_id not found in the original DataFrame. Skipping.")
                     continue

                original_row = df.loc[idx]
                processed_count += 1
                predicted_answer = None
                response_text = ""
                is_correct = False
                # 실제 정답 변환 (1->A, 2->B, ...)
                actual_answer = chr(64 + original_row['answer'])

                if error_body:
                    response_text = f"Error: {error_body.get('message', 'Unknown error')}"
                    predicted_answer = "Error" # 예측 실패로 처리
                elif response_body and response_body.get('choices'):
                    # 첫 번째 choice의 메시지 내용 추출
                    message = response_body['choices'][0].get('message', {})
                    response_text = message.get('content', '').strip()
                    # 응답 텍스트에서 선택지 추출 (개선된 함수 사용)
                    predicted_answer = extract_choice_from_response(response_text)

                    if predicted_answer:
                        is_correct = (predicted_answer == actual_answer)
                        if is_correct:
                            correct_count += 1
                    else:
                        predicted_answer = "Extraction Failed" # 추출 실패 명시
                else:
                     response_text = "Invalid/Empty Response"
                     predicted_answer = "No Response"

                results_data.append({
                    'question_id': idx,
                    'question': original_row['question'],
                    'A': original_row['A'], 'B': original_row['B'], 'C': original_row['C'], 'D': original_row['D'],
                    'predicted': predicted_answer,
                    'actual': actual_answer,
                    'is_correct': is_correct,
                    'response': response_text # LLM의 전체 응답 저장
                })
            except (ValueError, KeyError, IndexError) as e:
                print(f"Error processing result entry for custom_id {custom_id}: {e}. Entry: {result_entry}")
            except Exception as e:
                 print(f"An unexpected error occurred processing custom_id {custom_id}: {e}")

        if processed_count == 0:
            print("No results were processed successfully. Evaluation cannot proceed.")
            return None

        # 정확도 계산
        accuracy = correct_count / processed_count if processed_count > 0 else 0.0
        print(f"\n--- Evaluation Summary ---")
        print(f"Total results processed: {processed_count} / {len(df)}")
        print(f"Correct answers: {correct_count}")
        print(f"Accuracy: {accuracy:.4f} ({accuracy:.2%})")

        # 5. 상세 결과 저장
        results_df = pd.DataFrame(results_data)
        results_file_path = os.path.join(config["results_dir"], f"criminal_law_evaluation_results_{timestamp}.csv")
        try:
            results_df.to_csv(results_file_path, index=False, encoding='utf-8-sig') # UTF-8 with BOM for Excel compatibility
            print(f"Saved detailed evaluation results to {results_file_path}")
            return results_file_path # 성공 시 결과 파일 경로 반환
        except IOError as e:
             print(f"Error saving evaluation results file: {e}")
             return None

    except Exception as e:
        print(f"An error occurred during batch result processing and evaluation: {e}")
        return None

def analyze_results(results_file_path: str, config: Dict):
    """평가 결과를 분석 (간소화된 콘솔 출력 버전)"""
    print(f"\n--- Analyzing Results from: {results_file_path} ---")
    if not os.path.exists(results_file_path):
        print(f"Error: Results file not found at {results_file_path}")
        return

    try:
        results_df = pd.read_csv(results_file_path)
        # 'is_correct' 컬럼을 boolean 타입으로 변환 (True/False 문자열 처리 포함)
        results_df['is_correct'] = results_df['is_correct'].apply(lambda x: str(x).lower() == 'true')

    except Exception as e:
        print(f"Error reading or processing results file: {e}")
        return

    if results_df.empty:
        print("Results data is empty. Skipping analysis.")
        return

    # --- 요약 통계 ---
    total_questions = len(results_df)
    correct_answers = results_df['is_correct'].sum()
    incorrect_answers = total_questions - correct_answers
    accuracy = correct_answers / total_questions if total_questions > 0 else 0.0
    # 예측 실패/오류 건수
    error_predictions = results_df[results_df['predicted'].isin(['Error', 'Extraction Failed', 'No Response'])].shape[0]

    print("\n===== Benchmark Result Summary =====")
    print(f"Model Used: {config.get('llm_model', 'N/A')}")
    print(f"Total Questions: {total_questions}")
    print(f"Correct Answers: {correct_answers}")
    print(f"Incorrect Answers: {incorrect_answers}")
    print(f"Prediction Errors/Failures: {error_predictions}")
    print(f"Accuracy (excluding errors): {correct_answers / (total_questions - error_predictions) if (total_questions - error_predictions) > 0 else 0.0:.4f}")
    print(f"Overall Accuracy (including errors as incorrect): {accuracy:.4f} ({accuracy:.2%})")

    # 정답/오답 분석
    options = ['A', 'B', 'C', 'D']
    
    # 각 선택지별 정답률 계산
    prediction_counts = results_df[results_df['predicted'].isin(options)]['predicted'].value_counts()
    actual_counts = results_df['actual'].value_counts()
    
    print("\n--- Choice Distribution ---")
    print("Predicted answers distribution:")
    for option in options:
        count = prediction_counts.get(option, 0)
        print(f"{option}: {count} ({count/total_questions:.2%})")
    
    print("\nActual answers distribution:")
    for option in options:
        count = actual_counts.get(option, 0)
        print(f"{option}: {count} ({count/total_questions:.2%})")
    
    # 오답 유형 분석
    incorrect_df = results_df[(results_df['is_correct'] == False) & (results_df['predicted'].isin(options))]
    if not incorrect_df.empty:
        incorrect_predictions = incorrect_df['predicted'].value_counts()
        print("\n--- Incorrect Answer Analysis ---")
        print("Most Frequent Incorrect Predictions:")
        for option, count in incorrect_predictions.items():
            print(f"{option}: {count}")

    # 결과 파일 확인 방법 안내
    print(f"\nDetailed results have been saved to: {results_file_path}")
    print("You can open this CSV file to review all predictions and responses.")

def main():
    """전체 프로세스 실행 함수"""
    print("=== Criminal Law RAG + LLM Evaluation System ===")
    print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    start_time = time.time()

    # 1. 설정 로드 및 초기화
    config, embedding_model, openai_client = load_config_and_initialize()
    if not config or not embedding_model or not openai_client:
        print("Critical initialization failed. Exiting.")
        return

    # 2. 데이터 로드
    print("\n=== Loading Data ===")
    articles = load_articles_from_pdf(config["pdf_path"])
    precedents = load_precedents_from_json(config["precedent_dir"], sample_size=3000)
    
    # 평가 데이터 로드
    try:
        eval_df = pd.read_csv(config["test_csv_path"])
        # 데이터 검증 (필요한 컬럼 확인 등)
        required_cols = ['question', 'A', 'B', 'C', 'D', 'answer']
        if not all(col in eval_df.columns for col in required_cols):
             raise ValueError(f"Evaluation CSV must contain columns: {required_cols}")
        # 'answer' 컬럼 타입 확인 및 변환 (숫자 형태 가정)
        eval_df['answer'] = pd.to_numeric(eval_df['answer'], errors='coerce')
        eval_df.dropna(subset=['answer'], inplace=True) # answer 없는 행 제거
        eval_df['answer'] = eval_df['answer'].astype(int)

        print(f"\nLoaded and validated {len(eval_df)} questions for evaluation from {config['test_csv_path']}")
    except Exception as e:
        print(f"Error loading evaluation data: {e}")
        return

    # 3. Neo4j 연결 및 설정
    print("\n=== Connecting to Neo4j ===")
    neo4j_driver = connect_neo4j(config["neo4j_uri"], (config["neo4j_username"], config["neo4j_password"]))
    if not neo4j_driver:
        print("Failed to connect to Neo4j. Exiting.")
        return
    
    setup_neo4j_constraints_and_indexes(neo4j_driver, config["embedding_dimension"])

    # 4. 그래프 데이터 생성
    print("\n=== Creating Graph Data ===")
    create_graph_nodes_and_relationships(neo4j_driver, articles, precedents, embedding_model)

    # 5. RAG 컨텍스트 검색
    print("\n=== Performing RAG Context Search ===")
    retrieved_contexts = {}
    for idx, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Retrieving contexts"):
        question = row['question']
        try:
            contexts = retrieve_context_from_graph(neo4j_driver, question, embedding_model, top_k=8)
            retrieved_contexts[idx] = contexts
        except Exception as e:
            print(f"Error during RAG search for question index {idx}: {e}")
            retrieved_contexts[idx] = [] # 오류 발생 시 빈 컨텍스트

    print(f"Completed RAG search. Retrieved contexts for {len(retrieved_contexts)} questions.")

    # 6. 배치 요청 준비
    print("\n=== Preparing Batch Requests ===")
    batch_requests = prepare_batch_requests(eval_df, retrieved_contexts, config)

    # 7. Batch API 실행
    print("\n=== Executing Batch API Job ===")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_start = time.time()
    batch_id = run_batch_job(openai_client, batch_requests, config, timestamp)
    batch_end = time.time()
    
    if not batch_id:
        print("Batch job execution failed. Exiting.")
        close_neo4j(neo4j_driver)
        return

    # 8. 결과 처리 및 평가
    print("\n=== Processing Results and Evaluating ===")
    results_file = process_batch_results_and_evaluate(openai_client, batch_id, eval_df, config, timestamp)
    
    if not results_file:
        print("Result processing failed. Exiting.")
        close_neo4j(neo4j_driver)
        return

    # 9. 결과 분석
    print("\n=== Analyzing Results ===")
    analyze_results(results_file, config)

    # 10. 정리 및 마무리
    close_neo4j(neo4j_driver)

    end_time = time.time()
    batch_duration = batch_end - batch_start
    code_duration = (end_time - start_time) - batch_duration
    
    print("\n=== Process Complete ===")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    print(f"Batch API execution time: {batch_duration:.2f} seconds")
    print(f"Code execution time (excluding Batch API): {code_duration:.2f} seconds")
    print(f"Results saved to: {results_file}")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()