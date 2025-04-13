#!/usr/bin/env python3
# Criminal Law RAG + LLM Evaluation System with improved GraphRAG functionality

import os
import json
import re
import time
import random
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import Counter, defaultdict

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
    """JSON 파일에서 판례 정보를 로드하고 정제 (관계 추출 개선)"""
    print(f"Loading precedents from directory: {precedent_dir}")
    if not os.path.isdir(precedent_dir):
        print(f"Error: Precedent directory not found at {precedent_dir}")
        return []

    precedents = []
    # 법조항 패턴 (공백 유연성 증가, 더 다양한 형태 매칭)
    rule_pattern = re.compile(r'제\s*\d+\s*조(?:의\s*\d+)?(?:\s*\(.*?\))?')
    files_processed = 0
    files_skipped = 0
    keyword_stats = Counter()  # 키워드 통계
    rule_refs_stats = Counter()  # 참조 법조항 통계

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
                    "case_name": info.get("caseNm", "Unnamed Case"),
                    "judgment_summary": data.get("jdgmn", ""),
                    "full_summary": " ".join([s.get("summ_contxt", "") for s in summary_list]).strip(),
                    "keywords": [kw.get("keyword") for kw in keyword_list if kw.get("keyword")],
                    "referenced_rules_raw": ref_info.get("reference_rules", ""),
                    "referenced_cases_raw": ref_info.get("reference_court_case", ""),
                }

                # === 개선됨: 키워드가 없는 경우 간단한 키워드 추출 ===
                if not precedent_info["keywords"] and precedent_info["full_summary"]:
                    # 간단한 키워드 추출 (명사 빈도 기반)
                    extracted_keywords = extract_simple_keywords(precedent_info["full_summary"], top_n=5)
                    precedent_info["keywords"] = extracted_keywords
                    precedent_info["keywords_extracted"] = True  # 자동 추출 플래그

                # === 개선됨: 참조 법조항 추출 강화 ===
                raw_rules = precedent_info["referenced_rules_raw"].split(',') if precedent_info["referenced_rules_raw"] else []
                all_text = precedent_info["full_summary"] + " " + precedent_info["judgment_summary"]
                
                # 참조 법조항 정제 - 모든 가능한 소스에서 추출
                cleaned_rules = set()
                # 1. 명시적 참조 목록에서 추출
                for rule in raw_rules:
                    matches = rule_pattern.findall(rule.strip())
                    for match in matches:
                        clean_match = re.sub(r'\s+', '', match)  # 공백 제거
                        cleaned_rules.add(clean_match)
                
                # 2. 전체 텍스트에서 법조항 패턴 추출 (명시적 참조가 없거나 불완전한 경우)
                if len(cleaned_rules) < 2 and all_text:  # 참조 법조항이 없거나 적은 경우
                    text_matches = rule_pattern.findall(all_text)
                    for match in text_matches:
                        clean_match = re.sub(r'\s+', '', match)
                        cleaned_rules.add(clean_match)
                
                precedent_info["referenced_rules"] = list(cleaned_rules)

                # === 개선됨: 참조 판례 정제 및 추가 정보 ===
                raw_cases = precedent_info["referenced_cases_raw"].split(',') if precedent_info["referenced_cases_raw"] else []
                referenced_cases = []
                for case in raw_cases:
                    case_clean = case.strip()
                    if case_clean:
                        # 판례 ID 추출 시도 (예: "대법원 2020다12345" -> "2020다12345")
                        case_id_match = re.search(r'\d+\s*[가-힣]+\s*\d+', case_clean)
                        if case_id_match:
                            case_id = re.sub(r'\s+', '', case_id_match.group(0))
                            referenced_cases.append({
                                "full_reference": case_clean,
                                "case_id": case_id
                            })
                        else:
                            referenced_cases.append({
                                "full_reference": case_clean,
                                "case_id": None
                            })
                
                precedent_info["referenced_cases"] = referenced_cases

                # 임베딩할 텍스트 준비 (full_summary 우선, 없으면 judgment_summary)
                precedent_info["text_for_embedding"] = precedent_info["full_summary"] or precedent_info["judgment_summary"]

                # 유효한 데이터만 추가 (임베딩할 텍스트가 있어야 함)
                if precedent_info["text_for_embedding"]:
                    precedents.append(precedent_info)
                    files_processed += 1
                    
                    # 통계 수집
                    keyword_stats.update(precedent_info["keywords"])
                    rule_refs_stats.update(precedent_info["referenced_rules"])
                else:
                    files_skipped += 1

        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {filename}")
            files_skipped += 1
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            files_skipped += 1

    # === 개선됨: 데이터 로드 후 통계 출력 ===
    print(f"Loaded {len(precedents)} valid precedents. Skipped {files_skipped} files.")
    print(f"Total unique keywords found: {len(keyword_stats)}")
    print(f"Total unique referenced rules found: {len(rule_refs_stats)}")
    print(f"Top 10 most common keywords: {keyword_stats.most_common(10)}")
    print(f"Top 10 most referenced rules: {rule_refs_stats.most_common(10)}")
    
    # 키워드가 없거나 참조 법조항이 없는 판례 수 출력
    no_keywords = sum(1 for p in precedents if not p["keywords"])
    no_rules = sum(1 for p in precedents if not p["referenced_rules"])
    print(f"Precedents without keywords: {no_keywords} ({no_keywords/len(precedents):.1%})")
    print(f"Precedents without referenced rules: {no_rules} ({no_rules/len(precedents):.1%})")

    # 샘플링
    if sample_size is not None and len(precedents) > sample_size:
        print(f"Sampling {sample_size} precedents from {len(precedents)}...")
        random.seed(42) # 재현성을 위한 시드 고정
        precedents = random.sample(precedents, sample_size)
        print(f"Selected {len(precedents)} precedents after sampling.")

    return precedents

def extract_simple_keywords(text: str, top_n: int = 5) -> List[str]:
    """텍스트에서 간단한 키워드 추출 (빈도 기반)"""
    # 한국어 불용어 목록
    stopwords = set([
        "의", "가", "이", "은", "들", "는", "좀", "잘", "걍", "과", "도", "를", "으로", "자", "에", "와", "한", "하다",
        "것", "그", "저", "수", "때", "등", "및", "제", "조", "항", "관련", "대한", "대해", "위한", "있는", "하는",
        "그리고", "그러나", "그래서", "하지만", "또는", "다른", "모든", "어떤", "누구", "무엇", "언제", "어디서", "어떻게", "왜",
        "입니다", "습니다", "합니다", "에서", "에게", "부터", "까지", "보다", "만", "같이", "처럼", "따라", "통해"
    ])
    
    # 2글자 이상 한글 단어 추출
    words = re.findall(r'[가-힣]{2,}', text)
    words = [word for word in words if word not in stopwords]
    
    # 빈도수 계산 및 상위 키워드 추출
    counter = Counter(words)
    top_keywords = [word for word, _ in counter.most_common(top_n)]
    
    return top_keywords

def connect_neo4j(uri: str, auth: tuple) -> Optional[Driver]:
    """Neo4j 데이터베이스에 연결 (클라우드 Neo4j 연결 지원)"""
    print(f"Attempting to connect to Neo4j at {uri}...")
    try:
        # 중요: URI 스키마 확인하여 적절한 설정 사용
        is_secured_protocol = any(prefix in uri for prefix in ['+s', '+ssc'])
        
        # 보안 URI 스키마에 따라 설정 구성
        if is_secured_protocol:
            # neo4j+s://, bolt+s:// 등은 이미 암호화 설정이 포함됨
            driver = GraphDatabase.driver(
                uri, 
                auth=auth,
                max_connection_lifetime=3600,
                max_connection_pool_size=50,
                connection_acquisition_timeout=60
                # encrypted와 trust 설정을 제거함
            )
        else:
            # 비보안 URI (neo4j://, bolt://)는 암호화 설정 필요
            driver = GraphDatabase.driver(
                uri, 
                auth=auth,
                max_connection_lifetime=3600,
                max_connection_pool_size=50,
                connection_acquisition_timeout=60,
                encrypted=True,
                trust="TRUST_SYSTEM_CA_SIGNED_CERTIFICATES"
            )
        
        # 연결 테스트
        with driver.session(database="neo4j") as session:
            result = session.run("RETURN 1 as n")
            record = result.single()
            if record and record["n"] == 1:
                print("Successfully connected to Neo4j database.")
            else:
                print("Warning: Connection established but verification failed.")
        
        return driver
    except Exception as e:
        print(f"Failed to connect to Neo4j: {e}")
        print("Please ensure Neo4j is running and the connection details are correct.")
        
        # 디버깅 정보
        print(f"\nConnection Details for Debugging:")
        print(f"URI: {uri}")
        print(f"Username provided: {bool(auth[0])}")
        print(f"Password provided: {bool(auth[1])}")
        
        return None

def close_neo4j(driver: Optional[Driver]):
    """Neo4j 드라이버 연결 종료"""
    if driver:
        driver.close()
        print("Neo4j driver connection closed.")

def setup_neo4j_constraints_and_indexes(driver: Driver, dimension: int):
    """Neo4j 제약조건 및 벡터 인덱스 설정 (클라우드 Neo4j 호환성 개선)"""
    print("Setting up Neo4j constraints and indexes...")
    try:
        with driver.session(database="neo4j") as session:
            # === 개선됨: 제약조건 (클라우드 Neo4j 호환) ===
            constraints = [
                "CREATE CONSTRAINT article_id IF NOT EXISTS FOR (a:Article) REQUIRE a.id IS UNIQUE",
                "CREATE CONSTRAINT precedent_id IF NOT EXISTS FOR (p:Precedent) REQUIRE p.id IS UNIQUE",
                "CREATE CONSTRAINT keyword_text IF NOT EXISTS FOR (k:Keyword) REQUIRE k.text IS UNIQUE"
            ]
            
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    print(f"Warning: Could not create constraint ({constraint}): {e}")
            
            print("Constraints created or verified.")

            # === 개선됨: 일반 인덱스 (클라우드 Neo4j 호환) ===
            indexes = [
                "CREATE INDEX keyword_text_idx IF NOT EXISTS FOR (k:Keyword) ON (k.text)",
                "CREATE INDEX article_id_idx IF NOT EXISTS FOR (a:Article) ON (a.id)",
                "CREATE INDEX precedent_id_idx IF NOT EXISTS FOR (p:Precedent) ON (p.id)"
            ]
            
            for index in indexes:
                try:
                    session.run(index)
                except Exception as e:
                    print(f"Warning: Could not create index ({index}): {e}")

            # 벡터 인덱스 - 클라우드 Neo4j에 맞게 수정
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
                    # 벡터 인덱스 생성 실패는 경고로 처리
                    print(f"Warning: Could not create or verify vector index: {e}")
                    print("Vector search functionality might not be available.")

            # 인덱스 활성화 대기 (클라우드 환경에 맞게 타임아웃 조정)
            print("Waiting for indexes to come online (up to 30 seconds)...")
            try:
                session.run("CALL db.awaitIndexes(30000)") # 30초 타임아웃
                print("Indexes are online.")
            except Exception as e:
                 print(f"Warning: Could not explicitly wait for indexes: {e}")

    except Exception as e:
        print(f"An error occurred during Neo4j setup: {e}")

def create_graph_nodes_and_relationships(driver: Driver, articles: Dict[str, str], precedents: List[Dict[str, Any]], embed_model):
    """법 조항, 판례, 키워드 노드 및 관계를 배치 처리로 생성 (관계 생성 확실히)"""
    if not driver:
        print("Neo4j driver not available. Skipping graph creation.")
        return
    if not embed_model:
        print("Embedding model not available. Skipping graph creation.")
        return
    if not articles and not precedents:
        print("No articles or precedents data provided. Skipping graph creation.")
        return

    print(f"Starting graph data creation process... Articles: {len(articles)}, Precedents: {len(precedents)}")
    start_time = time.time()

    # 배치 처리 설정
    batch_size = 500 # 한 번에 처리할 노드/관계 수

    # --- 1. 법 조항 노드 처리 ---
    articles_batch = []
    if articles:
        print("Processing Article nodes...")
        article_texts = list(articles.values())
        article_ids = list(articles.keys())

        # 임베딩 (배치 처리)
        print(f"Embedding {len(article_texts)} articles...")
        try:
            article_embeddings = embed_model.embed_documents(article_texts, chunk_size=100) # 임베딩 API 배치 크기 조절
            print(f"Successfully embedded {len(article_embeddings)} articles.")

            if len(article_embeddings) != len(article_ids):
                 print(f"Warning: Mismatch between article count ({len(article_ids)}) and embedding count ({len(article_embeddings)}).")
                 # 길이를 맞추거나 오류 처리 필요 - 여기서는 짧은 쪽 기준으로 진행
                 min_len = min(len(article_ids), len(article_embeddings))
                 article_ids = article_ids[:min_len]
                 article_texts = article_texts[:min_len]
                 article_embeddings = article_embeddings[:min_len]

            for i in range(len(article_ids)):
                articles_batch.append({
                    "id": article_ids[i],
                    "text": article_texts[i],
                    "embedding": article_embeddings[i]
                })

            # Neo4j에 Article 노드 생성/업데이트 (배치)
            print(f"Writing {len(articles_batch)} Article nodes to Neo4j...")
            with driver.session(database="neo4j") as session:
                for i in range(0, len(articles_batch), batch_size):
                    batch = articles_batch[i:i+batch_size]
                    session.run(
                        """
                        UNWIND $batch as article_data
                        MERGE (a:Article {id: article_data.id})
                        SET a.text = article_data.text,
                            a.embedding = article_data.embedding,
                            a.last_updated = timestamp()
                        """,
                        batch=batch
                    )
                print(f"Successfully created/updated {len(articles_batch)} Article nodes.")

        except Exception as e:
            print(f"Error during Article node processing or embedding: {e}")
            articles_batch = [] # 오류 시 비움

    # --- 2. 판례 및 키워드 노드, 관계 처리 ---
    precedents_batch = []
    relationships_batch = defaultdict(list) # 관계 유형별 리스트
    all_keywords = set()
    precedents_embedded = 0
    precedents_skipped = 0

    if precedents:
        print("Processing Precedent nodes and relationships...")
        # 임베딩할 텍스트 목록 준비
        texts_to_embed = []
        valid_precedent_indices = []
        for i, p in enumerate(precedents):
            text = p.get("text_for_embedding")
            if text:
                texts_to_embed.append(text)
                valid_precedent_indices.append(i)
            else:
                precedents_skipped += 1
                # print(f"Skipping precedent {p.get('case_id', 'N/A')} due to missing text for embedding.")

        if texts_to_embed:
            print(f"Embedding {len(texts_to_embed)} precedents...")
            try:
                precedent_embeddings = embed_model.embed_documents(texts_to_embed, chunk_size=100)
                print(f"Successfully embedded {len(precedent_embeddings)} precedents.")
                precedents_embedded = len(precedent_embeddings)

                if len(precedent_embeddings) != len(valid_precedent_indices):
                    print(f"Warning: Mismatch between valid precedent count ({len(valid_precedent_indices)}) and embedding count ({len(precedent_embeddings)}). Adjusting...")
                    min_len = min(len(valid_precedent_indices), len(precedent_embeddings))
                    valid_precedent_indices = valid_precedent_indices[:min_len]
                    precedent_embeddings = precedent_embeddings[:min_len]
                    precedents_embedded = min_len # 실제 임베딩된 수 업데이트

                embedding_map = {idx: emb for idx, emb in zip(valid_precedent_indices, precedent_embeddings)}

                print("Preparing Precedent nodes and relationships batch...")
                for i, p in enumerate(tqdm(precedents, desc="Preparing batches")):
                    if i not in embedding_map: # 임베딩 실패 또는 스킵된 경우 건너뜀
                        continue

                    case_id = p.get("case_id")
                    if not case_id:
                        print(f"Skipping precedent at index {i} due to missing case_id.")
                        continue

                    # 판례 노드 데이터
                    precedent_node = {
                        "id": case_id,
                        "name": p.get("case_name"),
                        "judgment_summary": p.get("judgment_summary"),
                        "full_summary": p.get("full_summary"),
                        "embedding": embedding_map[i],
                        "keywords_original": p.get("keywords", []), # 원본 키워드 저장
                        "keywords_extracted": p.get("keywords_extracted", False) # 자동 추출 여부
                    }
                    precedents_batch.append(precedent_node)

                    # 키워드 및 HAS_KEYWORD 관계
                    current_keywords = set(p.get("keywords", []))
                    all_keywords.update(current_keywords) # 전체 키워드 집합에 추가
                    for kw in current_keywords:
                        relationships_batch["HAS_KEYWORD"].append({
                            "case_id": case_id,
                            "keyword_text": kw,
                            "source": "extracted" if p.get("keywords_extracted") else "original"
                        })

                    # REFERENCES_ARTICLE 관계 (개선된 매핑 로직)
                    referenced_rules = p.get("referenced_rules", [])
                    original_refs = p.get("referenced_rules_raw", "").split(',') if p.get("referenced_rules_raw") else []
                    
                    # 매핑 시도 (article ID 집합 필요)
                    article_ids_set = set(articles.keys()) if articles else set()
                    
                    for rule_ref in referenced_rules: # 정제된 rule ID 사용
                        match_type = "unknown"
                        original_full_ref = next((orig.strip() for orig in original_refs if rule_ref in re.sub(r'\s+', '', orig)), rule_ref) # 원본 참조 찾기

                        if rule_ref in article_ids_set:
                            match_type = "exact"
                            relationships_batch["REFERENCES_ARTICLE"].append({
                                "case_id": case_id, "article_ref": rule_ref, "match_type": match_type, "original_ref": original_full_ref
                            })
                        else:
                            # 부분 또는 매핑된 참조 시도 (예: '제10조'가 '제10조(상해)'에 포함되는 경우)
                            partial_match = next((a_id for a_id in article_ids_set if rule_ref in a_id or a_id in rule_ref), None)
                            if partial_match:
                                match_type = "partial" if rule_ref in partial_match or partial_match in rule_ref else "mapped" # 더 정교한 매핑 로직 필요 시 추가
                                relationships_batch["REFERENCES_ARTICLE"].append({
                                    "case_id": case_id, "article_ref": partial_match, "match_type": match_type, "original_ref": original_full_ref
                                })
                            # else: 매칭 실패 시 관계 생성 안 함

                    # REFERENCES_CASE 관계
                    referenced_cases = p.get("referenced_cases", [])
                    for ref_case in referenced_cases:
                        if ref_case.get("case_id"): # 유효한 판례 ID가 있는 경우만
                            relationships_batch["REFERENCES_CASE"].append({
                                "case_id": case_id,
                                "referenced_case_id": ref_case["case_id"],
                                "full_reference": ref_case.get("full_reference")
                            })

            except Exception as e:
                print(f"Error during Precedent embedding or batch preparation: {e}")
                # 오류 발생 시 관련 배치 초기화 또는 부분 처리 결정 필요
                precedents_batch = []
                relationships_batch = defaultdict(list)
                all_keywords = set()

    # --- 3. Neo4j에 데이터 쓰기 (배치) ---
    # 이 부분부터 기존 코드와 연결됩니다.
    if precedents_batch: # 이제 precedents_batch가 정의되었습니다.
        print(f"Embedding complete for {precedents_embedded} precedents (skipped {precedents_skipped}). Writing to Neo4j...")
        try:
            with driver.session(database="neo4j") as session:
                # --- 판례 노드 생성/업데이트 (배치) ---
                print(f"Writing {len(precedents_batch)} Precedent nodes to Neo4j...")
                for i in range(0, len(precedents_batch), batch_size):
                    batch = precedents_batch[i:i+batch_size]
                    session.run(
                        """
                        UNWIND $batch as p_data
                        MERGE (p:Precedent {id: p_data.id})
                        SET p.name = p_data.name,
                            p.judgment_summary = p_data.judgment_summary,
                            p.full_summary = p_data.full_summary,
                            p.embedding = p_data.embedding,
                            p.keywords_original = p_data.keywords_original,
                            p.keywords_extracted = p_data.keywords_extracted,
                            p.last_updated = timestamp()
                        """,
                        batch=batch
                    )
                print(f"Successfully created/updated {len(precedents_batch)} Precedent nodes.")

                # --- 키워드 노드 생성 (존재하지 않는 경우 - 배치) ---
                if all_keywords:
                    print(f"Ensuring {len(all_keywords)} Keyword nodes exist...")
                    keyword_list = list(all_keywords)
                    for i in range(0, len(keyword_list), batch_size * 10): # 키워드는 더 큰 배치로 처리 가능
                        batch = keyword_list[i:i+batch_size*10]
                        session.run(
                            """
                            UNWIND $keywords as keyword_text
                            MERGE (k:Keyword {text: keyword_text})
                            """,
                            keywords=batch
                        )
                    print(f"Ensured {len(all_keywords)} Keyword nodes exist.")

                # --- 관계 생성 (배치) ---
                rel_types_to_create = ["HAS_KEYWORD", "REFERENCES_ARTICLE", "REFERENCES_CASE"]
                total_rels_attempted = 0
                total_rels_created = 0

                for rel_type in rel_types_to_create:
                    rels = relationships_batch.get(rel_type, [])
                    if not rels:
                        print(f"No relationships of type {rel_type} to create.")
                        continue

                    print(f"Writing {len(rels)} relationships of type {rel_type}...")
                    created_count_for_type = 0
                    rel_batch_size = 5000 # 관계 배치 크기

                    # 관계 유형별 Cypher 쿼리 정의
                    cypher_query = ""
                    if rel_type == "HAS_KEYWORD":
                        cypher_query = """
                        UNWIND $rels as rel
                        MATCH (p:Precedent {id: rel.case_id})
                        MATCH (k:Keyword {text: rel.keyword_text})
                        MERGE (p)-[r:HAS_KEYWORD {source: rel.source}]->(k)
                        SET r.last_updated = timestamp()
                        RETURN count(r) as created_count
                        """
                    elif rel_type == "REFERENCES_ARTICLE":
                        cypher_query = """
                        UNWIND $rels as rel
                        MATCH (p:Precedent {id: rel.case_id})
                        MATCH (a:Article {id: rel.article_ref})
                        MERGE (p)-[r:REFERENCES_ARTICLE]->(a)
                        SET r.match_type = rel.match_type,
                            r.original_ref = rel.original_ref,
                            r.last_updated = timestamp()
                        RETURN count(r) as created_count
                        """
                    elif rel_type == "REFERENCES_CASE":
                        cypher_query = """
                        UNWIND $rels as rel
                        MATCH (p1:Precedent {id: rel.case_id})
                        MATCH (p2:Precedent {id: rel.referenced_case_id})
                        MERGE (p1)-[r:REFERENCES_CASE]->(p2)
                        SET r.full_reference = rel.full_reference,
                            r.last_updated = timestamp()
                        RETURN count(r) as created_count
                        """

                    if cypher_query:
                        total_rels_attempted += len(rels)
                        for i in range(0, len(rels), rel_batch_size):
                            batch = rels[i:i+rel_batch_size]
                            result = session.run(cypher_query, rels=batch)
                            summary = result.consume()
                            created_count_for_type += summary.counters.relationships_created
                        total_rels_created += created_count_for_type
                        print(f"Attempted to create/update {len(rels)} {rel_type} relationships. Actual created: {created_count_for_type}")
                    else:
                        print(f"Warning: No Cypher query defined for relationship type {rel_type}")

                print(f"\nTotal relationships attempted: {total_rels_attempted}")
                print(f"Total relationships actually created: {total_rels_created}")

                # === 관계 검증 (실제 생성된 관계 수 확인) ===
                print("\nVerifying created relationships counts in DB...")
                verification = session.run("""
                    MATCH ()-[r]->()
                    RETURN type(r) as rel_type, count(r) as count
                """)
                rel_counts = {record['rel_type']: record['count'] for record in verification}
                print(f"- HAS_KEYWORD: {rel_counts.get('HAS_KEYWORD', 0)} relationships")
                print(f"- REFERENCES_ARTICLE: {rel_counts.get('REFERENCES_ARTICLE', 0)} relationships")
                print(f"- REFERENCES_CASE: {rel_counts.get('REFERENCES_CASE', 0)} relationships")

                # 생성 시도 수와 실제 수 비교 (경고용) - 각 타입별로 비교하는 것이 더 정확
                # if total_rels_created != sum(rel_counts.values()): # 단순 합계 비교는 부정확할 수 있음
                #     print(f"Warning: Total relationship count mismatch (attempted: {total_rels_created}, actual in DB: {sum(rel_counts.values())})")

        except Exception as e:
            print(f"Error writing data to Neo4j: {e}")
            import traceback
            traceback.print_exc() # 상세 에러 로그 출력

    else: # precedents_batch가 비어있는 경우 (precedents 데이터가 없거나 처리 중 오류 발생)
        print("No valid precedent data processed or available to write to Neo4j.")

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
    """그래프 데이터베이스에서 관련 법 조항 및 판례 검색 (GraphRAG 강화 및 벡터 검색 수정)"""
    if not driver:
        print("Neo4j driver is not available. Cannot retrieve context.")
        return []
    if not embed_model:
        print("Embedding model is not available. Cannot retrieve context.")
        return []

    results = []
    all_results_map = {} # 중복 방지 및 결과 통합용 (key: (type, id))

    try:
        query_embedding = embed_model.embed_query(query_text)
        query_keywords = extract_query_keywords(query_text, max_keywords=5)
        article_pattern = r'(제\s*\d+\s*조(?:의\s*\d+)?(?:\s*\(.*?\))?)'
        article_matches = re.findall(article_pattern, query_text)
        query_articles = [re.sub(r'\s+', '', match) for match in article_matches]

        print(f"Query keywords: {query_keywords}")
        print(f"Query article references: {query_articles}")

        with driver.session(database="neo4j") as session:
            # === 다단계 검색 전략 ===

            # == 1. 직접 참조된 법조항 검색 ==
            if query_articles:
                try:
                    direct_article_query = """
                    MATCH (a:Article)
                    WHERE a.id IN $article_ids OR
                          ANY(id IN $article_ids WHERE a.id STARTS WITH id) OR
                          ANY(id IN $article_ids WHERE id STARTS WITH a.id)
                    OPTIONAL MATCH (p:Precedent)-[:REFERENCES_ARTICLE]->(a)
                    RETURN
                        a.id AS id, 'Article' AS type, a.text AS text, 1.0 AS score,
                        collect(DISTINCT {id: p.id, name: p.name})[..3] AS related_precedents
                    """
                    direct_article_results = session.run(direct_article_query, article_ids=query_articles)
                    count = 0
                    for record in direct_article_results:
                        key = ("Article", record["id"])
                        if key not in all_results_map:
                            all_results_map[key] = dict(record)
                            count += 1
                    if count > 0:
                        print(f"Found {count} directly referenced articles (with related precedents).")
                except Exception as e:
                    print(f"Warning: Error searching directly referenced articles: {e}")

            # == 2. 키워드 기반 판례 검색 ==
            if query_keywords:
                try:
                    keyword_query = """
                    MATCH (k:Keyword)<-[:HAS_KEYWORD]-(p:Precedent)
                    WHERE k.text IN $keywords
                    WITH p, count(DISTINCT k) AS keyword_match_count WHERE keyword_match_count > 0
                    OPTIONAL MATCH (p)-[:REFERENCES_ARTICLE]->(a:Article)
                    OPTIONAL MATCH (p)-[:HAS_KEYWORD]->(pk:Keyword)
                    WITH p, keyword_match_count,
                         collect(DISTINCT a.id)[..3] AS referenced_articles,
                         collect(DISTINCT pk.text)[..5] AS keywords
                    RETURN
                        p.id AS id, 'Precedent' AS type, p.name AS name,
                        coalesce(p.full_summary, p.judgment_summary) AS text,
                        keyword_match_count / $keyword_count AS score,
                        referenced_articles, keywords
                    ORDER BY score DESC LIMIT $limit
                    """
                    keyword_results = session.run(
                        keyword_query, keywords=query_keywords,
                        keyword_count=len(query_keywords), limit=top_k
                    )
                    count = 0
                    for record in keyword_results:
                        key = ("Precedent", record["id"])
                        if key not in all_results_map and record["score"] > 0.1:
                            all_results_map[key] = dict(record)
                            count += 1
                    if count > 0:
                         print(f"Found {count} precedents through keyword matching (with related info).")
                except Exception as e:
                    if "Neo.ClientNotification.Statement.UnknownRelationshipTypeWarning" in str(e) and "HAS_KEYWORD" in str(e):
                         print("Warning: 'HAS_KEYWORD' relationship type not found during keyword search. Skipping keyword search.")
                    else:
                         print(f"Warning: Error in keyword search: {e}")

            # === 수정: existing_keys 포맷 변경 (리스트의 리스트로) ===
            # Cypher에서 리스트 비교를 위해 ['Type', 'id'] 형식 사용
            existing_keys_list = [[key[0], key[1]] for key in all_results_map.keys()]

            # == 3. 벡터 검색 (Article) + GraphRAG 이웃 정보 ==
            try:
                article_vector_query = """
                CALL db.index.vector.queryNodes('article_embedding', $limit, $embedding)
                YIELD node AS article, score AS vector_score
                // === 수정된 WHERE 절 ===
                WHERE NOT ['Article', article.id] IN $existing_keys_list
                OPTIONAL MATCH (p:Precedent)-[:REFERENCES_ARTICLE]->(article)
                RETURN
                    article.id AS id, 'Article' AS type, article.text AS text,
                    vector_score AS score,
                    collect(DISTINCT {id: p.id, name: p.name})[..3] AS related_precedents
                LIMIT $limit
                """
                article_results = session.run(
                    article_vector_query,
                    embedding=query_embedding,
                    existing_keys_list=existing_keys_list, # 수정된 파라미터 이름 및 값
                    limit=max(1, top_k // 2) # limit은 1 이상이어야 함
                )
                count = 0
                new_keys_found = [] # 새로 찾은 키 저장
                for record in article_results:
                    key = ("Article", record["id"])
                    if key not in all_results_map:
                        all_results_map[key] = dict(record)
                        new_keys_found.append(['Article', record["id"]])
                        count += 1
                    elif 'related_precedents' not in all_results_map[key] or not all_results_map[key]['related_precedents']:
                         all_results_map[key]['related_precedents'] = record['related_precedents']
                if count > 0:
                    print(f"Found {count} new articles through vector search (with related precedents).")
                    # 다음 검색을 위해 existing_keys_list 업데이트
                    existing_keys_list.extend(new_keys_found)

            except Exception as e_vec1:
                # 오류 메시지에 SyntaxError 포함 시 구문 오류로 판단
                if "SyntaxError" in str(e_vec1):
                     print(f"Critical Error: Vector search syntax error for articles: {e_vec1}")
                     print("Skipping Article vector search due to syntax error.")
                else:
                     print(f"Warning: Vector search for articles failed: {e_vec1}")

            # == 4. 벡터 검색 (Precedent) + GraphRAG 이웃 정보 ==
            try:
                precedent_vector_query = """
                CALL db.index.vector.queryNodes('precedent_embedding', $limit, $embedding)
                YIELD node AS precedent, score AS vector_score
                // === 수정된 WHERE 절 ===
                WHERE NOT ['Precedent', precedent.id] IN $existing_keys_list
                OPTIONAL MATCH (precedent)-[:REFERENCES_ARTICLE]->(a:Article)
                OPTIONAL MATCH (precedent)-[:HAS_KEYWORD]->(k:Keyword)
                OPTIONAL MATCH (precedent)-[:REFERENCES_CASE]->(ref_p:Precedent)
                OPTIONAL MATCH (citing_p:Precedent)-[:REFERENCES_CASE]->(precedent)
                WITH precedent, vector_score,
                     collect(DISTINCT a.id)[..3] AS referenced_articles,
                     collect(DISTINCT k.text)[..5] AS keywords,
                     collect(DISTINCT {id: ref_p.id, name: ref_p.name})[..2] AS referenced_precedents,
                     collect(DISTINCT {id: citing_p.id, name: citing_p.name})[..2] AS citing_precedents
                RETURN
                    precedent.id AS id, 'Precedent' AS type, precedent.name AS name,
                    coalesce(precedent.full_summary, precedent.judgment_summary) AS text,
                    vector_score AS score,
                    referenced_articles, keywords, referenced_precedents, citing_precedents
                LIMIT $limit
                """
                precedent_results = session.run(
                    precedent_vector_query,
                    embedding=query_embedding,
                    existing_keys_list=existing_keys_list, # 수정된 파라미터 이름 및 값
                    limit=max(1, top_k // 2) # limit은 1 이상이어야 함
                )
                count = 0
                for record in precedent_results:
                    key = ("Precedent", record["id"])
                    if key not in all_results_map:
                        all_results_map[key] = dict(record)
                        count += 1
                    else:
                        if record['score'] > all_results_map[key].get('score', -1):
                             all_results_map[key].update(dict(record))
                        else:
                            for info_key in ['referenced_articles', 'keywords', 'referenced_precedents', 'citing_precedents']:
                                if info_key not in all_results_map[key] or not all_results_map[key][info_key]:
                                     all_results_map[key][info_key] = record[info_key]
                if count > 0:
                    print(f"Found {count} new precedents through vector search (with related info).")

            except Exception as e_vec2:
                 if "SyntaxError" in str(e_vec2):
                     print(f"Critical Error: Vector search syntax error for precedents: {e_vec2}")
                     print("Skipping Precedent vector search due to syntax error.")
                 else:
                     print(f"Warning: Vector search for precedents failed: {e_vec2}")

            # === 최종 결과 통합 및 정렬 ===
            final_results = list(all_results_map.values())
            final_results.sort(key=lambda x: x.get("score", -1), reverse=True)
            results = final_results[:top_k]

            # 텍스트 미리보기 추가
            for res in results:
                text_preview = res.get("text", "")
                res["text_preview"] = text_preview[:250] + "..." if len(text_preview) > 250 else text_preview

    except Exception as e:
        print(f"Error during graph retrieval for query '{query_text[:50]}...': {e}")
        results = []

    print(f"Retrieved {len(results)} final context items.")
    return results

def process_text_for_context(text: str, max_len: int = 800) -> str:
    """주어진 텍스트를 단순 길이 제한으로 처리"""
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    else:
        # 단순 자르기 + 말줄임표
        return text[:max_len] + "..."

def build_optimized_context(search_results: List[Dict[str, Any]], question: str) -> str:
    """검색 결과와 질문을 바탕으로 LLM에 제공할 최적화된 컨텍스트 구성 (GraphRAG 정보 활용)"""
    if not search_results:
        return "관련된 법 조항이나 판례 정보를 찾지 못했습니다. 질문 내용을 바탕으로 직접 답변해주세요."

    context_parts = ["### 참고 자료 (질문과 관련성이 높은 순서대로 정렬됨) ###"]
    processed_items = [] # (score, context_string) 저장

    for result in search_results:
        result_type = result.get("type", "Unknown")
        result_id = result.get("id", "N/A")
        score = result.get("score", 0.0)
        original_text = result.get("text", "")
        processed_text = process_text_for_context(original_text, max_len=800 if result_type == 'Article' else 1000)

        if not processed_text:
            continue

        related_info_parts = [] # 관련 정보 문자열 조각들

        if result_type == "Article":
            header = f"【법조항: {result_id} (관련성 점수: {score:.3f})】"
            # 관련 판례 정보
            related_precedents = result.get("related_precedents", [])
            if related_precedents:
                p_info = [f"{p.get('name', p.get('id', 'Unknown'))}" for p in related_precedents]
                related_info_parts.append(f"[관련 판례: {', '.join(p_info)}]")
            # (필요시 다른 관련 정보 추가)

        elif result_type == "Precedent":
            name = result.get("name", "")
            header = f"【판례: {name or result_id} (관련성 점수: {score:.3f})】"
            # 참조 법조항
            ref_articles = result.get("referenced_articles", [])
            if ref_articles:
                related_info_parts.append(f"[참조 법조항: {', '.join(ref_articles)}]")
            # 키워드
            keywords = result.get("keywords", [])
            if keywords:
                related_info_parts.append(f"[주요 키워드: {', '.join(keywords)}]")
            # 참조 판례
            ref_precedents = result.get("referenced_precedents", [])
            if ref_precedents:
                ref_p_info = [f"{p.get('name', p.get('id', 'Unknown'))}" for p in ref_precedents]
                related_info_parts.append(f"[참조 판례: {', '.join(ref_p_info)}]")
            # 인용 판례
            citing_precedents = result.get("citing_precedents", [])
            if citing_precedents:
                cite_p_info = [f"{p.get('name', p.get('id', 'Unknown'))}" for p in citing_precedents]
                related_info_parts.append(f"[인용된 판례: {', '.join(cite_p_info)}]")

        else: # Unknown type
            header = f"【정보: {result_id} (관련성 점수: {score:.3f})】"

        # 최종 컨텍스트 조립
        full_context = f"{header}\n{processed_text}"
        if related_info_parts:
            full_context += "\n" + " ".join(related_info_parts)

        processed_items.append((score, full_context))

    # 점수 순으로 정렬 (이미 retrieve_context_from_graph에서 정렬됨)
    # processed_items.sort(reverse=True) # 필요시 재정렬

    # 컨텍스트 길이 제한 적용하여 최종 문자열 생성
    max_total_context_len = 3800 # 컨텍스트 길이 약간 늘림
    current_total_len = len(context_parts[0])
    final_context_count = 0

    for _, item_text in processed_items:
        if current_total_len + len(item_text) + 2 < max_total_context_len:
            context_parts.append(item_text)
            current_total_len += len(item_text) + 2
            final_context_count += 1
        else:
            break # 길이 제한 도달

    context_parts.append(f"\n### 지침: 위의 {final_context_count}개 참고 자료(법조항, 판례 및 관련 정보 포함)를 바탕으로 질문에 가장 적합한 답변을 선택하세요. ###")

    return "\n\n".join(context_parts)

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
    # articles = {} # 임베딩 이미 했으면 주석해제
    # precedents = [] # 임베딩 이미 했으면 주석해제
    articles = load_articles_from_pdf(config["pdf_path"]) # 임베딩 이미 했으면 주석
    precedents = load_precedents_from_json(config["precedent_dir"], sample_size=3000) # 임베딩 이미 했으면 주석
    
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
    create_graph_nodes_and_relationships(neo4j_driver, articles, precedents, embedding_model) # 임베딩 이미 했으면 주석

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