# 셀1: import
import os
import json
import re
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from neo4j import GraphDatabase
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from openai import OpenAI
import csv
import pandas as pd
import json
import os
import time
import re
from datetime import datetime
from tqdm.notebook import tqdm
import time

# 셀2: 이것저것 로드
load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Embedding model 설정
embedding_model = OpenAIEmbeddings(model='text-embedding-3-small', api_key=OPENAI_API_KEY)
embedding_dimension = 1536 # text-embedding-3-small 차원

# Data path
pdf_path = './dataset/criminal-law.pdf'
precedent_dir = './dataset/precedent_label/'

# 셀3: Load PDF
loader = PyPDFLoader(pdf_path)
# pages = loader.load()[2:] # 첫 두 페이지 생략 (표지랑 목차)
pages = loader.load() # 아님말고
full_text = "\n".join(page.page_content for page in pages)

# 전체 텍스트에서 모든 조항 시작 위치 찾기
article_pattern = r'제\d+조(?:의\d+)?(?:\s*\(.+?\))?'
matches = list(re.finditer(article_pattern, full_text))

articles = {}
for i in range(len(matches)):
  current_match = matches[i]
  current_article_id = current_match.group(0).strip() # 현재 조항 ID
  
  # 현재 조항 시작 위치
  start_pos = current_match.start()
  
  # 다음 조항 시작 위치 (없으면 텍스트 끝까지)
  end_pos = matches[i+1].start() if i < len(matches)-1 else len(full_text)
  
  # 현재 조항의 전체 내용 (ID 포함)
  article_text = full_text[start_pos:end_pos].strip()
  
  # 저장 (ID는 조항 번호만)
  articles[current_article_id] = article_text
  
print(f"Processed {len(articles)} article from PDF")

# 예시 출력
if articles:
  article_ids = list(articles.keys())
  
  print("\n--- 처음 5개 조항 ---")
  for i in range(min(5, len(article_ids))):
    article_id = article_ids[i]
    content = articles[article_id]
    print(f"\n--- Article: {article_id} ---")
    print(content[:200] + "..." if len(content) > 200 else content)
    
  print("\n--- 마지막 5개 조항 ---")
  for i in range(max(0, len(article_ids)-10), len(article_ids)):
    article_id = article_ids[i]
    content = articles[article_id]
    print(f"\n--- Article: {article_id} ---")
    print(content[:200] + "..." if len(content) > 200 else content)
    
# 셀4: Load precedent JSON files (판례 불러오기)
precedents = []
for filename in os.listdir(precedent_dir):
    if filename.endswith(".json"):
        filepath = os.path.join(precedent_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 기존에 라벨링 되어있었음
                precedent_info = {
                    "case_id": data.get("info", {}).get("caseNoID", filename.replace(".json", "")), # 사건번호 (없으면 파일명 사용)
                    "case_name": data.get("info", {}).get("caseNm"), # 사건명
                    "judgment_summary": data.get("jdgmn"), # 판결 요약
                    "full_summary": " ".join([s.get("summ_contxt", "") for s in data.get("Summary", [])]), # 전체 요약 텍스트
                    "keywords": [kw.get("keyword") for kw in data.get("keyword_tagg", []) if kw.get("keyword")], # 키워드 목록
                    "referenced_rules": data.get("Reference_info", {}).get("reference_rules", "").split(',') if data.get("Reference_info", {}).get("reference_rules") else [], # 참조 법조항
                    "referenced_cases": data.get("Reference_info", {}).get("reference_court_case", "").split(',') if data.get("Reference_info", {}).get("reference_court_case") else [], # 참조 판례
                }
                # 참조 법조항 정제 (조항 번호만)
                cleaned_rules = []
                rule_pattern = re.compile(r'제\d+조(?:의\d+)?') # 패턴 찾기: "제X조" or "제X조의Y"
                for rule in precedent_info["referenced_rules"]:
                    # 각 규칙 문자열에서 모든 일치 항목 찾기
                    matches = rule_pattern.findall(rule.strip())
                    cleaned_rules.extend(matches)
                precedent_info["referenced_rules"] = list(set(cleaned_rules)) # 중복 제거하여 고유한 조항 번호만 유지

                precedents.append(precedent_info)
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON from {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")


print(f"Loaded {len(precedents)} precedents.")
# 예시 출력
if precedents:
    print("\n--- Example Precedent ---")
    print(json.dumps(precedents[0], indent=2, ensure_ascii=False))
    
# 셀5: 로드된 판례 중 무작위로 1,000개만 선택 (시간 문제 때문에...)
random.seed(42)  # 재현성을 위한 시드 설정

# 전체 판례 수 저장
total_precedents = len(precedents)

# 무작위로 1,000개 선택 (또는 전체 판례 수가 1,000개보다 적다면 모두 선택)
sample_size = min(1000, total_precedents)
precedents = random.sample(precedents, sample_size)

# 셀6: Neo4j 데이터베이스에 연결
try:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    driver.verify_connectivity()  # 연결 확인
    print("Successfully connected to Neo4j.")
except Exception as e:
    print(f"Failed to connect to Neo4j: {e}")
    # 연결 실패 시 실행 중단
    raise

# 빠른 조회와 임베딩 검색을 위한 제약조건과 인덱스 생성 함수
def setup_neo4j(driver, dimension):
    with driver.session(database="neo4j") as session:
        # 고유성을 위한 제약조건 설정
        session.run("CREATE CONSTRAINT article_id IF NOT EXISTS FOR (a:Article) REQUIRE a.id IS UNIQUE")  # 법조항 ID 고유성 제약
        session.run("CREATE CONSTRAINT precedent_id IF NOT EXISTS FOR (p:Precedent) REQUIRE p.id IS UNIQUE")  # 판례 ID 고유성 제약
        session.run("CREATE CONSTRAINT keyword_text IF NOT EXISTS FOR (k:Keyword) REQUIRE k.text IS UNIQUE")  # 키워드 텍스트 고유성 제약

        # 법조항(Article)에 대한 벡터 인덱스 생성
        try:
            session.run(
                "CREATE VECTOR INDEX article_embedding IF NOT EXISTS "  # 존재하지 않는 경우에만 생성
                "FOR (a:Article) ON (a.embedding) "  # Article 노드의 embedding 속성에 대한 인덱스
                f"OPTIONS {{indexConfig: {{`vector.dimensions`: {dimension}, `vector.similarity_function`: 'cosine'}}}}"  # 벡터 차원 및 유사도 함수 설정
            )
            print("Article vector index created or already exists.")
        except Exception as e:
            print(f"Error creating Article vector index (may require Neo4j 5.11+ and APOC): {e}")  # Neo4j 버전 문제일 수 있음
            print("Continuing without vector index creation for Article.")  # 인덱스 없이 계속 진행

        # 판례(Precedent)에 대한 벡터 인덱스 생성
        try:
            session.run(
                "CREATE VECTOR INDEX precedent_embedding IF NOT EXISTS "  # 존재하지 않는 경우에만 생성
                "FOR (p:Precedent) ON (p.embedding) "  # Precedent 노드의 embedding 속성에 대한 인덱스
                f"OPTIONS {{indexConfig: {{`vector.dimensions`: {dimension}, `vector.similarity_function`: 'cosine'}}}}"  # 벡터 차원 및 유사도 함수 설정
            )
            print("Precedent vector index created or already exists.")
        except Exception as e:
            print(f"Error creating Precedent vector index (may require Neo4j 5.11+ and APOC): {e}")  # Neo4j 버전 문제일 수 있음
            print("Continuing without vector index creation for Precedent.")  # 인덱스 없이 계속 진행

        # 인덱스가 활성화될 때까지 대기 (중요!)
        print("Waiting for indexes to populate...")
        session.run("CALL db.awaitIndexes(300)")  # 최대 300초(5분) 동안 대기
        print("Indexes should be online.")  # 인덱스 활성화 완료


setup_neo4j(driver, embedding_dimension)  # 설정 함수 호출, embedding_dimension은 임베딩 벡터의 차원 크기

# 셀7: 법조항 노드 생성 및 임베딩 생성/저장 (이미 했으면 생략가능)
def create_article_nodes(driver, articles_dict, embed_model):
    print(f"Creating {len(articles_dict)} Article nodes...")  # 생성할 법조항 노드 수 출력
    count = 0
    start_time = time.time()  # 실행 시간 측정 시작
    with driver.session(database="neo4j") as session:
        for article_id, content in articles_dict.items():
            if not content:  # 내용이 비어있는 경우 건너뛰기
                print(f"Skipping article {article_id} due to empty content.")
                continue
            try:
                # 텍스트에 대한 임베딩 생성
                embedding = embed_model.embed_query(content)

                # Neo4j에 노드 생성
                session.run(
                    """
                    MERGE (a:Article {id: $article_id})  # 해당 ID의 법조항이 있으면 찾고, 없으면 생성
                    SET a.text = $content,               # 법조항 텍스트 설정
                        a.embedding = $embedding         # 임베딩 벡터 설정
                    """,
                    article_id=article_id,
                    content=content,
                    embedding=embedding
                )
                count += 1
                if count % 50 == 0:  # 50개마다 진행상황 출력
                    print(f"  Processed {count}/{len(articles_dict)} articles...")
            except Exception as e:
                print(f"Error processing article {article_id}: {e}")  # 오류 발생 시 메시지 출력

    end_time = time.time()  # 실행 시간 측정 종료
    print(f"Finished creating {count} Article nodes in {end_time - start_time:.2f} seconds.")  # 총 처리 시간 출력

create_article_nodes(driver, articles, embedding_model)  # 함수 실행: 법조항 노드 생성 및 임베딩 저장

# 셀8: 판례 노드, 키워드 노드 생성 및 관계 설정 (이미 했으면 생략가능)
def create_precedent_nodes_and_relationships(driver, precedents_list, embed_model):
    print(f"Creating {len(precedents_list)} Precedent nodes and relationships...")  # 생성할 판례 노드 수 출력
    count = 0
    start_time = time.time()  # 실행 시간 측정 시작
    with driver.session(database="neo4j") as session:
        for precedent in precedents_list:
            # 임베딩에 사용할 텍스트: 전체 요약이 있으면 사용, 없으면 판결 요약 사용
            text_to_embed = precedent.get("full_summary") or precedent.get("judgment_summary")
            if not text_to_embed:
                print(f"Skipping precedent {precedent.get('case_id')} due to empty summary.")  # 요약이 없는 경우 건너뛰기
                continue

            try:
                # 텍스트 임베딩 생성
                embedding = embed_model.embed_query(text_to_embed)

                # 판례 노드 생성
                session.run(
                    """
                    MERGE (p:Precedent {id: $case_id})  # 해당 ID의 판례가 있으면 찾고, 없으면 생성
                    SET p.name = $case_name,            # 판례명 설정
                        p.judgment_summary = $judgment_summary,  # 판결 요약 설정
                        p.full_summary = $full_summary,          # 전체 요약 설정
                        p.embedding = $embedding         # 임베딩 벡터 설정
                    """,
                    case_id=precedent["case_id"],
                    case_name=precedent["case_name"],
                    judgment_summary=precedent["judgment_summary"],
                    full_summary=precedent["full_summary"],
                    embedding=embedding
                )

                # 키워드 노드 생성 및 판례와의 관계 설정
                for keyword_text in precedent["keywords"]:
                    session.run(
                        """
                        MERGE (k:Keyword {text: $keyword_text})  # 키워드 노드 생성 또는 찾기
                        WITH k
                        MATCH (p:Precedent {id: $case_id})       # 판례 노드 찾기
                        MERGE (p)-[:HAS_KEYWORD]->(k)            # 판례와 키워드 간 관계 생성
                        """,
                        keyword_text=keyword_text,
                        case_id=precedent["case_id"]
                    )

                # 참조된 법조항과의 관계 생성
                # 참고: 앞서 추출한 정제된 법조항 ID를 사용합니다
                # "제X조" 형식을 기반으로 매칭합니다.
                for article_id_ref in precedent["referenced_rules"]:
                     # 참조된 ID로 시작하는 법조항 노드 찾기(예: "제21조"는 "제21조(정당방위)"와 매칭됨)
                     # 정확한 제목이 참조에 없는 경우에도 매칭이 가능하도록 덜 정밀한 방식 사용
                    session.run(
                        """
                        MATCH (p:Precedent {id: $case_id})         # 판례 노드 찾기
                        MATCH (a:Article)                          # 모든 법조항 노드 찾기
                        WHERE a.id STARTS WITH $article_id_ref     # 특정 ID로 시작하는 법조항만 필터링
                        MERGE (p)-[:REFERENCES_ARTICLE]->(a)       # 판례가 법조항을 참조하는 관계 생성
                        """,
                        case_id=precedent["case_id"],
                        article_id_ref=article_id_ref  # 추출된 "제X조" 사용
                    )

                # 선택사항: 다른 참조된 판례와의 관계 생성 (필요한 경우)
                # for ref_case_id in precedent["referenced_cases"]:
                #    session.run(...) # MERGE (p)-[:REFERENCES_CASE]->(other_p:Precedent {id: ref_case_id})

                count += 1
                if count % 100 == 0:  # 100개마다 진행상황 출력
                    print(f"  Processed {count}/{len(precedents_list)} precedents...")

            except Exception as e:
                print(f"Error processing precedent {precedent.get('case_id')}: {e}")  # 오류 발생 시 메시지 출력

    end_time = time.time()  # 실행 시간 측정 종료
    print(f"Finished creating {count} Precedent nodes and relationships in {end_time - start_time:.2f} seconds.")  # 총 처리 시간 출력


create_precedent_nodes_and_relationships(driver, precedents, embedding_model)  # 함수 실행: 판례 노드 생성 및 관계 설정

# 작업 완료 후 드라이버 연결 종료
# driver.close()  # 다음 단계에서 쿼리를 위해 연결 상태 유지

# 셀9: graph rag 함수
def graph_enhanced_rag(driver, query_text, embed_model, top_k=8):
    # print(f"\n--- 그래프 기반 검색 실행: '{query_text}' ---")
    start_time = time.time()

    # 임베딩 생성
    query_embedding = embed_model.embed_query(query_text)
    
    # 키워드 추출
    keywords = [w for w in re.findall(r'\w+', query_text) if len(w) > 1]
    
    results = []
    with driver.session(database="neo4j") as session:
        try:
            # 그래프 구조를 활용한 검색
            cypher_query = """
            // 1. 벡터 검색으로 시작 법조항 찾기
            CALL db.index.vector.queryNodes('article_embedding', 15, $query_embedding) 
            YIELD node as article, score as article_score
            
            // 2. 해당 법조항과 연결된 판례와 키워드 찾기
            OPTIONAL MATCH (precedent:Precedent)-[:REFERENCES_ARTICLE]->(article)
            OPTIONAL MATCH (precedent)-[:HAS_KEYWORD]->(keyword:Keyword)
            
            // 3. 결과 집계 및 점수 계산
            WITH article, article_score, precedent, 
                 collect(DISTINCT keyword.text) as keywords,
                 count(precedent) as precedent_count
            
            // 법조항 점수 = 벡터 점수 + 판례 인용 수에 따른 보너스
            WITH article, article_score + (precedent_count * 0.01) as final_score,
                 precedent_count, keywords
            
            RETURN article.id as id, 
                   'Article' as type, 
                   article.text as text, 
                   final_score as score,
                   precedent_count,
                   keywords
            ORDER BY final_score DESC
            LIMIT $article_limit
            """
            
            # 법조항 검색
            article_results = session.run(
                cypher_query,
                query_embedding=query_embedding,
                article_limit=top_k
            )
            
            for record in article_results:
                results.append({
                    "type": record["type"],
                    "id": record["id"],
                    "score": record["score"],
                    "text": record["text"][:300] + "..." if len(record["text"]) > 300 else record["text"],
                    "precedent_count": record["precedent_count"],
                    "related_keywords": record["keywords"]
                })
            
            # 관련 판례 검색
            for article_result in results[:3]:  # 상위 3개 법조항에 대해서만
                if article_result["type"] == "Article":
                    precedent_query = """
                    // 1. 특정 법조항을 참조하는 판례 찾기
                    MATCH (precedent:Precedent)-[:REFERENCES_ARTICLE]->(article:Article)
                    WHERE article.id STARTS WITH $article_id
                    
                    // 2. 해당 판례와 키워드
                    OPTIONAL MATCH (precedent)-[:HAS_KEYWORD]->(keyword:Keyword)
                    
                    // 3. 벡터 유사도 계산
                    CALL db.index.vector.queryNodes('precedent_embedding', 20, $query_embedding) 
                    YIELD node as vector_node, score as vector_score
                    WHERE precedent = vector_node
                    
                    // 4. 검색어와 관련된 키워드가 있는지 확인하여 보너스 점수
                    WITH precedent, vector_score, 
                         collect(DISTINCT keyword.text) as keywords,
                         sum(CASE WHEN $query_keywords IS NULL THEN 0
                              WHEN any(k IN $query_keywords WHERE keyword.text CONTAINS k) 
                              THEN 0.05 ELSE 0 END) as keyword_bonus
                    
                    // 5. 다른 법조항도 참조하는지 확인
                    MATCH (precedent)-[:REFERENCES_ARTICLE]->(ref_article:Article)
                    
                    // 6. 최종 결과 반환
                    RETURN precedent.id as id,
                           'Precedent' as type,
                           precedent.name as name,
                           precedent.full_summary as text,
                           vector_score + keyword_bonus as score,
                           keywords,
                           collect(DISTINCT ref_article.id) as referenced_articles
                    ORDER BY score DESC
                    LIMIT 2
                    """
                    
                    precedent_results = session.run(
                        precedent_query,
                        article_id=article_result["id"],
                        query_embedding=query_embedding,
                        query_keywords=keywords
                    )
                    
                    for record in precedent_results:
                        # 중복 제거
                        if not any(r["type"] == "Precedent" and r["id"] == record["id"] for r in results):
                            results.append({
                                "type": record["type"],
                                "id": record["id"],
                                "name": record["name"],
                                "score": record["score"],
                                "text": record["text"][:300] + "..." if len(record["text"]) > 300 else record["text"],
                                "keywords": record["keywords"],
                                "referenced_articles": record["referenced_articles"]
                            })
        
        except Exception as e:
            print(f"그래프 검색 오류: {e}")
            # 백업: 기본 벡터 검색
            try:
                # Article 검색
                article_res = session.run(
                    """
                    CALL db.index.vector.queryNodes('article_embedding', $top_k, $query_embedding) 
                    YIELD node, score
                    RETURN node.id AS id, 'Article' as type, node.text AS text, score
                    """,
                    top_k=top_k,
                    query_embedding=query_embedding
                )
                
                for record in article_res:
                    results.append({
                        "type": record["type"],
                        "id": record["id"],
                        "score": record["score"],
                        "text": record["text"][:300] + "..." if len(record["text"]) > 300 else record["text"]
                    })
                
                # Precedent 검색
                precedent_res = session.run(
                    """
                    CALL db.index.vector.queryNodes('precedent_embedding', $top_k, $query_embedding) 
                    YIELD node, score
                    MATCH (node)-[:REFERENCES_ARTICLE]->(a:Article)
                    OPTIONAL MATCH (node)-[:HAS_KEYWORD]->(k:Keyword)
                    RETURN node.id AS id, 'Precedent' as type, 
                           node.name AS name, node.full_summary AS text, 
                           score,
                           collect(DISTINCT a.id) as referenced_articles,
                           collect(DISTINCT k.text) as keywords
                    """,
                    top_k=top_k,
                    query_embedding=query_embedding
                )
                
                for record in precedent_res:
                    results.append({
                        "type": record["type"],
                        "id": record["id"],
                        "name": record["name"],
                        "score": record["score"],
                        "text": record["text"][:300] + "..." if len(record["text"]) > 300 else record["text"],
                        "referenced_articles": record["referenced_articles"],
                        "keywords": record["keywords"]
                    })
            except Exception as e2:
                print(f"백업 검색 오류: {e2}")
    
    end_time = time.time()
    # print(f"검색 완료: {end_time - start_time:.2f}초 소요")

    # 결과를 스코어로 정렬
    results.sort(key=lambda x: x["score"], reverse=True)

    # print("\n--- 검색 결과 ---")
    # for i, res in enumerate(results[:top_k]):
    #     print(f"{i+1}. 유형: {res['type']}, ID: {res['id']}, 스코어: {res['score']:.4f}")
    #     if res['type'] == 'Precedent':
    #         print(f"   이름: {res.get('name')}")
    #         print(f"   키워드: {res.get('keywords')}")
    #         print(f"   참조 법조항: {res.get('referenced_articles')}")
    #     elif res['type'] == 'Article':
    #         print(f"   관련 판례 수: {res.get('precedent_count', 0)}")
    #         print(f"   관련 키워드: {res.get('related_keywords')}")
    #     print(f"   미리보기: {res['text']}")
    #     print("-" * 20)

    return results[:top_k]
  
# 셀10: 테스트 쿼리
search_function = graph_enhanced_rag 

# 테스트 쿼리
query = "정당방위의 요건은 무엇인가?"
retrieved_context = search_function(driver, query, embedding_model, top_k=8)

# 드라이버 연결 종료
driver.close()
print("\nNeo4j 드라이버 연결 종료")

# 셀 11
import re
from collections import Counter

def highlight_relevant_parts(text, question):
    """법조항 텍스트에서 질문과 관련된 부분을 추출하고 강조합니다."""
    # 질문에서 주요 키워드 추출 (개선된 함수 사용)
    keywords = extract_legal_keywords(question)
    
    # 전체 텍스트가 짧으면 그대로 반환 (길이 기준 확장)
    if len(text) < 500:
        return text
    
    # 문장 단위로 분리 (개선된 분리 패턴)
    sentences = re.split(r'(?<=[.!?])\s+|(?<=\n)', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # 관련 점수 계산 (개선된 점수 부여 시스템)
    scored_sentences = []
    for i, sentence in enumerate(sentences):
        # 기본 점수
        score = 0
        
        # 키워드 매칭 점수 (개선됨)
        for keyword in keywords:
            if keyword in sentence:
                # 중요 법률 용어는 더 높은 가중치
                if keyword in ["구성요건", "위법성", "책임", "정당방위", "긴급피난", 
                             "고의", "과실", "미수", "예비", "음모", "공범"]:
                    score += 3
                else:
                    score += 1
        
        # 법조항 번호 포함 문장은 높은 가중치
        if re.search(r'제\d+조', sentence):
            score += 5
        
        # 법률적 중요 문장 패턴에 가중치
        if any(term in sentence for term in ["~으로 한다", "~라 함은", "~을 말한다", "다만", "단,", "제외한다"]):
            score += 3
        
        # 위치 가중치 (개선됨)
        if i == 0:  # 첫 문장 (제목이나 조항 번호일 가능성)
            score += 5
        elif i == len(sentences) - 1:  # 마지막 문장 (결론일 가능성)
            score += 2
        elif i <= 2:  # 앞부분 문장들 (정의나 개요일 가능성)
            score += 1
            
        scored_sentences.append((sentence, score, i))  # 원래 순서 저장
    
    # 점수 및 원래 순서 기준 정렬
    # 상위 70% 이상 점수 문장 또는 최소 8문장 선택 (너무 적은 문장만 선택되는 것 방지)
    min_score = max(1, sorted([score for _, score, _ in scored_sentences], reverse=True)[0] * 0.3)
    relevant_sentences = [(s, score, i) for s, score, i in scored_sentences if score >= min_score]
    min_sentences = min(8, len(sentences))
    
    if len(relevant_sentences) < min_sentences:
        # 점수순으로 정렬하여 상위 min_sentences개 선택
        relevant_sentences = sorted(scored_sentences, key=lambda x: x[1], reverse=True)[:min_sentences]
    
    # 원래 순서로 재정렬
    ordered_sentences = sorted(relevant_sentences, key=lambda x: x[2])
    
    # 법조항 번호와 제목은 항상 포함
    if len(ordered_sentences) > 0 and (0, sentences[0], 0) not in ordered_sentences:
        ordered_sentences.insert(0, (sentences[0], 0, 0))
    
    result = " ".join([s for s, _, _ in ordered_sentences])
    
    # 결과가 원본의 30% 미만이라면 원본 반환 (너무 많은 내용 손실 방지)
    if len(result) < len(text) * 0.3:
        return text
        
    return result

def summarize_precedent(text, question):
    """판례 텍스트에서 질문과 관련된 핵심 내용을 요약합니다."""
    # 질문에서 주요 키워드 추출
    keywords = extract_legal_keywords(question)
    
    # 텍스트가 이미 짧으면 그대로 반환 (기준 확대)
    if len(text) < 300:
        return text
    
    # 문장 단위로 분리 (개선된 분리)
    sentences = re.split(r'(?<=[.!?])\s+|(?<=\n)', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # 관련 점수 계산 (개선된 점수 시스템)
    scored_sentences = []
    for i, sentence in enumerate(sentences):
        score = 0
        
        # 키워드 포함 여부 (더 정교한 매칭)
        for keyword in keywords:
            if keyword in sentence:
                score += 2
                # 문장에서 키워드가 중앙에 있으면 추가 점수
                position = sentence.find(keyword) / len(sentence)
                if 0.2 <= position <= 0.8:
                    score += 1
        
        # 확장된 법률 용어 포함 여부
        legal_terms = [
            "판시", "판결", "법리", "해석", "적용", "요건", "효과", "정당", "위법", "책임",
            "대법원", "판례", "법원", "결정", "원심", "상고", "청구", "기각", "구성요건",
            "피고인", "원고", "법률", "양형", "해당", "인정"
        ]
        
        term_count = sum(1 for term in legal_terms if term in sentence)
        score += min(term_count, 3)  # 최대 3점까지만 추가
        
        # 결론 표현 포함 시 높은 점수
        conclusion_terms = ["따라서", "그러므로", "결론적으로", "이유로", "판단한다"]
        for term in conclusion_terms:
            if term in sentence:
                score += 3
                break
        
        # 위치 보너스 (더 정교함)
        if i == 0:
            score += 3  # 첫 문장 (사건 개요일 가능성 높음)
        elif i == len(sentences) - 1:
            score += 3  # 마지막 문장 (결론일 가능성 높음)
        elif 0 < i <= 2:
            score += 1  # 초반 문장
        elif i >= len(sentences) - 3:
            score += 1  # 후반 문장
            
        scored_sentences.append((sentence, score, i))
    
    # 상위 문장 선택 (더 많은 문장 포함)
    top_count = max(5, min(len(sentences) // 3, 10))  # 전체의 1/3 또는 최소 5문장, 최대 10문장
    
    # 점수 기준 정렬 및 상위 문장 선택
    top_sentences = sorted(scored_sentences, key=lambda x: x[1], reverse=True)[:top_count]
    
    # 원래 순서로 재정렬
    ordered_sentences = sorted(top_sentences, key=lambda x: x[2])
    
    result = " ".join([s for s, _, _ in ordered_sentences])
    
    # 결과가 너무 짧으면 더 많은 문장 포함
    if len(result) < len(text) * 0.25:
        top_count = min(len(sentences) // 2, 15)  # 더 많은 문장 선택
        top_sentences = sorted(scored_sentences, key=lambda x: x[1], reverse=True)[:top_count]
        ordered_sentences = sorted(top_sentences, key=lambda x: x[2])
        result = " ".join([s for s, _, _ in ordered_sentences])
    
    return result

def extract_legal_keywords(text):
    """질문에서 법률 관련 주요 키워드를 추출합니다."""
    # 불용어 정의 (확장)
    stopwords = [
        "무엇", "어떤", "어떻게", "언제", "누구", "왜", "어디", "경우", "관하여", "대하여", 
        "은", "는", "이", "가", "을", "를", "에", "의", "와", "과", "로", "으로",
        "있다", "없다", "경우", "때", "것", "등", "수", "그", "이", "저", "그렇게",
        "그런", "이런", "저런", "하는", "다음", "또는", "또한", "그리고", "만약", "만일"
    ]
    
    # 주요 법률 용어 정의 (확장)
    legal_terms = {
        # 형법 기본 원칙
        "고의": 3, "과실": 3, "인과관계": 3, "위법성": 3, "책임": 3, 
        "구성요건": 3, "위법성조각사유": 3, "책임조각사유": 3,
        
        # 정당화 사유
        "정당방위": 3, "긴급피난": 3, "자구행위": 3, "피해자동의": 3, "정당행위": 3, 
        "업무로인한행위": 3, "강요된행위": 3,
        
        # 범죄의 실행 단계
        "미수": 3, "기수": 3, "예비": 3, "음모": 3, "중지": 3, "불능미수": 3,
        
        # 공범 관련
        "공범": 3, "교사": 3, "방조": 3, "공동정범": 3, "간접정범": 3, "종범": 3,
        
        # 형벌 관련
        "형": 2, "징역": 2, "벌금": 2, "집행유예": 2, "선고유예": 2, "누범": 2,
        
        # 법이론
        "법익": 2, "작위": 2, "부작위": 2, "결과범": 2, "거동범": 2, "상당인과관계": 2,
        
        # 기타 형법 용어
        "불법": 2, "과잉방위": 2, "우발적": 2, "착오": 2, "원인에서자유로운행위": 2
    }
    
    # 기본 키워드 추출 (2글자 이상)
    words = re.findall(r'\w{2,}', text)
    keywords = [w for w in words if w not in stopwords]
    
    # 인접한 단어들도 함께 고려 (복합 키워드)
    bigrams = []
    for i in range(len(words) - 1):
        if words[i] not in stopwords or words[i+1] not in stopwords:
            bigram = words[i] + words[i+1]
            if len(bigram) >= 4:  # 4글자 이상 복합어만 고려
                bigrams.append(bigram)
    
    # 법률 용어 가중치 적용 및 동의어 추가
    weighted_keywords = []
    
    # 동의어 매핑
    synonyms = {
        "고의": ["범의", "의도적", "계획적"],
        "과실": ["부주의", "태만", "소홀"],
        "위법성": ["불법성", "위법", "불법"],
        "책임": ["형사책임", "귀책", "비난가능성"],
        "미수": ["미완성", "불성공"],
        "정당방위": ["자기방어", "방어행위"]
    }
    
    for keyword in keywords + bigrams:
        if keyword in legal_terms:
            # 가중치만큼 반복 추가
            weighted_keywords.extend([keyword] * legal_terms[keyword])
            
            # 동의어도 추가
            if keyword in synonyms:
                for synonym in synonyms[keyword]:
                    weighted_keywords.append(synonym)
        else:
            weighted_keywords.append(keyword)
    
    # 키워드 빈도수 계산과 중요도 기반 필터링
    counter = Counter(weighted_keywords)
    min_count = 1 if len(counter) < 10 else 2  # 키워드가 적으면 모두 사용
    
    final_keywords = [k for k, v in counter.items() if v >= min_count]
    
    # 키워드가 너무 적으면 원래 키워드 반환
    if len(final_keywords) < 3:
        return weighted_keywords
    
    return final_keywords

def optimize_context_construction(search_results, question):
    """검색 결과에서 질문에 최적화된 컨텍스트를 구성합니다."""
    # 1. 관련성에 따른 컨텍스트 재배열 (더 많은 결과 포함)
    results_by_type = {"Article": [], "Precedent": []}
    for result in search_results:
        results_by_type[result["type"]].append(result)
    
    # 2. 법조항 요약 및 핵심 추출 (더 많은 법조항 포함)
    article_contexts = []
    for article in results_by_type["Article"][:5]:  # 상위 3개에서 5개로 증가
        # 법조항 원문과 요약 모두 포함하여 정보 손실 방지
        highlighted_text = highlight_relevant_parts(article["text"], question)
        
        # 점수가 높은 법조항은 더 강조
        if article.get("score", 0) > 0.7:  # 높은 유사도 점수
            article_contexts.append(f"【중요 법조항: {article['id']}】\n{highlighted_text}")
        else:
            article_contexts.append(f"【{article['id']}】\n{highlighted_text}")
            
        # 관련 키워드가 있으면 함께 표시
        if article.get("related_keywords") and len(article.get("related_keywords")) > 0:
            keywords_str = ", ".join(article.get("related_keywords")[:5])
            article_contexts[-1] += f"\n[관련 키워드: {keywords_str}]"
    
    # 3. 판례 요약 및 핵심 추출 (더 많은 판례 포함)
    precedent_contexts = []
    for precedent in results_by_type["Precedent"][:3]:  # 상위 2개에서 3개로 증가
        # 판례 요약 처리
        summary = summarize_precedent(precedent["text"], question)
        
        # 판례명이 있으면 함께 표시
        name_str = f" - {precedent.get('name', '')}" if precedent.get('name') else ""
        precedent_contexts.append(f"【판례 {precedent.get('id', '')}{name_str}】\n{summary}")
        
        # 참조 법조항이 있으면 함께 표시
        if precedent.get("referenced_articles") and len(precedent.get("referenced_articles")) > 0:
            refs = ", ".join(precedent.get("referenced_articles")[:3])
            precedent_contexts[-1] += f"\n[참조 법조항: {refs}]"
            
        # 키워드가 있으면 함께 표시
        if precedent.get("keywords") and len(precedent.get("keywords")) > 0:
            keywords_str = ", ".join(precedent.get("keywords")[:5])
            precedent_contexts[-1] += f"\n[관련 키워드: {keywords_str}]"
    
    # 4. 정보 밀도 최적화 (형법 특화 구조화)
    optimized_context = "\n\n".join([
        "### 형법 관련 참고 자료 ###",
        "## 관련 법조항:",
        "\n\n".join(article_contexts) if article_contexts else "관련 법조항 정보가 없습니다.",
        "## 관련 판례:",
        "\n\n".join(precedent_contexts) if precedent_contexts else "관련 판례 정보가 없습니다.",
        "### 참고사항: 형법 해석 시 구성요건-위법성-책임 순서로 판단하며, 법조항과 판례를 함께 고려하십시오. ###"
    ])
    
    return optimized_context

def extract_answer_improved(text):
    """텍스트에서 A, B, C, D 중 하나를 정교한 방식으로 추출합니다."""
    # 정규표현식 패턴들 (확장 및 정교화)
    patterns = [
        # 직접 응답 패턴
        r'^([A-D])$',
        r'^답(?:변|안|)(?:은|): ?([A-D])',
        r'정답(?:은|): ?([A-D])',
        r'([A-D])(?:가|이|을|를) 선택',
        r'([A-D])(?:가|이|이) 정답',
        r'([A-D])(?:가|이|을|를) (?:고른다|고릅니다|고르겠습니다)',
        
        # 간접 응답 패턴
        r'따라서 (?:정답은 |답은 |)([A-D])',
        r'([A-D])(?:가|이|은|는) (?:가장 적절|가장 정확|옳은)',
        
        # 결론 문장 패턴 (확장)
        r'(?:최종적으로|결론적으로|종합하면|따라서|분석 결과|이상의 이유로).{1,50}(?:정답은|답은|옳은 것은|맞는 것은) ?([A-D])',
        r'(?:선택지|옵션) ?([A-D])(?:가|이|은|는) (?:정답|맞습니다|맞다|적절|적합|옳은|타당)',
        r'정답은 선택지 ?([A-D])',
        r'선택지 ?([A-D])(?:을|를)? ?(?:선택합니다|고릅니다|고르겠습니다|골라야 합니다)',
        
        # 비교 분석 패턴
        r'(?:따라서|그러므로|그래서|이에).{0,30}([A-D])(?:이외|를 제외하고|빼고).{0,20}(?:모두|다른 선택지|다른 것)(?:는|은) (?:틀리|오답|부적절|타당하지 않)',
        r'선택지 ([A-D])(?:만|이).{0,30}(?:정확|옳|적절|타당|맞)',
        
        # 기존 패턴들
        r'^\s*([A-D])\s*$',  # 단일 문자 A, B, C, D
        r'(?:정답은|answer is|choice is|선택지는|답은)\s*([A-D])',  # "정답은 A" 등
        r'(?:선택합니다|선택하겠습니다|선택해야 합니다)\s*([A-D])',  # "A를 선택합니다"
        r'([A-D])(?:가|이|을|를)?\s*(?:정답|맞습니다|적절합니다|선택|적절)',  # "A가 정답" 등
        r'([A-D])\s*선택지',  # "A 선택지"
        
        # 부정 표현을 통한 정답 유추
        r'(?:선택지|옵션) ([A-D])(?:을|를)? ?제외한.{1,20}(?:틀리|오답|부적절)',
        r'([A-D])(?:을|를)? ?제외한.{1,20}(?:나머지|다른).{1,20}(?:틀리|오답|부적절)',
    ]
    
    # 패턴 적용 (우선순위 순서로)
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()  # 대문자로 정규화
    
    # 문장별 검색 (더 강화)
    # 마지막 세 문장에 집중 (결론이 주로 마지막에 위치)
    lines = text.split('\n')
    last_lines = lines[-3:] if len(lines) >= 3 else lines
    
    for line in last_lines:
        for pattern in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                return match.group(1).upper()
    
    # 모든 문장 검색
    for line in lines:
        for pattern in patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                return match.group(1).upper()
    
    # 빈도 및 문맥 기반 접근 (개선)
    options = ['A', 'B', 'C', 'D']
    
    # 더 정교한 문맥 패턴 검색
    context_patterns = {
        'A': [r'A\s*(?:만|이|가)\s*(?:옳|정답|맞|적절)', r'선택지\s*A\s*(?:가|이|은|는)\s*(?:옳|정답|맞|적절)'],
        'B': [r'B\s*(?:만|이|가)\s*(?:옳|정답|맞|적절)', r'선택지\s*B\s*(?:가|이|은|는)\s*(?:옳|정답|맞|적절)'],
        'C': [r'C\s*(?:만|이|가)\s*(?:옳|정답|맞|적절)', r'선택지\s*C\s*(?:가|이|은|는)\s*(?:옳|정답|맞|적절)'],
        'D': [r'D\s*(?:만|이|가)\s*(?:옳|정답|맞|적절)', r'선택지\s*D\s*(?:가|이|은|는)\s*(?:옳|정답|맞|적절)']
    }
    
    for option, patterns in context_patterns.items():
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return option
    
    # 빈도 기반 접근 (확장)
    option_counts = {option: 0 for option in options}
    
    # 다양한 형태의 언급 패턴 고려
    for option in options:
        option_counts[option] += text.count(f"선택지 {option}")
        option_counts[option] += text.count(f"{option}.")
        option_counts[option] += text.count(f"{option}이 ")
        option_counts[option] += text.count(f"{option}가 ")
        option_counts[option] += text.count(f"{option}은 ")
        option_counts[option] += text.count(f"{option}는 ")
        option_counts[option] += text.count(f"{option}을 ")
        option_counts[option] += text.count(f"{option}를 ")
        
        # 부정 표현은 빼기
        option_counts[option] -= 2 * text.count(f"{option}이 아닌")
        option_counts[option] -= 2 * text.count(f"{option}가 아닌")
        option_counts[option] -= 2 * text.count(f"{option}은 틀린")
        option_counts[option] -= 2 * text.count(f"{option}는 틀린")
        option_counts[option] -= 2 * text.count(f"{option}은 부적절")
        option_counts[option] -= 2 * text.count(f"{option}는 부적절")
    
    if any(count > 0 for count in option_counts.values()):
        return max(option_counts.items(), key=lambda x: x[1])[0]
    
    # 단순 문자 존재 확인 (최후의 수단)
    option_simple_counts = {option: text.count(option) for option in options}
    if any(count > 0 for count in option_simple_counts.values()):
        return max(option_simple_counts.items(), key=lambda x: x[1])[0]
    
    # 응답에서 아무 것도 찾지 못한 경우
    return None
  
# 셀12: Criminal-Law 평가를 위한 Batch API 사용

# CSV 파일 로드
df = pd.read_csv('./dataset/Criminal-Law-test.csv')
print(f"Loaded {len(df)} questions from CSV file")

# Neo4j 드라이버 다시 연결
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
print("Connected to Neo4j")

# 결과 디렉토리 생성
os.makedirs("results", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 모든 질문에 대해 RAG 검색 실행
print("Performing RAG search for all questions...")
retrieved_contexts = {}

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Searching contexts"):
    question = row['question']
    try:
        # RAG 검색으로 문맥 가져오기
        contexts = graph_enhanced_rag(driver, question, embedding_model, top_k=8)
        retrieved_contexts[idx] = contexts
    except Exception as e:
        print(f"Error in RAG search for question {idx}: {e}")
        retrieved_contexts[idx] = []

print(f"Completed RAG search for {len(retrieved_contexts)} questions")

# Batch API 요청 준비
# Batch API 요청 준비 (수정된 부분)
batch_requests = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preparing batch requests"):
    question = row['question']
    options = {
        'A': row['A'],
        'B': row['B'], 
        'C': row['C'],
        'D': row['D']
    }
    
    # 검색된 문맥 가져오기
    contexts = retrieved_contexts.get(idx, [])
    
    # 최적화된 컨텍스트 구성 (변경된 부분)
    if contexts:
        context_str = optimize_context_construction(contexts, question)
    else:
        context_str = "관련 문맥 정보가 없습니다."
    
    # 프롬프트 작성 (기존 단계적 추론 부분 유지)
    prompt = f"""다음은 한국 형법에 관한 객관식 문제입니다. 제공된 문맥 정보를 참고하여 가장 적절한 답변을 선택하세요.

질문: {question}

선택지:
A. {options['A']}
B. {options['B']}
C. {options['C']}
D. {options['D']}

관련 문맥 정보:
{context_str}

답변 단계:
1) 문제의 핵심 형법 쟁점 파악: 구성요건, 위법성, 책임 중 어떤 단계의 문제인지 분석
2) 관련 법조항 적용: 제시된 법조항이 문제에 어떻게 적용되는지 분석
3) 판례 원칙 적용: 유사한 판례가 확립한 법리를 문제에 적용
4) 각 선택지 법적 분석: 각 선택지가 법조항과 판례에 비추어 왜 맞는지 또는 틀린지 분석
5) 최종 선택: 가장 정확한 선택지 선택

전체 분석 후 최종 답변은 A, B, C, D 중 하나만 제시하세요.
"""
    
    # Batch 요청 생성
    request = {
        "custom_id": f"q_{idx}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "당신은 한국 형법 전문가입니다. 주어진 문맥을 기반으로 가장 적절한 답변을 선택하세요. 답변은 A, B, C, D 중 하나만 명확히 제시하세요."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 300
        }
    }
    
    batch_requests.append(request)

# JSONL 파일로 저장
batch_file_path = f"results/criminal_law_batch_input_{timestamp}.jsonl"
with open(batch_file_path, 'w', encoding='utf-8') as f:
    for request in batch_requests:
        f.write(json.dumps(request, ensure_ascii=False) + '\n')

print(f"Saved {len(batch_requests)} batch requests to {batch_file_path}")

# OpenAI 클라이언트 초기화 및 Batch API 실행
client = OpenAI(api_key=OPENAI_API_KEY)

# 배치 파일 업로드
batch_input_file = client.files.create(
    file=open(batch_file_path, "rb"),
    purpose="batch"
)
batch_input_file_id = batch_input_file.id
print(f"Uploaded batch file with ID: {batch_input_file_id}")

# 배치 작업 생성
batch_job = client.batches.create(
    input_file_id=batch_input_file_id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={"description": "Criminal Law benchmark evaluation"}
)
batch_id = batch_job.id
print(f"Created batch job with ID: {batch_id}")

# 배치 작업 상태 확인 함수
def check_batch_status(client, batch_id):
    """배치 작업의 상태를 확인합니다."""
    batch_status = client.batches.retrieve(batch_id)
    return batch_status

# 작업이 완료될 때까지 대기
print("Waiting for batch job to complete...")
start_time = time.time()
status = None

while True:
    status = check_batch_status(client, batch_id)
    elapsed_time = time.time() - start_time
    print(f"Current status: {status.status} (Elapsed: {elapsed_time:.2f}s)")
    
    if status.status in ['completed', 'failed', 'cancelled', 'expired']:
        break
    
    # 처음 10분은 30초마다, 이후에는 2분마다 체크
    if elapsed_time < 600:  # 10분
        time.sleep(30)
    else:
        time.sleep(120)

end_time = time.time()
total_time = end_time - start_time
print(f"Batch job finished with status: {status.status} in {total_time:.2f} seconds")

# 작업이 성공적으로 완료된 경우 결과 처리
if status.status == 'completed':
    output_file_id = status.output_file_id
    print(f"Batch job completed successfully. Output file ID: {output_file_id}")
    
    # 결과 파일 다운로드
    file_response = client.files.content(output_file_id)
    batch_results = []
    
    for line in file_response.text.split('\n'):
        if line.strip():
            batch_results.append(json.loads(line))
    
    print(f"Downloaded {len(batch_results)} results from the batch job")
    
    # 결과 파일 저장 (요구사항대로)
    output_file_path = f"results/criminal_law_batch_output_{timestamp}.jsonl"
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for result in batch_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"Saved batch output to {output_file_path}")
    
    # 정확도 평가 준비
    def extract_answer(text):
        return extract_answer_improved(text)
    
    # 정확도 평가
    correct_count = 0
    results_with_answers = []
    
    for result in batch_results:
        custom_id = result['custom_id']
        idx = int(custom_id.split('_')[1])
        
        if result.get('error') is not None:
            print(f"Error in result {custom_id}: {result['error']}")
            continue
        
        try:
            response_text = result['response']['body']['choices'][0]['message']['content'].strip()
            
            # 응답에서 답변 추출 (A, B, C, D 중 하나)
            answer = extract_answer(response_text)
            
            if answer is None:
                print(f"Could not extract answer from response for question {idx}: {response_text}")
                continue
            
            # 정답과 비교 (CSV에서는 1-indexed, 1=A, 2=B, 3=C, 4=D)
            correct_answer = chr(64 + df.iloc[idx]['answer'])  # 1->A, 2->B, 3->C, 4->D
            is_correct = (answer == correct_answer)
            
            if is_correct:
                correct_count += 1
            
            results_with_answers.append({
                'question_id': idx,
                'question': df.iloc[idx]['question'],
                'predicted': answer,
                'actual': correct_answer,
                'is_correct': is_correct,
                'response': response_text
            })
        except Exception as e:
            print(f"Error processing result for question {idx}: {e}")
    
    accuracy = correct_count / len(results_with_answers) if results_with_answers else 0
    print(f"Accuracy: {accuracy:.4f} ({correct_count}/{len(results_with_answers)})")
    
    # 결과를 CSV 파일로 저장
    results_df = pd.DataFrame(results_with_answers)
    results_file = f"results/criminal_law_results_{timestamp}.csv"
    results_df.to_csv(results_file, index=False)
    print(f"Saved detailed results to {results_file}")

    # 결과 요약 정보 저장
    summary = {
        'timestamp': timestamp,
        'total_questions': len(df),
        'processed_questions': len(results_with_answers),
        'correct_answers': correct_count,
        'accuracy': accuracy,
        'batch_processing_time_seconds': total_time,
        'input_file': batch_file_path,
        'output_file': output_file_path,
        'results_file': results_file,
        'batch_id': batch_id
    }
    
    with open(f"results/criminal_law_benchmark_summary_{timestamp}.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        
    print(f"Benchmark evaluation completed. Final accuracy: {accuracy:.4f}")
    
else:
    print(f"Batch job did not complete successfully. Final status: {status.status}")
    if hasattr(status, 'errors') and status.errors:
        print("Errors:")
        for error in status.errors:
            print(f"  - {error}")

# 드라이버 연결 종료
driver.close()
print("Neo4j driver connection closed")

# 셀 13: 결과 분석 및 시각화
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import display, HTML
import plotly.io as pio

# 기본 테마 설정
pio.templates.default = "plotly_white"

# 가장 최근 결과 파일 찾기
import glob
import os
result_files = glob.glob("results/criminal_law_results_20250412_174738.csv")
latest_result_file = max(result_files, key=os.path.getctime)
print(f"분석할 결과 파일: {latest_result_file}")

# 결과 데이터 로드
results_df = pd.read_csv(latest_result_file)
print(f"로드된 결과 수: {len(results_df)}")

# 요약 통계
total_questions = len(results_df)
correct_answers = results_df['is_correct'].sum()
accuracy = correct_answers / total_questions

print("\n===== 벤치마크 결과 요약 =====")
print(f"총 문제 수: {total_questions}")
print(f"정답 수: {int(correct_answers)}")
print(f"정확도: {accuracy:.4f} ({int(correct_answers)}/{total_questions})")

# 예측 분포 표시
prediction_counts = results_df['predicted'].value_counts()
print("\n===== 예측 분포 =====")
for option, count in prediction_counts.items():
    print(f"옵션 {option}: {count}개 ({count/total_questions*100:.1f}%)")

# 실제 정답 분포 표시
actual_counts = results_df['actual'].value_counts()
print("\n===== 실제 정답 분포 =====")
for option, count in actual_counts.items():
    print(f"옵션 {option}: {count}개 ({count/total_questions*100:.1f}%)")

# 시각화: 옵션 선택 분포 (예측 vs 실제)
fig = go.Figure()
options = sorted(list(set(prediction_counts.index) | set(actual_counts.index)))

fig.add_trace(go.Bar(
    x=options,
    y=[prediction_counts.get(option, 0) for option in options],
    name='모델 예측',
    marker_color='rgb(55, 83, 109)'
))

fig.add_trace(go.Bar(
    x=options,
    y=[actual_counts.get(option, 0) for option in options],
    name='실제 정답',
    marker_color='rgb(26, 118, 255)'
))

fig.update_layout(
    title='옵션 선택 분포 (예측 vs 실제)',
    xaxis_title='옵션',
    yaxis_title='문제 수',
    barmode='group',
    bargap=0.15,
    bargroupgap=0.1,
    font=dict(size=14)
)

fig.show()

# 시각화: 혼동 행렬(Confusion Matrix)
conf_matrix_data = pd.crosstab(results_df['actual'], results_df['predicted']).values
conf_matrix_norm = (conf_matrix_data.T / conf_matrix_data.sum(axis=1)).T  # 행별 정규화

options = sorted(list(set(results_df['actual']) | set(results_df['predicted'])))
conf_matrix_df = pd.DataFrame(0, index=options, columns=options)
for actual in results_df['actual'].unique():
    for predicted in results_df[results_df['actual'] == actual]['predicted'].unique():
        count = len(results_df[(results_df['actual'] == actual) & (results_df['predicted'] == predicted)])
        conf_matrix_df.loc[actual, predicted] = count

# 비율로 변환
conf_matrix_norm = conf_matrix_df.div(conf_matrix_df.sum(axis=1), axis=0)

fig = px.imshow(
    conf_matrix_norm,
    labels=dict(x="예측", y="실제", color="비율"),
    x=options,
    y=options,
    color_continuous_scale="Blues",
    text_auto='.2f'
)

fig.update_layout(
    title='혼동 행렬 (Confusion Matrix)',
    xaxis_title='예측',
    yaxis_title='실제',
    font=dict(size=14)
)

fig.show()

# 정답률 파이 차트
labels = ['정답', '오답']
values = [correct_answers, total_questions - correct_answers]
colors = ['rgb(46, 204, 113)', 'rgb(231, 76, 60)']

fig = go.Figure(data=[go.Pie(
    labels=labels,
    values=values,
    hole=.4,
    marker_colors=colors
)])

fig.update_layout(
    title='정답률',
    font=dict(size=14)
)

fig.show()

# 정확도가 높은 문제와 낮은 문제 살펴보기
print("\n===== 정답 예시 (5개) =====")
correct_examples = results_df[results_df['is_correct'] == True].head(5)
for i, row in correct_examples.iterrows():
    print(f"문제 ID: {row['question_id']}")
    print(f"질문: {row['question'][:100]}..." if len(row['question']) > 100 else f"질문: {row['question']}")
    print(f"예측/정답: {row['predicted']}/{row['actual']}")
    print("-" * 80)

print("\n===== 오답 예시 (5개) =====")
incorrect_examples = results_df[results_df['is_correct'] == False].head(5)
for i, row in incorrect_examples.iterrows():
    print(f"문제 ID: {row['question_id']}")
    print(f"질문: {row['question'][:100]}..." if len(row['question']) > 100 else f"질문: {row['question']}")
    print(f"예측/정답: {row['predicted']}/{row['actual']}")
    print("-" * 80)

# 요약 정보 박스 표시 (강조)
summary_html = f"""
<div style="background-color: #f8f9fa; padding: 20px; border-radius: 5px; border-left: 5px solid #4e73df;">
    <h3 style="margin-top: 0;">형법 벤치마크 평가 결과</h3>
    <p><b>정확도:</b> {accuracy:.2%} ({int(correct_answers)}/{total_questions})</p>
    <p><b>모델:</b> GPT-4o-mini</p>
    <p><b>평가 방식:</b> 그래프 기반 RAG + Batch API</p>
</div>
"""
display(HTML(summary_html))

# 각 문제 유형별 정답률 분석 - 추가 CSV 파일 필요 시
try:
    # 원본 테스트 CSV가 있다면 카테고리별 분석 시도
    original_test_df = pd.read_csv('./dataset/CriminalLawtest.csv')
    if 'Category' in original_test_df.columns:
        # 결과와 원본 테스트 데이터 병합
        merged_df = pd.merge(results_df, original_test_df[['question', 'Category']], on='question', how='left')
        category_accuracy = merged_df.groupby('Category')['is_correct'].mean().sort_values(ascending=False)
        category_counts = merged_df.groupby('Category')['is_correct'].count()
        category_correct = merged_df.groupby('Category')['is_correct'].sum()
        
        print("\n===== 카테고리별 정확도 =====")
        
        # 카테고리별 정확도 그래프
        fig = px.bar(
            x=category_accuracy.index,
            y=category_accuracy.values,
            color=category_accuracy.values,
            color_continuous_scale='RdYlGn',
            labels={'x': '카테고리', 'y': '정확도', 'color': '정확도'},
            text=[f"{v:.2f}" for v in category_accuracy.values]
        )
        
        fig.update_layout(
            title='카테고리별 정확도',
            xaxis_title='카테고리',
            yaxis_title='정확도',
            font=dict(size=14),
            xaxis={'categoryorder': 'total descending'}
        )
        
        fig.show()
        
        # 표 형태로도 출력
        for category, acc in category_accuracy.items():
            correct = int(category_correct[category])
            count = int(category_counts[category])
            print(f"{category}: {acc:.4f} ({correct}/{count})")
except Exception as e:
    print(f"카테고리별 분석을 수행할 수 없습니다: {e}")

# 예측 정확도 추세 - 문제 번호별
results_df['bin'] = results_df['question_id'] // 20  # 20문제씩 그룹화
bin_accuracy = results_df.groupby('bin')['is_correct'].mean()
bin_labels = [f"{i*20}-{(i+1)*20-1}" for i in bin_accuracy.index]

fig = px.line(
    x=bin_labels,
    y=bin_accuracy.values,
    markers=True,
    labels={'x': '문제 번호 구간', 'y': '정확도'}
)

fig.update_layout(
    title='문제 번호별 정확도 추세',
    xaxis_title='문제 번호 구간',
    yaxis_title='정확도',
    font=dict(size=14)
)

fig.add_hline(
    y=accuracy, 
    line_dash="dash", 
    line_color="red",
    annotation_text=f"평균 정확도: {accuracy:.2f}",
    annotation_position="bottom right"
)

fig.show()