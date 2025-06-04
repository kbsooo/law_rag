import sys
import types
import os

# Ensure project root is on the import path
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Stub external dependencies required by main.py so the module can be imported
stub_names = [
    "pandas",
    "numpy",
    "neo4j",
    "openai",
]
for name in stub_names:
    mod = sys.modules.setdefault(name, types.ModuleType(name))
    if name == "pandas":
        mod.DataFrame = object
    if name == "neo4j":
        mod.GraphDatabase = object
        mod.Driver = object
    if name == "openai":
        mod.OpenAI = object

dotenv = types.ModuleType("dotenv")
dotenv.load_dotenv = lambda *_, **__: None
sys.modules.setdefault("dotenv", dotenv)

lc_comm = types.ModuleType("langchain_community")
document_loaders = types.ModuleType("langchain_community.document_loaders")
document_loaders.PyPDFLoader = object
lc_comm.document_loaders = document_loaders
sys.modules.setdefault("langchain_community", lc_comm)
sys.modules.setdefault("langchain_community.document_loaders", document_loaders)

lc_openai = types.ModuleType("langchain_openai")
lc_openai.OpenAIEmbeddings = object
sys.modules.setdefault("langchain_openai", lc_openai)

tqdm_mod = types.ModuleType("tqdm")
tqdm_mod.tqdm = lambda x, *a, **k: x
sys.modules.setdefault("tqdm", tqdm_mod)

import pytest
from main import extract_choice_from_response


def test_standard_pattern():
    text = "문제에 대한 해설\n정답: A"
    assert extract_choice_from_response(text) == "A"


def test_bracket_pattern_last_line():
    text = "이 문제의 답은 다음과 같습니다.\n(B)"
    assert extract_choice_from_response(text) == "B"


def test_ambiguous_sentence_scoring():
    text = "A 는 틀린 설명이다. B 는 맞는 선택이다."
    assert extract_choice_from_response(text) == "B"


def test_unmatched_returns_none():
    text = "정답을 확신할 수 없습니다."
    assert extract_choice_from_response(text) is None
