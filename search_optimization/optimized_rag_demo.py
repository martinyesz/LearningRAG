#!/usr/bin/env python3
"""
====================================================
Advanced RAG Retrieval Demo（纯 Python，无第三方依赖）
====================================================
本脚本演示了 5 种优化检索的常见手段：
1. 混合检索（Dense + Sparse）
2. 查询构建（Query Construction）
3. 查询翻译（Query Translation）
4. 查询路由（Query Routing）
5. 重排序（Re-ranking）

运行方式：
$ python advanced_rag_demo.py
"""

import json
import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

# ---------- 1. 模拟语料库 ----------
CORPUS = [
    "Python 是一种解释型、面向对象、动态数据类型的高级程序设计语言。",
    "FastAPI 是一个现代、快速（高性能）的 Web 框架，用于基于 Python 3.7+ 构建 API。",
    "RAG（Retrieval-Augmented Generation）通过检索外部知识来增强大模型生成效果。",
    "向量数据库（如 Chroma、Pinecone、Weaviate）是 Dense Retrieval 的核心基础设施。",
    "BM25 是一种经典稀疏检索算法，常用于 Elasticsearch、Lucene 等搜索系统。",
    "混合检索结合 Dense 与 Sparse 的优点，可显著提升召回率与准确率。",
    "查询构建技术包括拼写纠错、同义词扩展、实体识别等。",
    "查询翻译可将用户问题转化为结构化查询（如 SQL、Cypher）或另一种自然语言。",
    "查询路由根据意图将问题分发到最合适的知识库或检索子系统。",
    "重排序阶段使用更精细的模型（如 Cross-Encoder）对初排结果再次打分。",
]

# ---------- 2. 工具函数 ----------
def tokenize(text: str) -> List[str]:
    """极简中文分词：按字切分 + 小写化"""
    return list(text.lower())

def bm25_scores(query: str, docs: List[str], k1: float = 1.5, b: float = 0.75) -> List[float]:
    """简易 BM25 稀疏打分"""
    N = len(docs)
    doc_lens = [len(tokenize(d)) for d in docs]
    avgdl = sum(doc_lens) / N
    freqs = []  # 每个文档的词频 Counter
    df = Counter()  # 文档频率
    for d in docs:
        cnt = Counter(tokenize(d))
        freqs.append(cnt)
        for w in cnt:
            df[w] += 1
    query_tokens = tokenize(query)
    scores = []
    for i, cnt in enumerate(freqs):
        score = 0.0
        dl = doc_lens[i]
        for w in query_tokens:
            tf = cnt[w]
            idf = math.log((N - df[w] + 0.5) / (df[w] + 0.5) + 1.0)
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * dl / avgdl)
            score += idf * numerator / denominator
        scores.append(score)
    return scores

def dense_scores(query: str, docs: List[str]) -> List[float]:
    """极简 Dense 向量打分：用共享随机向量模拟语义相似度"""
    import random
    random.seed(42)
    dim = 128
    # 预生成随机向量
    vocab = set(tokenize(" ".join(docs + [query])))
    vec_map = {w: [random.gauss(0, 1) for _ in range(dim)] for w in vocab}

    def vec(text):
        tokens = tokenize(text)
        return [sum(col) / len(tokens) for col in zip(*[vec_map[t] for t in tokens])]

    qvec = vec(query)
    dvecs = [vec(d) for d in docs]

    def cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        return dot / (norm_a * norm_b + 1e-8)

    return [cosine(qvec, dv) for dv in dvecs]

def hybrid_retrieve(query: str, docs: List[str], top_k: int = 3) -> List[Tuple[int, float]]:
    """1. 混合检索 = BM25 + Dense，分数加权平均"""
    bm25 = bm25_scores(query, docs)
    dense = dense_scores(query, docs)
    # 直接加权平均（跳过归一化）
    fusion = [0.7 * b + 0.3 * d for b, d in zip(bm25, dense)]
    scored = sorted(enumerate(fusion), key=lambda x: x[1], reverse=True)[:top_k]
    return scored

# ---------- 3. 查询构建 ----------
def spell_correct(query: str) -> str:
    """极简拼写纠错：预定义映射表"""
    corrections = {"pyton": "Python", "fastpi": "FastAPI", "rag": "RAG"}
    for wrong, right in corrections.items():
        query = re.sub(rf"\b{wrong}\b", right, query, flags=re.I)
    return query

def synonym_expand(query: str) -> str:
    """极简同义词扩展"""
    synonyms = {
        "Python": ["py"],
        "FastAPI": ["fast api"],
        "RAG": ["retrieval augmented generation"],
    }
    tokens = query.split()
    new_tokens = []
    for tok in tokens:
        new_tokens.append(tok)
        for key, syns in synonyms.items():
            if re.match(rf"\b{key}\b", tok, re.I):
                new_tokens.extend(syns)
    return " ".join(new_tokens)

def construct_query(raw_query: str) -> str:
    """2. 查询构建入口"""
    q = spell_correct(raw_query)
    q = synonym_expand(q)
    return q

# ---------- 4. 查询翻译 ----------
def translate_to_english(query: str) -> str:
    """极简中→英翻译（映射表）"""
    mapping = {
        "python": "Python",
        "fastapi": "FastAPI",
        "rag": "RAG",
        "向量数据库": "vector database",
        "混合检索": "hybrid retrieval",
        "重排序": "re-ranking",
    }
    for zh, en in mapping.items():
        query = re.sub(rf"\b{zh}\b", en, query, flags=re.I)
    return query

# ---------- 5. 查询路由 ----------
def route_query(query: str) -> str:
    """3. 查询路由：根据关键词决定用哪个子语料库"""
    keywords = {
        "python": "python",
        "fastapi": "fastapi",
        "rag": "rag",
        "向量": "vector",
        "数据库": "vector",
        "bm25": "bm25",
        "混合": "hybrid",
        "重排序": "rerank",
    }
    lowered = query.lower()
    for kw, domain in keywords.items():
        if kw in lowered:
            return domain
    return "general"

# ---------- 6. 重排序 ----------
def rerank(query: str, candidates: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
    """4. 使用 Cross-Encoder 风格的重打分（模拟）"""
    # 用更复杂的规则：计算 query 与文档的 3-gram 重叠率
    def tri_gram_overlap(q, d):
        q3 = set("".join(t) for t in zip(*[iter(q)] * 3))
        d3 = set("".join(t) for t in zip(*[iter(d)] * 3))
        return len(q3 & d3) / (len(q3 | d3) + 1e-8)

    new_scores = []
    for idx, old_score in candidates:
        doc = CORPUS[idx]
        overlap = tri_gram_overlap(query, doc)
        # 旧分数与重叠率加权
        new_score = 0.7 * old_score + 0.3 * overlap
        new_scores.append((idx, new_score))
    return sorted(new_scores, key=lambda x: x[1], reverse=True)

# ---------- 7. 端到端 Pipeline ----------
def advanced_rag_pipeline(raw_query: str):
    print("【原始查询】", raw_query)
    # 查询构建
    q1 = construct_query(raw_query)
    print("【查询构建】", q1)
    # 查询翻译
    q2 = translate_to_english(q1)
    print("【查询翻译】", q2)
    # 查询路由
    domain = route_query(q2)
    print("【查询路由】domain =", domain)
    # 根据路由过滤语料（优化：强制包含"混合检索"的文档）
    filtered = [
        d for d in CORPUS if "混合检索" in d or domain == "general" or domain.lower() in d.lower()
    ]
    print("【路由过滤后文档数】", len(filtered))
    print("【过滤后文档】", filtered)
    # 检查路由结果是否为空
    if not filtered:
        print("【警告】未找到相关文档，将返回通用结果")
        filtered = CORPUS
    # 混合检索（打印原始分数）
    bm25 = bm25_scores(q2, filtered)
    dense = dense_scores(q2, filtered)
    print("【BM25原始分数】", bm25)
    print("【Dense原始分数】", dense)
    candidates = hybrid_retrieve(q2, filtered, top_k=5)
    print("【混合检索 Top-5】")
    for idx, score in candidates:
        print(f"  {score:.3f} | {filtered[idx]}")
    # 重排序
    reranked = rerank(q2, candidates)
    print("【重排序后 Top-3】")
    for idx, score in reranked[:3]:
        print(f"  {score:.3f} | {filtered[idx]}")

# ---------- 8. 运行示例 ----------
if __name__ == "__main__":
    # 演示问题列表
    demo_queries = [
        "什么是混合检索？",
        "Python 是什么？",
        "FastAPI 的用途是什么？",
        "RAG 如何工作？",
        "BM25 是什么？"
    ]

    # 自动运行演示
    print("===== 开始自动演示 =====")
    for query in demo_queries:
        print(f"\n=== 演示问题: {query} ===")
        advanced_rag_pipeline(query)
    print("\n===== 自动演示完成 =====")