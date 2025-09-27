# RAG 检索优化的五种常见手段及实现

## 概述
RAG（Retrieval-Augmented Generation）通过检索外部知识来增强大模型的生成效果。本文介绍五种常见的 RAG 检索优化手段，并通过一个纯 Python 实现的示例脚本展示其实现细节。

## 优化手段

### 1. 混合检索（Dense + Sparse）
混合检索结合了稀疏检索（如 BM25）和密集检索（如向量检索）的优点，显著提升召回率与准确率。

**实现代码片段**：
```python
def hybrid_retrieve(query: str, docs: List[str], top_k: int = 3) -> List[Tuple[int, float]]:
    bm25 = bm25_scores(query, docs)
    dense = dense_scores(query, docs)
    fusion = [0.7 * b + 0.3 * d for b, d in zip(bm25, dense)]
    return sorted(enumerate(fusion), key=lambda x: x[1], reverse=True)[:top_k]
```

### 2. 查询构建（Query Construction）
查询构建技术包括拼写纠错、同义词扩展等，用于优化用户输入的查询。

**实现代码片段**：
```python
def construct_query(raw_query: str) -> str:
    q = spell_correct(raw_query)  # 拼写纠错
    q = synonym_expand(q)         # 同义词扩展
    return q
```

### 3. 查询翻译（Query Translation）
将用户问题转化为结构化查询或另一种自然语言，提升检索效果。

**实现代码片段**：
```python
def translate_to_english(query: str) -> str:
    mapping = {"混合检索": "hybrid retrieval", "重排序": "re-ranking"}
    for zh, en in mapping.items():
        query = re.sub(rf"\b{zh}\b", en, query, flags=re.I)
    return query
```

### 4. 查询路由（Query Routing）
根据查询意图将问题分发到最合适的知识库或检索子系统。

**实现代码片段**：
```python
def route_query(query: str) -> str:
    keywords = {"混合": "hybrid", "重排序": "rerank"}
    lowered = query.lower()
    for kw, domain in keywords.items():
        if kw in lowered:
            return domain
    return "general"
```

### 5. 重排序（Re-ranking）
使用更精细的模型对初排结果再次打分，提升结果相关性。

**实现代码片段**：
```python
def rerank(query: str, candidates: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
    def tri_gram_overlap(q, d):
        q3 = set("".join(t) for t in zip(*[iter(q)] * 3))
        d3 = set("".join(t) for t in zip(*[iter(d)] * 3))
        return len(q3 & d3) / (len(q3 | d3) + 1e-8)

    new_scores = []
    for idx, old_score in candidates:
        overlap = tri_gram_overlap(query, CORPUS[idx])
        new_score = 0.7 * old_score + 0.3 * overlap
        new_scores.append((idx, new_score))
    return sorted(new_scores, key=lambda x: x[1], reverse=True)
```

## 端到端 Pipeline
将上述优化手段整合为一个端到端的 RAG 检索流程：

```python
def advanced_rag_pipeline(raw_query: str):
    q1 = construct_query(raw_query)          # 查询构建
    q2 = translate_to_english(q1)            # 查询翻译
    domain = route_query(q2)                # 查询路由
    filtered = [d for d in CORPUS if ...]   # 路由过滤
    candidates = hybrid_retrieve(q2, filtered)  # 混合检索
    reranked = rerank(q2, candidates)       # 重排序
    return reranked
```

## 总结
通过混合检索、查询构建、查询翻译、查询路由和重排序五种手段，可以显著提升 RAG 的检索效果。本文的示例代码展示了如何用纯 Python 实现这些优化技术，无需依赖第三方库。

**完整代码**：[optimized_rag_demo.py](optimized_rag_demo.py)