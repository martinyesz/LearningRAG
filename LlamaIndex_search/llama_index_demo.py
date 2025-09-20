# llama_index_demo.py
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

# 设置嵌入模型（本地免费模型）
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 禁用 LLM（避免依赖 OpenAI）
Settings.llm = None

# 1. 加载数据
print("📂 正在加载数据...")
from llama_index.core import Document
import json
import os

documents = []
for filename in os.listdir("./data"):
    if filename.endswith(".json"):
        with open(os.path.join("./data", filename), "r", encoding="utf-8") as f:
            data = json.load(f)
            documents.append(Document(text=data["content"], metadata=data["metadata"]))

# 2. 构建向量索引
print("🏗️  正在构建向量索引...")
index = VectorStoreIndex.from_documents(documents)

# 3. 创建查询引擎
print("🔍 创建查询引擎...")
query_engine = index.as_query_engine(similarity_top_k=2)  # 返回最相似的2个结果

# 4. 执行相似性搜索
queries = [
    "哪种水果富含维生素C？",
    "运动员适合吃什么水果补充能量？",
    "对心脏有益的水果是什么？"
]

for i, query in enumerate(queries, 1):
    print(f"\n--- 查询 {i}: {query} ---")
    response = query_engine.query(query)
    print("💬 回答:", str(response))
    # 打印参考来源（可选）
    print("📄 参考来源:")
    for node in response.source_nodes:
        print(f"  - {node.node.text[:100]}...")

print("\n✅ 演示完成！")