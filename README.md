# LearningRAG
Python samples for how to implement RAG (Retrieval Augmented Generation).

## LlamaIndex_search 目录

### 功能说明
`llama_index_demo.py` 是一个基于 LlamaIndex 的相似性搜索演示脚本，支持以下功能：
1. **加载数据**：从 `./data` 目录加载 JSON 文件，提取 `content` 和 `metadata` 字段。
2. **构建向量索引**：使用 `VectorStoreIndex` 构建文档的向量索引。
3. **创建查询引擎**：支持相似性搜索，返回最相似的 2 个结果。
4. **执行查询**：对预设的查询问题（如“哪种水果富含维生素C？”）进行搜索，并显示结果和参考来源。

### 运行方式
1. 确保安装依赖（见 `requirements.txt`）。
2. 在 `LlamaIndex_search` 目录下运行：
   ```bash
   python llama_index_demo.py
   ```
