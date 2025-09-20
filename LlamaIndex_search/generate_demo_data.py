# generate_demo_data.py
import os
import json

os.makedirs("data", exist_ok=True)

docs = {
    "data/doc1.json": {
        "id": 1,
        "content": "苹果是一种常见的水果，富含维生素C和纤维。它有助于消化和增强免疫力。每天吃一个苹果，医生远离我。",
        "metadata": {"source": "doc1.json"}
    },
    "data/doc2.json": {
        "id": 2,
        "content": "香蕉是热带水果，含有丰富的钾元素，有助于维持心脏健康和肌肉功能。香蕉也常被运动员作为能量补充食品。",
        "metadata": {"source": "doc2.json"}
    },
    "data/doc3.json": {
        "id": 3,
        "content": "橙子酸甜可口，是冬季最受欢迎的水果之一。它含有大量维生素C，可以预防感冒，促进胶原蛋白合成。",
        "metadata": {"source": "doc3.json"}
    }
}

for filename, data in docs.items():
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

print("✅ 3个演示数据文件（JSON格式）已生成在 ./data/ 目录下")