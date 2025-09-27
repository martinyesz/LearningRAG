# 通过LangChain、LlamaIndex和Function Calling实现结构化输出

## 1. LangChain
LangChain 是一个强大的框架，用于构建基于语言模型的应用程序。它提供了多种工具和方法来生成结构化输出。

### 方法
- **使用Pydantic模型**：通过定义Pydantic模型，可以强制语言模型输出符合特定结构的数据。
- **链式调用**：通过链式调用（Chain）将多个步骤组合起来，逐步生成结构化数据。
- **输出解析器**：使用`OutputParser`将模型的自由文本输出转换为结构化格式。

### 示例
```python
from langchain import LLMChain, PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

parser = PydanticOutputParser(pydantic_object=Person)
prompt = PromptTemplate(
    template="Provide a person's name and age.\n{format_instructions}\n",
    input_variables=[],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = LLMChain(llm=llm, prompt=prompt)
output = chain.run({})
parsed_output = parser.parse(output)
```

---

## 2. LlamaIndex
LlamaIndex 是一个用于构建和查询索引的工具，特别适合处理结构化数据。

### 方法
- **索引构建**：通过构建索引，将非结构化数据转换为结构化数据。
- **查询引擎**：使用查询引擎从索引中提取结构化信息。
- **自定义输出格式**：通过定义模板或规则，控制输出的格式。

### 示例
```python
from llama_index import VectorStoreIndex, SimpleDirectoryReader

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("What is the capital of France?")
print(response)
```

---

## 3. Function Calling
Function Calling 是一种通过定义函数和参数，让语言模型生成结构化输出的方法。

### 方法
- **定义函数**：明确函数的输入和输出结构。
- **调用模型**：将函数定义传递给模型，模型返回符合函数签名的结构化数据。
- **验证输出**：确保模型的输出符合预期的结构。

### 示例
```python
import openai

function_descriptions = [
    {
        "name": "get_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g., San Francisco, CA"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"]
                }
            },
            "required": ["location"]
        }
    }
]

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "What's the weather like in Boston?"}],
    functions=function_descriptions,
    function_call="auto"
)

print(response["choices"][0]["message"]["function_call"])
```

---

## 总结
- **LangChain**：适合复杂的数据处理流程，支持链式调用和输出解析。
- **LlamaIndex**：适合从非结构化数据中提取结构化信息。
- **Function Calling**：适合简单的结构化输出需求，通过函数定义实现。

根据具体需求选择合适的方法，可以高效地生成结构化输出。