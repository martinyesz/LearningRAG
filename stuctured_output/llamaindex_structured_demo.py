from llama_index.core import Settings
from llama_index.core.llms import ChatMessage
from llama_index.llms.deepseek import DeepSeek
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.prompts import PromptTemplate
from pydantic import BaseModel, Field
import os

# 定义数据结构
class BookInfo(BaseModel):
    title: str = Field(description="书籍标题")
    author: str = Field(description="作者姓名")
    publication_year: int = Field(description="出版年份")
    genre: str = Field(description="书籍类型")
    summary: str = Field(description="书籍摘要")

# 初始化 DeepSeek LLM
llm = DeepSeek(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0.2
)

Settings.llm = llm

# 创建输出解析器
output_parser = PydanticOutputParser(BookInfo)

# 创建提示模板
prompt_str = (
    "请从以下文本中提取书籍信息，并严格按照指定格式输出。\n"
    "{format_instructions}\n\n"
    "文本内容：{text}"
)

prompt = PromptTemplate(prompt_str)

# 测试数据
book_text = """
《三体》是中国科幻作家刘慈欣创作的科幻小说，首次出版于2006年。
这部作品属于硬科幻类型，讲述了地球文明与三体文明的接触和冲突。
小说以其宏大的宇宙观和深刻的哲学思考而闻名。
"""

# 构建消息
messages = [
    ChatMessage(
        role="user",
        content=prompt.format(
            format_instructions=output_parser.get_format_string(),
            text=book_text
        )
    )
]

# 调用 LLM
try:
    response = llm.chat(messages)
    parsed_output = output_parser.parse(response.message.content)
    print("LlamaIndex Structured Output 结果:")
    print(f"标题: {parsed_output.title}")
    print(f"作者: {parsed_output.author}")
    print(f"出版年份: {parsed_output.publication_year}")
    print(f"类型: {parsed_output.genre}")
    print(f"摘要: {parsed_output.summary}")
    print(f"JSON 格式: {parsed_output.model_dump_json(indent=2)}")
except Exception as e:
    print(f"解析错误: {e}")