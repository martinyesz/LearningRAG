import os
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_deepseek import ChatDeepSeek

# 初始化 LLM
llm = ChatDeepSeek(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0.1,
)

# 定义数据结构
class UserInfo(BaseModel):
    name: str = Field(description="用户的姓名")
    age: int = Field(description="用户的年龄")
    email: str = Field(description="用户的邮箱地址")
    interests: list[str] = Field(description="用户的兴趣爱好列表")

# 创建输出解析器
parser = PydanticOutputParser(pydantic_object=UserInfo)

# 创建提示模板
prompt = PromptTemplate(
    template="请根据以下信息提取用户数据：\n{format_instructions}\n\n用户描述：{user_description}\n",
    input_variables=["user_description"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)


# 创建链
chain = prompt | llm | parser

# 测试数据
user_description = "张三是一名28岁的软件工程师，他的邮箱是zhangsan@example.com，喜欢编程、阅读和徒步旅行。"

# 执行链
try:
    result = chain.invoke({"user_description": user_description})
    print("PydanticOutputParser 结果:")
    print(f"姓名: {result.name}")
    print(f"年龄: {result.age}")
    print(f"邮箱: {result.email}")
    print(f"兴趣: {', '.join(result.interests)}")
    print(f"JSON 格式: {result.model_dump_json(indent=2)}")
except Exception as e:
    print(f"解析错误: {e}")