from openai import OpenAI
import os
import json
from typing import List, Dict, Any, Optional

# 初始化 OpenAI 客户端（兼容 DeepSeek API）
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

def send_messages(
    messages: List[Dict[str, Any]], 
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: str = "auto"
) -> Any:
    """
    发送消息并获取模型响应
    
    Args:
        messages: 消息历史列表
        tools: 可用工具列表
        tool_choice: 工具选择策略 ("auto", "none", 或指定工具)
    
    Returns:
        模型返回的消息对象
    """
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        return response.choices[0].message
    except Exception as e:
        print(f"❌ API 调用失败: {e}")
        raise

# 1. 定义工具（函数）的 Schema
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定地点的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市和省份，例如：杭州市, 浙江省",
                    }
                },
                "required": ["location"],
                "additionalProperties": False  # 增强安全性
            },
        }
    },
    {
        "type": "function",
        "function": {1
            "name": "get_current_time",
            "description": "获取当前日期和时间",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "时区，例如：Asia/Shanghai",
                    }
                },
                "required": [],
                "additionalProperties": False
            },
        }
    }
]

def execute_tool(tool_name: str, tool_arguments: Dict[str, Any]) -> str:
    """
    执行工具函数（模拟实现）
    
    Args:
        tool_name: 工具名称
        tool_arguments: 工具参数字典
    
    Returns:
        工具执行结果
    """
    if tool_name == "get_weather":
        location = tool_arguments.get("location", "未知地点")
        # 这里可以集成真实的天气 API
        return f"{location}当前天气：24℃，晴朗，微风"
    
    elif tool_name == "get_current_time":
        import datetime
        timezone = tool_arguments.get("timezone", "UTC")
        if timezone == "Asia/Shanghai":
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return f"当前北京时间：{now}"
        else:
            now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            return f"当前时间：{now}"
    
    else:
        return f"未知工具: {tool_name}"

def handle_conversation(user_input: str, tools: List[Dict[str, Any]]) -> None:
    """
    处理完整的对话流程（包括可能的工具调用）
    
    Args:
        user_input: 用户输入
        tools: 可用工具列表
    """
    messages = [{"role": "user", "content": user_input}]
    print(f"User> {user_input}\n")
    
    try:
        # 第一次调用：获取模型响应（可能包含工具调用）
        message = send_messages(messages, tools=tools)
        
        # 检查是否需要工具调用
        if hasattr(message, 'tool_calls') and message.tool_calls:
            print("--- 模型发起了工具调用 ---")
            messages.append(message)  # 保存模型的工具调用请求
            
            # 处理所有工具调用（支持多个）
            for tool_call in message.tool_calls:
                function_info = tool_call.function
                print(f"工具名称: {function_info.name}")
                
                # 安全解析 JSON 参数
                try:
                    arguments = json.loads(function_info.arguments)
                    print(f"工具参数: {arguments}")
                except json.JSONDecodeError as e:
                    print(f"❌ 参数解析失败: {e}")
                    arguments = {}
                
                # 执行工具
                tool_output = execute_tool(function_info.name, arguments)
                print(f"--- 执行工具并返回结果 ---")
                print(f"工具执行结果: {tool_output}\n")
                
                # 将工具结果添加到消息历史
                messages.append({
                    "role": "tool", 
                    "tool_call_id": tool_call.id, 
                    "content": tool_output
                })
            
            # 第二次调用：将工具结果返回给模型，获取最终回答
            print("--- 将工具结果返回给模型，获取最终答案 ---")
            final_message = send_messages(messages, tools=tools)
            print(f"Model> {final_message.content}")
            
        else:
            # 模型直接回答，无需工具调用
            print(f"Model> {message.content}")
            
    except Exception as e:
        print(f"❌ 对话处理失败: {e}")

if __name__ == "__main__":
    # 示例1：天气查询
    print("=" * 60)
    print("示例1: 天气查询")
    print("=" * 60)
    handle_conversation("杭州今天天气怎么样？", tools)
    
    print("\n" + "=" * 60)
    print("示例2: 时间查询")
    print("=" * 60)
    handle_conversation("现在几点了？", tools)
    
    print("\n" + "=" * 60)
    print("示例3: 直接问答（无需工具）")
    print("=" * 60)
    handle_conversation("你好啊！", tools)