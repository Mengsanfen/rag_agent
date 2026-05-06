from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig


@tool
def get_user_info_by_name(config: RunnableConfig) -> dict:
    """获取用户的所有信息，包括性别，年龄等."""
    user_name = config.get("configurable", {}).get("user_name", "zhangsan")
    print(f"获取用户{user_name}的所有信息")
    # 模拟
    return {'username': user_name, 'sex': '男', 'age': 18}