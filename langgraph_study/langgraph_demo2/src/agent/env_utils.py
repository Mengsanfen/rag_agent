import os
from dotenv import load_dotenv

load_dotenv(override=True)

DASHSCOPE_API_KEY=os.getenv('DASHSCOPE_API_KEY')
DASHSCOPE_BASE_URL=os.getenv('DASHSCOPE_BASE_URL')

DEEPSEEK_API_KEY=os.getenv('DEEPSEEK_API_KEY')
DEEPSEEK_BASE_URL=os.getenv('DEEPSEEK_BASE_URL')
TAVILY_API_KEY=os.getenv('TAVILY_API_KEY')

