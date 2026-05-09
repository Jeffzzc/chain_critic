# 导入list_providers函数
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.common.llm_client import list_providers

# 查看支持的运营商
providers = list_providers()
print("支持的运营商：", providers)