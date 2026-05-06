from langchain_core.tools import tool

# class CalculateArgs(BaseModel):
#     a: float = Field(description="第一个需要输入的数字。")
#     b: float = Field(description="第二个需要输入的数字。")
#     operation: str = Field(description="运算类型, 只能是add、subtract、multiply和divide中任意一个")

@tool('calculate', parse_docstring=True)
def calculate3(a: float, b: float, operation: str) -> float:
    """工具函数：计算两个数字的运算结果
    
    Args:
        a: 第一个需要输入的数字。
        b: 第二个需要输入的数字。
        operation: 运算类型, 只能是add、subtract、multiply和divide中任意一个。

    Returns:
        返回两个输入数字的运算结果。

    """

    print(f"调用 calculate 工具， 第一个数字: {a}， 第二个数字: {b}, 运算类型：{operation}")

    result = 0.0
    match operation:
        case "add":
            result = a + b
        case "subtract":
            result = a - b
        case "multiply":
            result = a * b
        case "divide":
            if b != 0:
                result = a / b
            else:
                raise ValueError("除数不能为0")
            
    return result

print(calculate3.name)
print(calculate3.description)
print(calculate3.args)
print(calculate3.args_schema.model_json_schema())    # type: ignore