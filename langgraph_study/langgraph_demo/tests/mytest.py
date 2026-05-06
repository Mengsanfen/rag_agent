from langgraph_sdk import get_client
import asyncio

# 调用agent发布的api接口
client = get_client(url="http://localhost:2024")

async def main():
    async for chunk in client.runs.stream(
        None, # Threadless run
        "agent", # Name of assistant. Defined in langgraph.json
        input = {
            "message": [{
                "role": "human",
                "content": "给当前用户一个祝福"
            }],
        },
    ):
        print(f"Receving new event of type: {chunk.event}...")
        print(chunk.data)
        print('\n\n')

if __name__ == "__main__":
    asyncio.run(main())