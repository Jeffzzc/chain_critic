import asyncio
import json
import os
import aiohttp
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

# ================= 配置区域 =================

# 1. 端口配置 (8000 - 8007)
PORTS = range(8000, 8008)
BASE_URLS = [f"http://localhost:{port}/v1" for port in PORTS]

# 2. 文件路径
INPUT_FILE = "/vepfs/algorithm-multimodal-arch/zuozecheng-jk/ChainCritic/datasets/QwQ-LongCoT-130K/train_random_sample.jsonl"
OUTPUT_FILE = "/vepfs/algorithm-multimodal-arch/zuozecheng-jk/ChainCritic/datasets/QwQ-LongCoT-130K/Qwen3-0.6B_result.jsonl"

# 3. 模型配置
MODEL_NAME = "Qwen3-0.6B"

# 4. 推理参数
MAX_TOKENS = 1024
TEMPERATURE = 0.0

# 5. [关键优化] 并发控制
# 0.6B 模型很小，吞吐量很高，建议每张卡至少给 32-50 个并发，才能吃满显卡
CONCURRENCY_PER_GPU = 32

# ===========================================

async def wait_for_servers():
    """在开始推理前，确保所有端口都已就绪"""
    print("正在检查服务状态 (Waiting for servers to be ready)...")
    async with aiohttp.ClientSession() as session:
        for port in PORTS:
            url = f"http://localhost:{port}/v1/models"
            while True:
                try:
                    async with session.get(url) as resp:
                        if resp.status == 200:
                            # print(f"Port {port} is ready.")
                            break
                except aiohttp.ClientConnectorError:
                    pass
                except Exception:
                    pass
                
                # 如果没通，稍微等一下
                print(f"Waiting for Port {port}...", end="\r")
                await asyncio.sleep(2)
    print("\n所有服务已就绪，开始推理！\n")

async def fetch_response(client, prompt, retry=3):
    """调用 VLLM 接口获取回复，带重试机制"""
    for attempt in range(retry):
        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt == retry - 1:
                print(f"\n[Error] Port {client.base_url} failed after retries: {e}")
                return None
            await asyncio.sleep(1) # 出错后小睡一下再重试

async def worker(client, input_queue, writer_queue, pbar):
    """消费者 Worker"""
    while True:
        try:
            # 从队列获取任务
            item = input_queue.get_nowait()
        except asyncio.QueueEmpty:
            break

        original_data = item
        question = original_data.get("question", "")
        
        full_prompt = f"{question}\nPlease reason step by step and put your final answer within \\boxed{{}}."

        answer = await fetch_response(client, full_prompt)

        if answer:
            final_record = {
                "question": question,
                "answer": answer
            }
            await writer_queue.put(final_record)
        
        pbar.update(1)
        input_queue.task_done()

async def file_writer(output_file, writer_queue):
    """单独的写入协程，避免文件锁竞争"""
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "a", encoding="utf-8") as f:
        while True:
            item = await writer_queue.get()
            if item is None: # 结束信号
                break
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            f.flush()
            writer_queue.task_done()

async def main():
    if not os.path.exists(INPUT_FILE):
        print(f"错误：找不到输入文件 {INPUT_FILE}")
        return

    # === 步骤 1: 等待服务启动 ===
    await wait_for_servers()

    # === 步骤 2: 读取数据 ===
    print(f"正在读取数据集: {INPUT_FILE} ...")
    tasks_data = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    tasks_data.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    
    # === 步骤 3: 断点续传检查 ===
    completed_questions = set()
    if os.path.exists(OUTPUT_FILE):
        print(f"检测到输出文件 {OUTPUT_FILE}，正在检查已完成任务...")
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "question" in data:
                        completed_questions.add(data["question"]) 
                except:
                    pass
        print(f"已跳过 {len(completed_questions)} 条已完成任务。")

    input_queue = asyncio.Queue()
    writer_queue = asyncio.Queue()

    tasks_to_run = 0
    for data in tasks_data:
        q = data.get("question")
        if q and q not in completed_questions:
            input_queue.put_nowait(data)
            tasks_to_run += 1
            
    if tasks_to_run == 0:
        print("所有任务已完成！")
        return

    # === 步骤 4: 启动 Workers ===
    # 初始化 8 个 Client
    clients = [AsyncOpenAI(api_key="EMPTY", base_url=url) for url in BASE_URLS]
    
    # 启动写入线程
    writer_task = asyncio.create_task(file_writer(OUTPUT_FILE, writer_queue))

    print(f"开始并行推理 (Model: {MODEL_NAME})")
    print(f"总任务数: {tasks_to_run} | GPU数量: {len(clients)} | 单卡并发: {CONCURRENCY_PER_GPU}")
    
    pbar = tqdm(total=tasks_to_run, desc="Inference Progress")
    
    workers = []
    # 关键修改：为每个 Client 启动多个 Worker
    # 这样可以保证每张卡时刻都有 CONCURRENCY_PER_GPU 个请求在排队，最大化利用 vLLM 的 batching
    for client in clients:
        for _ in range(CONCURRENCY_PER_GPU):
            workers.append(asyncio.create_task(worker(client, input_queue, writer_queue, pbar)))

    await asyncio.gather(*workers)
    
    # === 步骤 5: 收尾 ===
    await writer_queue.put(None) # 发送结束信号给 writer
    await writer_task
    
    pbar.close()
    
    # 关闭 client 连接（虽然脚本结束会自动关闭，但显式关闭是好习惯）
    for client in clients:
        await client.close()

    print(f"\n推理完成！结果已保存至: {OUTPUT_FILE}")

if __name__ == "__main__":
    asyncio.run(main())

