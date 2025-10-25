from metagpt.ext.aflow.benchmark.benchmark import BaseBenchmark
from metagpt.ext.aflow.benchmark.drop import DROPBenchmark
from metagpt.ext.aflow.benchmark.gsm8k import GSM8KBenchmark
from metagpt.ext.aflow.benchmark.hotpotqa import HotpotQABenchmark
from metagpt.ext.aflow.benchmark.humaneval import HumanEvalBenchmark
from metagpt.ext.aflow.benchmark.math import MATHBenchmark
from metagpt.ext.aflow.benchmark.mbpp import MBPPBenchmark
import asyncio

BENCHMARK_MAPPING = {
    "gsm8k": GSM8KBenchmark,
    "math": MATHBenchmark,
    "humaneval": HumanEvalBenchmark,
    "hotpotqa": HotpotQABenchmark,
    "mbpp": MBPPBenchmark,
    "drop": DROPBenchmark,
}

async def create_benchmark(name: str, is_test: bool = False) -> BaseBenchmark:
    if is_test:
        file_path = f'metagpt/ext/aflow/data/{name.lower()}_test.jsonl'
    else:
        file_path = f'metagpt/ext/aflow/data/{name.lower()}_validate.jsonl'
        
    log_path = "logs"
    benchmark = BENCHMARK_MAPPING[name.lower()](name, file_path, log_path)
    await benchmark.load_and_save_data()
    return benchmark

async def main():
    benchmark_name = ['gsm8k', 'math', 'hotpotqa', 'mbpp', 'drop', 'humaneval']
    for benchmark in benchmark_name:
        benchmark_obj = await create_benchmark(benchmark)
        await benchmark_obj.load_and_save_data()
        
        print(f"Benchmark: {benchmark}, description: {benchmark_obj.get_description()}")

if __name__ == "__main__":
    asyncio.run(main())