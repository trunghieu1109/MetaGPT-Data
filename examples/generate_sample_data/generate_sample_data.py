# -*- coding: utf-8 -*-
# @Date    : 8/23/2024 20:00 PM
# @Author  : didi
# @Desc    : Entrance of AFlow.

import argparse
from typing import Dict, List

from metagpt.configs.models_config import ModelsConfig
from metagpt.configs.llm_config import LLMConfig
from metagpt.ext.aflow.data.download_data import download
from metagpt.ext.aflow.scripts.optimizer import Optimizer
from metagpt.actions.action_node import ActionNode
from metagpt.provider.llm_provider_registry import create_llm_instance
from metagpt.ext.aflow.scripts.operator_an import (
    AnswerGenerateOp,
    CodeGenerateOp,
    FormatOp,
    GenerateOp,
    MdEnsembleOp,
    ReflectionTestOp,
    ReviewOp,
    ReviseOp,
    ScEnsembleOp,
)

from meta_constants import OPERATORS_LIST, MAS_TEMPLATE
from experiment_config import ExperimentConfig, EXPERIMENT_CONFIGS

import asyncio

def parse_args():
    parser = argparse.ArgumentParser(description="AFlow Optimizer")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(EXPERIMENT_CONFIGS.keys()),
        required=True,
        help="Dataset type",
    )
    parser.add_argument("--sample", type=int, default=4, help="Sample count")
    parser.add_argument(
        "--results_path",
        type=str,
        default="generated_data",
        help="The generated data folders",
    )
    parser.add_argument(
        "--gen_model",
        type=str,
        default="openai/gpt-oss-20b",
        help="Specifies the model used to generate mas",
    )
    parser.add_argument(
        "--exec_model",
        type=str,
        default="openai/gpt-oss-20b",
        help="Specifies the model used to execute mas",
        
    )
    return parser.parse_args()

class DataGenerator:
    def __init__(self, gen_model: LLMConfig, exec_model: LLMConfig, dataset: str):
        self.gen_llm = create_llm_instance(gen_model)
        self.exec_llm = create_llm_instance(exec_model)
        self.dataset = dataset
        self.benchmark = create_benchmark(dataset)
        pass
    
    async def generate(self):
        pass


if __name__ == "__main__":
    args = parse_args()

    dataset_config = EXPERIMENT_CONFIGS[args.dataset]

    models_config = ModelsConfig.default()
    gen_model_config = models_config.get(args.gen_model)
    if gen_model_config is None:
        raise ValueError(
            f"The optimization model '{args.gen_model}' was not found in the 'models' section of the configuration file. "
            "Please add it to the configuration file or specify a valid model using the --gen_model flag. "
        )

    exec_model_config = models_config.get(args.exec_model)
    if exec_model_config is None:
        raise ValueError(
            f"The execution model '{args.exec_model}' was not found in the 'models' section of the configuration file. "
            "Please add it to the configuration file or specify a valid model using the --exec_model flag. "
        )

    # TODO: Generate data for this dataset
    step_num = 5
    problem = '''
Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop.
    '''
    
    operators_desc = [OPERATORS_LIST[op] for op in dataset_config.operators]
    
    prompt = f"""
The problem is:
{problem}

Decompose this problem into about {step_num} steps.
    """
    
    print(prompt)
    
    models_config = ModelsConfig.default()
    opt_llm_config = models_config.get("openai/gpt-oss-20b")
    llm = create_llm_instance(opt_llm_config)
    mode = 'single_fill'
    
    fill_kwargs = {"context": prompt, "llm": llm, "mode": mode}
    response = asyncio.run(ActionNode.from_pydantic(GenerateOp).fill(**fill_kwargs))
    
    print(response)