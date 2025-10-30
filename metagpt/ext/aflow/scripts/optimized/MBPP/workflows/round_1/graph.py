# -*- coding: utf-8 -*-
# @Date    : 6/27/2024 22:07 PM
# @Author  : didi
# @Desc    : Basic Graph Class

from typing import Literal
import metagpt.ext.aflow.scripts.optimized.DROP.workflows.template.operator as operator
import metagpt.ext.aflow.scripts.optimized.DROP.workflows.round_1.prompt as prompt_custom
from metagpt.provider.llm_provider_registry import create_llm_instance
from metagpt.utils.cost_manager import CostManager

DatasetType = Literal["HumanEval", "MBPP", "GSM8K", "MATH", "HotpotQA", "DROP"]

class Workflow:
    def __init__(
        self,
        name: str,
        llm_config,
        dataset: DatasetType,
    ) -> None:
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)

    async def __call__(self, problem: str, entry_point: str):
        """
        Implementation of the workflow
        """
        invoking_logs = []
        
        solution, logs = await self.custom(input=problem, instruction="")
        invoking_logs.append(logs)
        
        return solution['response'], invoking_logs
