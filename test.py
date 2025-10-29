from metagpt.logs import logger
import importlib.util
import os
from metagpt.ext.aflow.scripts.prompts.optimize_prompt import (
    WORKFLOW_CUSTOM_USE,
    WORKFLOW_INPUT,
    WORKFLOW_OPTIMIZE_PROMPT,
    WORKFLOW_TEMPLATE,
    WORKFLOW_TEST_TEMPLATE
)

import asyncio

async def test():
    directory = "/data/nguyentrunghieu/node-DeepResearch/MetaGPT-Data/metagpt/ext/aflow/scripts/optimized/DROP/workflows/round_2"

    import argparse
    from typing import Dict, List

    from metagpt.configs.models_config import ModelsConfig
    # from metagpt.ext.aflow.data.download_data import download
    from metagpt.ext.aflow.scripts.optimizer import Optimizer

    models_config = ModelsConfig.default()
    opt_llm_config = models_config.get("openai/gpt-oss-20b")

    try:
        path = os.path.join(directory, "graph.py")
        spec = importlib.util.spec_from_file_location("graph", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        workflow = module.Workflow(name="TestWorkflow", llm_config=opt_llm_config, dataset="DROP")

        results = ""
        logger.info("Executing test workflow to validate graph syntax...")
        results = await workflow("test")
        logger.info("Graph syntax is correct.")
    except Exception as e:
        logger.error(f"Graph syntax error: {e}")

asyncio.run(test())