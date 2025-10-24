from queue import Queue
import argparse
import sys
import os
import json
from typing import Dict, List
import asyncio

from metagpt.configs.models_config import ModelsConfig
from metagpt.configs.llm_config import LLMConfig
from metagpt.actions.action_node import ActionNode
from metagpt.ext.aflow.scripts.evaluator import Evaluator
from metagpt.provider.llm_provider_registry import create_llm_instance
from metagpt.ext.aflow.scripts.operator_an import (
    GeneratePlanOp,
    GenerateMASOp
)

from meta_constants import OPERATORS_LIST, MAS_TEMPLATE
from prompts.generate_data_prompt import TASK_DECOMPOSER_PROMPT, MAS_CODE_GENERATOR_PROMPT, COMPLETE_MAS_TEMPLATE
from experiment_config import ExperimentConfig, EXPERIMENT_CONFIGS
from benchmark_loader import create_benchmark
from metagpt.logs import logger

def exit():
    sys.exit()

async def invoking(output_format, prompt, llm):
    fill_kwargs = {
        'context': prompt,
        'llm': llm,
        'mode': 'single_fill'
    }
    response = await ActionNode.from_pydantic(output_format).fill(**fill_kwargs)
    return response.instruct_content.model_dump()

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
    def __init__(self, gen_model: LLMConfig, exec_model: LLMConfig, dataset: str, save_path: str, sample_id: int, max_scenario_len: int = 5, max_scenarios: int = 5):
        self.gen_model_config = gen_model
        self.exec_model_config = exec_model
        self.avail_operators = [OPERATORS_LIST[op] for op in EXPERIMENT_CONFIGS[dataset].operators]
        self.max_scenario_len = max_scenario_len
        self.dataset = dataset
        self.sample_id = sample_id
        self.max_scenarios = max_scenarios
        self.scenarios = Queue(maxsize=self.max_scenarios)
        self.save_path = os.path.join(save_path, self.dataset.lower(), f"sample_{self.sample_id}")
        
    async def create(self):
        self.gen_llm = create_llm_instance(self.gen_model_config)
        self.exec_llm = create_llm_instance(self.exec_model_config)
        self.benchmark = await create_benchmark(self.dataset)
        
    async def _generate_scenario(self):
        # TODO: From the available opearators, generate one scenario that has exactly max_scenario_len steps, and different from the existing scenarios
        return """
Custom
for n = 3:
    AnswerGenerate
ScEnsemble
Review
Revise
Format
        """
    
    async def generate_all_scenario(self):
        # TODO: From the available operators, generate all scenarios that have exactly max_scenario_len steps, each step is addressed by only one operator.
        # Iterate each combinations of operators, generate all the possible scenarios. consider the relationships and constraints between operators.
        for i in range(self.max_scenarios):
            self.scenarios.put(await self._generate_scenario())
    
    async def _generate_plan(self, task: str, scenario: str):
        prompt = TASK_DECOMPOSER_PROMPT.format(task=task, scenario=scenario, operators=self.avail_operators)
        llm = self.gen_llm
        
        # TODO: Generate task decomposition in exactly max_scenario_len steps
        response = await invoking(GeneratePlanOp, prompt, llm)
        return response['detailed_plan']
    async def _generate_mas(self, scenario: str, plan: dict):
        prompt = MAS_CODE_GENERATOR_PROMPT.format(template=MAS_TEMPLATE, scenario=scenario, operator_description=self.avail_operators, plan=plan)
        llm = self.gen_llm
        
        # TODO: Generate task decomposition in exactly max_scenario_len steps
        response = await invoking(GenerateMASOp, prompt, llm)
        return response['mas_code']
        
    async def _generate(self):
        # TODO: Consider each scenario, generate the corresponding mas in form of an executable Python code function
        task = self.benchmark.get_description()
        
        scenario_path = os.path.join(self.save_path, "scenario.txt")
        plan_path = os.path.join(self.save_path, "plan.json")
        mas_path = os.path.join(self.save_path, "graph.py")
        
        scenario, plan, mas = "", "", ""
        
        # check scenario
        if os.path.exists(scenario_path):
            with open(scenario_path, "r", encoding="utf-8") as file:
                scenario = file.read()
                
        else:
            # get the scenario
            scenario = self.scenarios.get()
            
            directory = os.path.dirname(scenario_path)
            os.makedirs(directory, exist_ok=True)
            
            with open(scenario_path, "w", encoding="utf-8") as file:
                file.write(scenario)
        
        # check plan     
        if os.path.exists(plan_path):
            with open(plan_path, "r", encoding="utf-8") as file:
                plan = json.load(file)
                
        else:        
            # generate the task decomposition in this scenario
            plan = await self._generate_plan(task, scenario)
            
            directory = os.path.dirname(plan_path)
            os.makedirs(directory, exist_ok=True)
            
            with open(plan_path, "w", encoding="utf-8") as file:
                json.dump(plan, file, ensure_ascii=False, indent=4)
          
        # check mas  
        if os.path.exists(mas_path):
            with open(mas_path, "r", encoding="utf-8") as file:
                mas = file.read()
                
        else:
            # generate the corresponding mas
            mas = await self._generate_mas(scenario, plan)
            
            # postprocess mas
            mas = COMPLETE_MAS_TEMPLATE.format(workflow_class=mas)

            directory = os.path.dirname(mas_path)
            os.makedirs(directory, exist_ok=True)
                
            with open(mas_path, "w") as f:
                f.write(mas)
            
        # return scenario, plan and mas
        return scenario, plan, mas
    
    async def _execute(self, exec_code):
        # TODO: Execute generated mas, log all the information as much as possible
        log_path = os.path.join(self.save_path)
        
        # init evaluator
        evaluator = Evaluator(eval_path=log_path)
        
        workflows_path = self.save_path.replace("\\", ".").replace("/", ".")
        graph_module_name = f"{workflows_path}.graph"
        
        # load mas as graph
        try:
            graph_module = __import__(graph_module_name, fromlist=[""])
            graph_class = getattr(graph_module, "Workflow")
            
        except ImportError as e:
            logger.info(f"Error loading graph for sample_id: {self.sample_id}, dataset: {self.dataset}, error : {e}")
            raise
        
        # execute and then evaluate code -> get the logs and labels for each sample
        score = await evaluator.graph_evaluate(
            self.dataset,
            graph_class,
            {"dataset": self.dataset, "llm_config": self.exec_model_config},
            log_path,
            is_test=False,
        )
    
    async def generate_sample_data(self):
        await self.generate_all_scenario()
        
        # generate mas
        scenario, plan, mas = await self._generate()
        # execute mas
        await self._execute(mas)
        
        # synthesize the data, include: MAS, input, output of each agent / operator, instruction, role, reasoning process, the label (True / False)
        pass

async def main():
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
    
    models_config = ModelsConfig.default()
    gen_model = models_config.get("openai/gpt-oss-20b")
    exec_model = models_config.get("openai/gpt-oss-20b")
    
    n_sample = 3
    
    for i in range(n_sample):
        generator = DataGenerator(gen_model, exec_model, args.dataset, "examples/generate_sample_data/generated_data", i)
        await generator.create()
        await generator.generate_sample_data()
    
if __name__ == "__main__":
    asyncio.run(main())