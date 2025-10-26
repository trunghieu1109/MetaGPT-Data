from queue import Queue
import argparse
import sys
import os
import json
import random
import pandas as pd
from typing import Dict, List
import asyncio
import ast

from metagpt.configs.models_config import ModelsConfig
from metagpt.configs.llm_config import LLMConfig
from metagpt.actions.action_node import ActionNode
from metagpt.ext.aflow.scripts.evaluator import Evaluator
from metagpt.provider.llm_provider_registry import create_llm_instance
from metagpt.ext.aflow.scripts.operator_an import (
    GeneratePlanOp,
    GenerateMASOp
)

from meta_constants import OPERATORS_LIST, MAS_TEMPLATE, CODING_MAS_TEMPLATE
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
        default="examples/generate_sample_data/generated_data",
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
    
    parser.add_argument(
        "--max_scenario_lens",
        type=int,
        default=15,
        help="Specifies the max length of a scenario",
    )
    
    parser.add_argument(
        "--start_scenario_idx",
        type=int,
        default=0,
        help="Specifies the start scenario index",
    )
    
    parser.add_argument(
        "--end_scenario_idx",
        type=int,
        default=5,
        help="Specifies the end scenario index",
    )
    
    parser.add_argument(
        "--n_sample",
        type=int,
        default=1,
        help="Specifies number of samples for each scenario",
    )
    
    return parser.parse_args()

class DataGenerator:
    def __init__(self, gen_model: LLMConfig, exec_model: LLMConfig, dataset: str, save_path: str, sample_id: int, 
                 max_scenario_len: int = 10, max_scenarios: int = 5, start_scenario_idx: int = 0, end_scenario_idx: int = 5):
        self.gen_model_config = gen_model
        self.exec_model_config = exec_model
        self.avail_operators = [OPERATORS_LIST[op] for op in EXPERIMENT_CONFIGS[dataset].operators]
        self.max_scenario_len = max_scenario_len
        self.dataset = dataset
        self.sample_id = sample_id
        self.max_scenarios = max_scenarios
        self.scenarios = Queue(maxsize=self.max_scenarios)
        self.save_path = os.path.join(save_path, self.dataset.lower(), f"sample_{self.sample_id}")
        self.full_scenario_path = os.path.join(save_path, self.dataset.lower(), "scenario_list.json")
        self.start_scenario_idx = start_scenario_idx
        self.end_scenario_idx = end_scenario_idx
        self.scenario_list = []
        
    async def create(self):
        self.gen_llm = create_llm_instance(self.gen_model_config)
        self.exec_llm = create_llm_instance(self.exec_model_config)
        self.benchmark = await create_benchmark(self.dataset)
    
    async def generate_all_scenario(self):
        directory = os.path.dirname(self.full_scenario_path)
        os.makedirs(directory, exist_ok=True)
        if os.path.exists(self.full_scenario_path):
            with open(self.full_scenario_path, "r", encoding="utf-8") as f:
                return json.load(f)
        # TODO: From the available operators, generate all scenarios that have exactly max_scenario_len steps, each step is addressed by only one operator.
        # Iterate each combinations of operators, generate all the possible scenarios. consider the relationships and constraints between operators.
        graph = self.benchmark.create_graph()
        paths = []
        formatted_paths = []
        for path in graph.find_paths("P", "Q", curr_len=0, num_cag = 0, num_agc = 0, max_lens = self.max_scenario_len):
            if 'Format' in path:
                formatted_paths.append(" -> ".join(path))
            else:
                paths.append(" -> ".join(path))
        
        # resample format-contained mas (because it has small impact on model performance)
        formatted_paths = random.sample(formatted_paths, min(len(formatted_paths), 2000))
        paths = paths + formatted_paths
            
        output_data = {
            "total_paths": len(paths),
            "paths": paths,
        }
        
        with open(self.full_scenario_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
            
        return output_data

    async def _generate_plan(self, task: str, scenario: str):
        prompt = TASK_DECOMPOSER_PROMPT.format(task=task, scenario=scenario, operators=self.avail_operators)
        llm = self.gen_llm
        
        # TODO: Generate task decomposition in exactly max_scenario_len steps
        response = await invoking(GeneratePlanOp, prompt, llm)
        return response['detailed_plan']
    async def _generate_mas(self, scenario: str, plan: dict):
        if self.dataset in ['MBPP', 'HumanEval']:
            prompt = MAS_CODE_GENERATOR_PROMPT.format(template=CODING_MAS_TEMPLATE, scenario=scenario, operator_description=self.avail_operators, plan=plan)
        else:
            prompt = MAS_CODE_GENERATOR_PROMPT.format(template=MAS_TEMPLATE, scenario=scenario, operator_description=self.avail_operators, plan=plan)
        llm = self.gen_llm
        
        # TODO: Generate task decomposition in exactly max_scenario_len steps
        response = await invoking(GenerateMASOp, prompt, llm)
        return response['mas_code']
        
    async def _generate(self, scenario_idx: int):
        # TODO: Consider each scenario, generate the corresponding mas in form of an executable Python code function
        task = self.benchmark.get_description()
        
        scenario_path = os.path.join(self.save_path, f"scenario_{scenario_idx}", "scenario.txt")
        plan_path = os.path.join(self.save_path, f"scenario_{scenario_idx}", "plan.json")
        mas_path = os.path.join(self.save_path, f"scenario_{scenario_idx}", "graph.py")
        
        scenario, plan, mas = "", "", ""
        
        # check scenario
        if os.path.exists(scenario_path):
            with open(scenario_path, "r", encoding="utf-8") as file:
                scenario = file.read()
                
        else:
            # get the scenario
            scenario = self.scenario_list[scenario_idx]
            
            directory = os.path.dirname(scenario_path)
            os.makedirs(directory, exist_ok=True)
            
            with open(scenario_path, "w", encoding="utf-8") as file:
                file.write(scenario)
        scenario = "Start -> Custom -> Format -> End"
        print(scenario)
        
        # check plan     
        if os.path.exists(plan_path):
            with open(plan_path, "r", encoding="utf-8") as file:
                plan = json.load(file)
                
        else:        
            # generate the task decomposition in this scenario
            plan = await self._generate_plan(task, scenario)
            plan = plan.strip("```")
            plan = plan.strip("```json")
            
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
    
    async def _execute(self, scenario_idx, exec_code):
        # TODO: Execute generated mas, log all the information as much as possible
        log_path = os.path.join(self.save_path, f"scenario_{scenario_idx}")
        new_data_path = os.path.join(self.save_path, f"scenario_{scenario_idx}", "data.jsonl")
        
        # init evaluator
        evaluator = Evaluator(eval_path=log_path)
        mas_path = os.path.join(self.save_path, f"scenario_{scenario_idx}")
        workflows_path = mas_path.replace("\\", ".").replace("/", ".")
        graph_module_name = f"{workflows_path}.graph"
        
        # load mas as graph
        try:
            graph_module = __import__(graph_module_name, fromlist=[""])
            graph_class = getattr(graph_module, "Workflow")
            
        except ImportError as e:
            logger.info(f"Error loading graph for sample_id: {self.sample_id}, dataset: {self.dataset}, error : {e}")
            raise
        
        # execute and then evaluate code -> get the logs and labels for each sample
        score, output_file = await evaluator.graph_evaluate(
            self.dataset,
            graph_class,
            {"dataset": self.dataset, "llm_config": self.exec_model_config},
            log_path,
            # eval_list=self.benchmark.get_lower_accuracy_data(), 
            eval_list=None,
            is_test=False,
        )
        
        results = pd.read_csv(output_file, encoding="utf-8")
        results['code'] = ""
        for idx, row in results.iterrows():
            results.at[idx, 'code'] = exec_code
        
        with open(new_data_path, "w", encoding="utf-8") as f:
            for _, row in results.iterrows():
                json_line = json.dumps(row.to_dict(), ensure_ascii=False)
                f.write(json_line + "\n")
    
    async def generate_sample_data(self):
        scenario_list = await self.generate_all_scenario()
        self.scenario_list = scenario_list['paths']
        
        print(f"Total scenarios: {scenario_list['total_paths']}")
        
        for idx, scenario in enumerate(self.scenario_list):
            if idx < self.start_scenario_idx or idx > self.end_scenario_idx:
                continue
            # generate mas
            scenario_, plan, mas = await self._generate(idx)
            # execute mas
            await self._execute(idx, mas)

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
    gen_model = models_config.get(args.gen_model)
    exec_model = models_config.get(args.exec_model)
    
    for i in range(args.n_sample):
        generator = DataGenerator(
            gen_model, 
            exec_model,
            args.dataset, 
            args.results_path, 
            i, 
            max_scenario_len=args.max_scenario_lens, 
            start_scenario_idx=args.start_scenario_idx, 
            end_scenario_idx=args.end_scenario_idx
        )
        await generator.create()
        await generator.generate_sample_data()
    
if __name__ == "__main__":
    asyncio.run(main())