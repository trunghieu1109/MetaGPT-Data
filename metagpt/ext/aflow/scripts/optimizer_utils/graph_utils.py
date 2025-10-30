import json
import os
import re
import time
import sys
import traceback
from typing import List
import importlib.util
import os
import shutil
import errno

from metagpt.ext.aflow.scripts.prompts.optimize_prompt import (
    WORKFLOW_CUSTOM_USE,
    WORKFLOW_INPUT,
    WORKFLOW_OPTIMIZE_PROMPT,
    WORKFLOW_TEMPLATE,
    WORKFLOW_TEST_TEMPLATE
)
from metagpt.logs import logger


class GraphUtils:
    def __init__(self, root_path: str):
        self.root_path = root_path

    def create_round_directory(self, graph_path: str, round_number: int) -> str:
        directory = os.path.join(graph_path, f"round_{round_number}")
        os.makedirs(directory, exist_ok=True)
        return directory

    def load_graph(self, round_number: int, workflows_path: str):
        workflows_path = workflows_path.replace("\\", ".").replace("/", ".")
        graph_module_name = f"{workflows_path}.round_{round_number}.graph"

        try:
            graph_module = __import__(graph_module_name, fromlist=[""])
            graph_class = getattr(graph_module, "Workflow")
            return graph_class
        except ImportError as e:
            logger.info(f"Error loading graph for round {round_number}: {e}")
            raise

    def read_graph_files(self, round_number: int, workflows_path: str):
        prompt_file_path = os.path.join(workflows_path, f"round_{round_number}", "prompt.py")
        graph_file_path = os.path.join(workflows_path, f"round_{round_number}", "graph.py")

        try:
            with open(prompt_file_path, "r", encoding="utf-8") as file:
                prompt_content = file.read()
            with open(graph_file_path, "r", encoding="utf-8") as file:
                graph_content = file.read()
        except FileNotFoundError as e:
            logger.info(f"Error: File not found for round {round_number}: {e}")
            raise
        except Exception as e:
            logger.info(f"Error loading prompt for round {round_number}: {e}")
            raise
        return prompt_content, graph_content

    def extract_solve_graph(self, graph_load: str) -> List[str]:
        pattern = r"class Workflow:.+"
        return re.findall(pattern, graph_load, re.DOTALL)

    def load_operators_description(self, operators: List[str]) -> str:
        path = f"{self.root_path}/workflows/template/operator.json"
        operators_description = ""
        for id, operator in enumerate(operators):
            operator_description = self._load_operator_description(id + 1, operator, path)
            operators_description += f"{operator_description}\n"
        return operators_description

    def _load_operator_description(self, id: int, operator_name: str, file_path: str) -> str:
        with open(file_path, "r") as f:
            operator_data = json.load(f)
            matched_data = operator_data[operator_name]
            desc = matched_data["description"]
            interface = matched_data["interface"]
            return f"{id}. {operator_name}: {desc}, with interface {interface})."

    def create_graph_optimize_prompt(
        self,
        experience: str,
        score: float,
        graph: str,
        prompt: str,
        operator_description: str,
        type: str,
        log_data: str,
    ) -> str:
        graph_input = WORKFLOW_INPUT.format(
            experience=experience,
            score=score,
            graph=graph,
            prompt=prompt,
            operator_description=operator_description,
            type=type,
            log=log_data,
        )
        graph_system = WORKFLOW_OPTIMIZE_PROMPT.format(type=type)
        return graph_input + WORKFLOW_CUSTOM_USE + graph_system

    async def get_graph_optimize_response(self, graph_optimize_node):
        max_retries = 5
        retries = 0

        while retries < max_retries:
            try:
                response = graph_optimize_node.instruct_content.model_dump()
                return response
            except Exception as e:
                retries += 1
                logger.info(f"Error generating prediction: {e}. Retrying... ({retries}/{max_retries})")
                if retries == max_retries:
                    logger.info("Maximum retries reached. Skipping this sample.")
                    break
                traceback.print_exc()
                time.sleep(5)
        return None

    def write_graph_files(self, directory: str, response: dict, round_number: int, dataset: str):
        graph_ = response["graph"].strip("```")
        graph_ = response["graph"].strip("```python")
        graph = WORKFLOW_TEMPLATE.format(graph=graph_, round=round_number, dataset=dataset)

        with open(os.path.join(directory, "graph.py"), "w", encoding="utf-8") as file:
            file.write(graph)

        with open(os.path.join(directory, "prompt.py"), "w", encoding="utf-8") as file:
            file.write(response["prompt"])

        with open(os.path.join(directory, "__init__.py"), "w", encoding="utf-8") as file:
            file.write("")

    async def check_graph_syntax(self, llm_config, directory: str, response: dict, round_number: int, dataset: str) -> bool:
        
        logger.info("Checking graph syntax...")
        
        graph_ = response["graph"].strip("```")
        graph_ = response["graph"].strip("```python")
        graph = WORKFLOW_TEST_TEMPLATE.format(graph=graph_, round=round_number, dataset=dataset)
    
        os.makedirs(os.path.join(directory, "test"), exist_ok=True)

        with open(os.path.join(directory, "test", "graph.py"), "w", encoding="utf-8") as file:
            file.write(graph)

        with open(os.path.join(directory, "test", "prompt.py"), "w", encoding="utf-8") as file:
            file.write(response["prompt"])

        with open(os.path.join(directory, "test", "__init__.py"), "w", encoding="utf-8") as file:
            file.write("")
            
        def unload_previous_modules():
            # Các tên module có thể tồn tại từ lần load trước
            # print(sys.modules.keys())
            candidates = [k for k in sys.modules.keys() if "test" in k and ("graph" in k or "prompt" in k)]
            for name in candidates:
                sys.modules.pop(name, None)
                logger.info(f"Unloaded cached module: {name}")
            
        unload_previous_modules()
            
        # is_countinue = input("Wait for 10 seconds to allow file system to sync. Continue? (y/n): ")
        # if is_countinue.lower() == 'n':
        #     sys.exit()
            
        # time.sleep(10)
        
        def load_module_from_path(name: str, path: str):
            spec = importlib.util.spec_from_file_location(name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            return module
            
        # sys.exit()
        try:
            module_path = os.path.join(directory, "test", "graph.py")
            module_name = "test_graph_dynamic"
            
            module_path_prompt = os.path.join(directory, "test", "prompt.py")
            module_name_prompt = os.path.join(directory, "test").replace("/", ".") + ".prompt"
            print("Module prompt name: ", module_name_prompt)

            logger.info(f"Loading updated graph module from file: {module_path}")
            module_prompt = load_module_from_path(module_name_prompt, module_path_prompt)
            module = load_module_from_path(module_name, module_path)
            workflow = module.Workflow(name="TestWorkflow", llm_config=llm_config, dataset="DROP")
            
            results = ""
            logger.info("Executing test workflow to validate graph syntax...")
            results = await workflow("Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop.")
            logger.info("Graph syntax is correct.")
        except Exception as e:
            logger.error(f"Graph syntax error: {e}")
            return False
        return True
