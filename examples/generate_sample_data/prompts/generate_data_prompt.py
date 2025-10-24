TASK_DECOMPOSER_PROMPT = """
You are a Task Decomposer Agent responsible for breaking down a complex problem into a structured sequence of smaller subtasks.

Given:
- A main task description: {task}
- A predefined scenario (operator sequence) that dictates the logical order of execution: {scenario}
- A set of available operators with distinct roles and functions: {operators}

Your goal:
1. Decompose the main task into multiple **well-defined subtasks** (each handled by one or more operators).
2. Ensure that the decomposition strictly follows the **given scenario order** — do not change the order of operators.
3. For each subtask, clearly specify:
   - `subtask_id`
   - `operator` (which agent/operator should handle it)
   - `objective` (what it aims to achieve)
4. The total number of subtasks should align with the number of operators in the scenario.

Guidelines:
- Each subtask must be **concise**, **actionable**, and **independent** enough for its assigned operator to perform.
- Maintain **logical flow** between subtasks (data dependencies, reasoning continuity, etc.).
- Avoid redundancy or vague steps.
- The final decomposition should read like a **workflow plan** executable by the operator sequence.

Return in JSON format:
{{
    'detailed_plan': [
        {{
            'subtask_id': 'subtask_<id>',
            'objective': '<objective>',
            'operator': '<operator>'
        }}
    ]
}}
"""

MAS_CODE_GENERATOR_PROMPT = """
You are an expert Python code generator for Multi-Agent Systems (MAS).

Your task is to generate a complete Python function that implements a **multi-agent workflow** according to the given inputs.

Inputs:
- MAS Template: {template}
- Plan: {plan}
  (A list of subtasks, each including `objective` and the `operator` that performs it.)
- Scenario: {scenario}
  (Defines the control flow and logical order between operators — may include sequential, conditional, or looping structures.)
- Operator Description: {operator_description}
  (Specifies each operator’s name, arguments, and behavior.)

Your goals:
1. Use the provided **MAS Template** as the structural base (e.g., class definition, init function, and call method).
2. Implement all subtasks listed in **Plan**, ensuring each operator call:
   - Matches its definition and argument signature in `operator_description`.
   - Appears in the correct order and control flow according to `Scenario`.
   - Includes the objectives of each subtask into the corresponding code snippet (you could inject them into the arguments of the operator call).
3. Integrate **control flow logic** (IF-ELSE, loops, or branching) that reflects the scenario’s description precisely.
4. Correctly await async operators and propagate their outputs (e.g., `solution = await self.custom(input=..., instruction=...)`).
5. Ensure all variable dependencies between subtasks are handled consistently (outputs from previous steps are used as inputs to later steps where appropriate).
6. Return the final result in a clean, formatted structure (e.g., `solution['response']` or equivalent).

Code requirements:
- The generated code must be syntactically correct and executable as a standalone Python module.
- Each operator call should use the correct attributes (e.g., `self.custom`, `self.answer_generate`, etc.) from the imported operator set.
- Do not include placeholder text — fill in all necessary fields with meaningful values or contextual data.
- Add concise docstrings explaining the workflow purpose and flow.
- All the libraries / frameworks were imported. So you only need to implement the Workflow class.

Output format:
Only output the **Python code** of the complete `Workflow` class (or equivalent function).
Do NOT include explanations or markdown formatting.

The output must startswith: 
class Workflow:
...
"""

COMPLETE_MAS_TEMPLATE = """
from typing import Literal
import metagpt.ext.aflow.scripts.operator as operator
from metagpt.provider.llm_provider_registry import create_llm_instance
from metagpt.configs.models_config import ModelsConfig
import asyncio

DatasetType = Literal["HumanEval", "MBPP", "GSM8K", "MATH", "HotpotQA", "DROP"]

{workflow_class}
"""