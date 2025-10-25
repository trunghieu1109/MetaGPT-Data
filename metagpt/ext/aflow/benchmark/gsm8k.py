# -*- coding: utf-8 -*-
# @Date    :
# @Author  : all
# @Desc    : test on gsm8k
import re
from typing import Callable, List, Optional, Tuple

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from metagpt.ext.aflow.benchmark.benchmark import BaseBenchmark
from metagpt.logs import logger


class GSM8KBenchmark(BaseBenchmark):
    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)

    def extract_number(self, text: str) -> Optional[float]:
        matches = re.findall(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?|\d+\.\d+", str(text))
        if matches:
            last_number = matches[-1].replace(",", "")
            try:
                return float(last_number)
            except ValueError:
                return None
        else:
            return None

    def calculate_score(self, expected_output: float, prediction: float) -> Tuple[float, float]:
        if prediction is None:
            return 0.0, prediction
        return 1.0 if abs(expected_output - prediction) <= 1e-6 else 0.0, prediction

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, graph, input_text):
        return await graph(input_text)

    async def evaluate_problem(self, problem: dict, graph: Callable) -> Tuple[str, str, float, float]:
        input_text = problem["question"]
        expected_output = self.extract_number(problem["answer"])

        try:
            output, logs = await self._generate_output(graph, input_text)
            predicted_number = self.extract_number(output)
            score, extracted_output = self.calculate_score(expected_output, predicted_number)

            if score == 0:
                self.log_mismatch(input_text, expected_output, output, extracted_output)

            return input_text, output, expected_output, score, logs

        except Exception as e:
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}")
            return input_text, str(e), expected_output, 0.0, f"Error: {e}"

    def get_result_columns(self) -> List[str]:
        return ["task", "output", "expected_output", "score", "execution_logs"]

    def get_raw_description(self):
        general_desc = """
Task: Solve grade school level math word problems with multi-step arithmetic reasoning.
Input: A math problem described as a word problem in natural language, involving quantities and conditions.
Output: A final numeric answer derived through reasoning across the problem text.
The benchmark checks the model’s ability to extract facts, reason logically, and perform calculations.
        """
        
        sample_data = [f"Question: {sample['question']}. Possible Answer: {sample['answer']}" for sample in self.val_data[:3]]
        
        return general_desc + "\n\n" + "\n\n".join(sample_data)
    
    def get_description(self):
        return """
* **Main Objective:**
  The goal of this task is to evaluate a model’s ability to **solve mathematically complex problems** stated in natural language, requiring *multi-step reasoning*, *symbolic manipulation*, and *conceptual understanding* across domains such as algebra, geometry, number theory, or arithmetic logic.

* **Input Format:**
  The input consists of a **natural-language math problem** describing a scenario that may involve relationships between quantities, geometric figures, equations, or logical constraints.
  Problems may include mathematical symbols, expressions, or diagrams that must be interpreted correctly as part of the reasoning process.

* **Output Format:**
  The output should be a **single final answer** — either a numerical value or a symbolic expression — representing the correct solution to the problem.
  No explanation, reasoning steps, or intermediate calculations are required in the output.

* **Additional Requirements:**

  * The answer must be **mathematically correct** and consistent with the conditions stated in the problem.
  * Problems may require performing **multi-step reasoning** such as algebraic manipulation, substitution, or geometric deduction.
  * The model should accurately interpret any **mathematical notations or contextual clues** in the text.
  * The final output must be **concise and precise**, containing only the result of the problem.
        """