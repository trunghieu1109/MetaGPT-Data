import re
import string
from collections import Counter
from typing import Callable, List, Tuple

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from metagpt.ext.aflow.benchmark.benchmark import BaseBenchmark
from metagpt.logs import logger


class DROPBenchmark(BaseBenchmark):
    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)

    def normalize_answer(self, s: str) -> List[str]:
        """
        Normalize answers for evaluation.
        """

        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def calculate_score(self, ground_truth: str, prediction: str) -> Tuple[float, str]:
        """
        Compute the F1 score between prediction and ground truth answers.
        """
        prediction_tokens = self.normalize_answer(prediction).split()
        ground_truth_tokens = self.normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0, prediction
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1, prediction

    @retry(stop=stop_after_attempt(5), wait=wait_fixed(1), retry=retry_if_exception_type(Exception), reraise=True)
    async def _generate_output(self, graph, input_text):
        return await graph(input_text)

    async def evaluate_problem(self, problem: dict, graph: Callable) -> Tuple[str, str, str, float]:
        input_text = problem["context"]
        expected_output = problem["ref_text"]
        answers = expected_output.split("|")

        try:
            output, logs = await self._generate_output(graph, input_text)
            f1_scores = []

            for answer in answers:
                if answer.strip() != "":
                    output_parts = output.split("|")
                    for output_part in output_parts:
                        f1_score, _ = self.calculate_score(answer, output_part)
                        f1_scores.append(f1_score)

            uni_score = max(f1_scores)

            if uni_score < 0.3:
                self.log_mismatch(input_text, expected_output, output, output)

            return input_text, output, expected_output, uni_score, logs

        except Exception as e:
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}")
            return input_text, str(e), expected_output, 0.0, f"Error: {e}"

    def get_result_columns(self) -> List[str]:
        return ["task", "output", "expected_output", "score", "execution_logs"]

    def get_description(self):
        general_desc = """
The DROP (Discrete Reasoning Over Paragraphs) benchmark challenges models to perform multi-step discrete reasoning on text. Given a paragraph and a complex question, the model must extract relevant information and execute operations such as addition, subtraction, counting, sorting, or comparison to generate the correct answer. The task requires understanding both the semantics of the paragraph and the logical structure of the question, often necessitating event coreference resolution and numerical reasoning over multiple data points."
This description highlights the core challenges and reasoning skills needed for the benchmark, helping an agent focus on strategies to handle language understanding combined with symbolic-like numeric processing.
        """
        
        sample_data = [f"Question and context: {sample['context']}. Possible Answer: {sample['ref_text']}" for sample in self.val_data[:3]]
        
        return general_desc + "\n\n" + "\n\n".join(sample_data)