import re
import string
from collections import Counter
from typing import Callable, List, Tuple

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

from metagpt.ext.aflow.benchmark.benchmark import BaseBenchmark
from metagpt.logs import logger


class HotpotQABenchmark(BaseBenchmark):
    def __init__(self, name: str, file_path: str, log_path: str):
        super().__init__(name, file_path, log_path)

    def normalize_answer(self, s: str) -> str:
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

    async def evaluate_problem(self, problem: dict, graph: Callable) -> Tuple[str, str, str, str, float]:
        input_text = problem["question"]
        expected_output = problem["answer"]
        paragraphs = [item[1] for item in problem["context"] if isinstance(item[1], list)]
        context_str = "\n".join(" ".join(paragraph) for paragraph in paragraphs)
        inputs = f"Context: {context_str}\n\nQuestion: {input_text}\n\nAnswer:"

        try:
            output, logs = await self._generate_output(graph, inputs)
            score, extracted_output = self.calculate_score(expected_output, output)

            if (
                score < 0.3
            ):  # We set the threshold for collecting incorrect questions to 0.3, as F1 Score cannot be simply judged using 0-1
                self.log_mismatch(input_text, expected_output, output, extracted_output)

            return input_text, context_str, output, expected_output, score, logs

        except Exception as e:
            logger.info(f"Maximum retries reached. Skipping this sample. Error: {e}")
            return input_text, context_str, str(e), expected_output, 0.0, f"Error: {e}"

    def get_result_columns(self) -> List[str]:
        return ["question", "context", "prediction", "expected_output", "score", "execution_logs"]

    def get_description(self):
        general_desc = """
Task: Multi-hop question answering requiring reasoning over multiple paragraphs to answer a complex question.
Input: A natural language question and a set of supporting paragraphs (typically from Wikipedia).
Output: A text answer which can be a span from the paragraphs or a "yes/no" response, along with identified supporting sentences that justify the answer.
The model must aggregate relevant facts across documents and perform multi-step inference to generate both the answer and evidence sentences.
        """
        
        sample_data = [f"Context: {sample['context']}\n\nQuestion: {sample['question']}\n\n Sample Answer: {sample['answer']}" for sample in self.val_data[:3]]
        
        return general_desc + "\n\n" + "\n\n".join(sample_data)
        
        
        