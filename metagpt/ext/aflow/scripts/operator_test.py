from typing import Literal
import metagpt.ext.aflow.scripts.operator as operator
from metagpt.provider.llm_provider_registry import create_llm_instance
from metagpt.configs.models_config import ModelsConfig
import asyncio

class OperatorTest: 
    def __init__(self):
        models_config = ModelsConfig.default()
        test_llm_config = models_config.get("openai/gpt-oss-20b")
        self.llm = create_llm_instance(test_llm_config)

    async def test_custom(self):
        custom_op = operator.Custom(self.llm)
        problem = """
Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop.
        """
        
        response = await custom_op(input=problem, instruction="Just return the answer, don't do anything else.")
        print(response)
    
    async def test_answer_generate(self):
        answer_generate_op = operator.AnswerGenerate(self.llm)
        input = """
Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop.
        """
        
        response = await answer_generate_op(input=input)
        print(response)
        
    async def test_custom_code_generate(self):
        custom_code_generate_op = operator.CustomCodeGenerate(self.llm)
        problem = """
from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n
        """
        
        entry_point = "has_close_elements"
        instruction = "Write the Python code function"
        
        response, reasoning = await custom_code_generate_op(problem=problem, entry_point=entry_point, instruction=instruction)
        
        test_op = operator.Test(self.llm)
        
        test_results, logs = await test_op(problem=problem, solution=response['code'], entry_point=entry_point)
        print(test_results['result'])
        print(test_results['solution'])
        
        for log in logs:
            print("------------------------------------")
            print(log)
        
    async def test_sc_ensemble(self):
        sc_ensemble_op = operator.ScEnsemble(self.llm)
        problem = """
Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop.
        """
        
        solutions = [
            "We set up equations for walking time at two speeds. Let t be coffee time in minutes, so t/60 hours. For speed s: 9/s = 4 - t/60. For speed s+2: 9/(s+2) = 2.4 - t/60. Subtracting gives 9/s - 9/(s+2) = 1.6. Simplify: 18/(s(s+2)) = 1.6 → s(s+2) = 11.25. Solve s^2 + 2s - 11.25 = 0 → s = 2.5 km/h. Plug back to find t: 9/2.5 = 3.6 = 4 - t/60 → t/60 = 0.4 → t = 24 minutes. For speed s+½ = 3 km/h, walking time = 9/3 = 3 hours. Total time = 3 hours + 24 minutes = 3h24m = 204 minutes.",
            "We set up equations for walking time at two speeds, subtract to find s, solve quadratic, find t, then compute total time at new speed including coffee time. The result is 3 hours 24 minutes, which equals 204 minutes.",
            "We set up equations for walking time at speeds s and s+2, subtract to find s, then compute t, and finally compute total time at speed s+½. The result is 3 hours 24 minutes, which equals 204 minutes."
        ]
        
        response = await sc_ensemble_op(solutions=solutions, problem=problem)
        print(response)
        
    async def test_programmer(self):
        programmer_op = operator.Programmer(self.llm)
        problem = """
Let $\mathcal{B}$ be the set of rectangular boxes with surface area $54$ and volume $23$. Let $r$ be the radius of the smallest sphere that can contain each of the rectangular boxes that are elements of $\mathcal{B}$. The value of $r^2$ can be written as $\frac{p}{q}$, where $p$ and $q$ are relatively prime positive integers. Find $p+q$.
        """
        
        analysis = """
The problem asks for the smallest sphere radius that can contain all rectangular boxes with total surface area (54) and volume (23). Let the sides be (a,b,c>0). Then (2(ab+bc+ca)=54) and (abc=23). The radius of the sphere containing a box is (r=\frac{1}{2}\sqrt{a^2+b^2+c^2}). We need to maximize (a^2+b^2+c^2) under these constraints. By symmetry, assume (a=b), reduce it to a single-variable problem, compute (r^2=\frac{p}{q}), and find (p+q).
        """
        
        response = await programmer_op(problem=problem, analysis=analysis)
        print(response)
        
    async def test_format(self):
        format_op = operator.Format(self.llm)
        problem = """
Let $\mathcal{B}$ be the set of rectangular boxes with surface area $54$ and volume $23$. Let $r$ be the radius of the smallest sphere that can contain each of the rectangular boxes that are elements of $\mathcal{B}$. The value of $r^2$ can be written as $\frac{p}{q}$, where $p$ and $q$ are relatively prime positive integers. Find $p+q$.
        """
        
        solution = "We set up equations for walking time at two speeds. Let t be coffee time in minutes, so t/60 hours. For speed s: 9/s = 4 - t/60. For speed s+2: 9/(s+2) = 2.4 - t/60. Subtracting gives 9/s - 9/(s+2) = 1.6. Simplify: 18/(s(s+2)) = 1.6 → s(s+2) = 11.25. Solve s^2 + 2s - 11.25 = 0 → s = 2.5 km/h. Plug back to find t: 9/2.5 = 3.6 = 4 - t/60 → t/60 = 0.4 → t = 24 minutes. For speed s+½ = 3 km/h, walking time = 9/3 = 3 hours. Total time = 3 hours + 24 minutes = 3h24m = 204 minutes.",
        
        response = await format_op(problem=problem, solution=solution)
        print(response)
        
    async def test_review(self):
        review_op = operator.Review(self.llm)
        problem = """
Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop.
        """
        
        solution = "We set up equations for walking time at two speeds. Let t be coffee time in minutes, so t/60 hours. For speed s: 9/s = 4 - t/60. For speed s+2: 9/(s+2) = 2.4 - t/60. Subtracting gives 9/s - 9/(s+2) = 1.6. Simplify: 18/(s(s+2)) = 1.6 → s(s+2) = 11.25. Solve s^2 + 2s - 11.25 = 0 → s = 2.5 km/h. Plug back to find t: 9/2.5 = 3.6 = 4 - t/60 → t/60 = 0.4 → t = 24 minutes. For speed s+½ = 3 km/h, walking time = 9/3 = 3 hours. Total time = 3 hours + 24 minutes = 3h24m = 204 minutes.",
        
        response = await review_op(problem=problem, solution=solution)
        print(response)
        
    async def test_revise(self):
        revise_op = operator.Revise(self.llm)
        problem = """
Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop.
        """
        
        solution = "We set up equations for walking time at two speeds. Let t be coffee time in minutes, so t/60 hours. For speed s: 9/s = 4 - t/60. For speed s+2: 9/(s+2) = 2.4 - t/60. Subtracting gives 9/s - 9/(s+2) = 1.6. Simplify: 18/(s(s+2)) = 1.6 → s(s+2) = 11.25. Solve s^2 + 2s - 11.25 = 0 → s = 2.5 km/h. Plug back to find t: 9/2.5 = 3.6 = 4 - t/60 → t/60 = 0.4 → t = 24 minutes. For speed s+½ = 3 km/h, walking time = 9/3 = 3 hours. Total time = 3 hours + 24 minutes = 3h24m = 204 minutes.",
        
        feedback = 'The solution correctly sets up the equations for the walking times at the two given speeds, solves for the unknown speed \\(s\\) and coffee time \\(t\\), and then applies these values to find the total time when walking at \\(s+\\tfrac12\\). All algebraic steps are valid, the discriminant is computed correctly, and the final total time of 3 hours 24 minutes (204 minutes) matches the problem’s requirements. No errors were found.'
        
        response = await revise_op(problem=problem, solution=solution, feedback=feedback)
        print(response)
        
    async def test_debater(self):
        debater_op = operator.Debater(self.llm)
        problem = """
Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop.
        """
        
        proposed_solutions = [
            "We set up equations for walking time at two speeds. Let t be coffee time in minutes, so t/60 hours. For speed s: 9/s = 4 - t/60. For speed s+2: 9/(s+2) = 2.4 - t/60. Subtracting gives 9/s - 9/(s+2) = 1.6. Simplify: 18/(s(s+2)) = 1.6 → s(s+2) = 11.25. Solve s^2 + 2s - 11.25 = 0 → s = 2.5 km/h. Plug back to find t: 9/2.5 = 3.6 = 4 - t/60 → t/60 = 0.4 → t = 24 minutes. For speed s+½ = 3 km/h, walking time = 9/3 = 3 hours. Total time = 3 hours + 24 minutes = 3h24m = 204 minutes.",
            "We set up equations for walking time at two speeds, subtract to find s, solve quadratic, find t, then compute total time at new speed including coffee time. The result is 3 hours 24 minutes, which equals 204 minutes.",
            "We set up equations for walking time at speeds s and s+2, subtract to find s, then compute t, and finally compute total time at speed s+½. The result is 3 hours 24 minutes, which equals 204 minutes."
        ]
        
        response = await debater_op(problem=problem, proposed_solutions=proposed_solutions)
        print(response)
        
    async def test_judge(self):
        judge_op = operator.Judge(self.llm)
        problem = """
Every morning Aya goes for a $9$-kilometer-long walk and stops at a coffee shop afterwards. When she walks at a constant speed of $s$ kilometers per hour, the walk takes her 4 hours, including $t$ minutes spent in the coffee shop. When she walks $s+2$ kilometers per hour, the walk takes her 2 hours and 24 minutes, including $t$ minutes spent in the coffee shop. Suppose Aya walks at $s+\frac{1}{2}$ kilometers per hour. Find the number of minutes the walk takes her, including the $t$ minutes spent in the coffee shop.
        """
        
        solutions = [
            "We set up equations for walking time at two speeds. Let t be coffee time in minutes, so t/60 hours. For speed s: 9/s = 4 - t/60. For speed s+2: 9/(s+2) = 2.4 - t/60. Subtracting gives 9/s - 9/(s+2) = 1.6. Simplify: 18/(s(s+2)) = 1.6 → s(s+2) = 11.25. Solve s^2 + 2s - 11.25 = 0 → s = 2.5 km/h. Plug back to find t: 9/2.5 = 3.6 = 4 - t/60 → t/60 = 0.4 → t = 24 minutes. For speed s+½ = 3 km/h, walking time = 9/3 = 3 hours. Total time = 3 hours + 24 minutes = 3h24m = 204 minutes.",
            "We set up equations for walking time at two speeds, subtract to find s, solve quadratic, find t, then compute total time at new speed including coffee time. The result is 3 hours 24 minutes, which equals 204 minutes.",
            "We set up equations for walking time at speeds s and s+2, subtract to find s, then compute t, and finally compute total time at speed s+½. The result is 3 hours 24 minutes, which equals 204 minutes."
        ]
        
        response = await judge_op(problem=problem, solutions=solutions)
        print(response)
        
async def main():
    operator_test = OperatorTest()
    # await operator_test.test_custom()
    # await operator_test.test_answer_generate()
    await operator_test.test_custom_code_generate()
    # await operator_test.test_sc_ensemble()
    # await operator_test.test_programmer()
    # await operator_test.test_format()
    # await operator_test.test_review()
    # await operator_test.test_revise()
    # await operator_test.test_debater()
    # await operator_test.test_judge()
        
if __name__ == '__main__':
    asyncio.run(main())
