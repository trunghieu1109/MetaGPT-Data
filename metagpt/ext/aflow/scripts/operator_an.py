# -*- coding: utf-8 -*-
# @Date    : 6/27/2024 19:46 PM
# @Author  : didi
# @Desc    : action nodes for operator

from pydantic import BaseModel, Field


class GenerateOp(BaseModel):
    response: str = Field(default="", description="Your solution for this problem")


class CodeGenerateOp(BaseModel):
    code: str = Field(default="", description="Your complete code solution for this problem")


class AnswerGenerateOp(BaseModel):
    thought: str = Field(default="", description="The step by step thinking process")
    answer: str = Field(default="", description="The final answer to the question")


class FormatOp(BaseModel):
    solution: str = Field(default="", description="Your formatted answer for this problem")


class ScEnsembleOp(BaseModel):
    thought: str = Field(default="", description="The thought of the most consistent solution.")
    sc_solution: str = Field(default="", description="The most consistent solution.")


class ReflectionTestOp(BaseModel):
    reflection_and_solution: str = Field(
        default="", description="Corrective solution for code execution errors or test case failures"
    )


class MdEnsembleOp(BaseModel):
    thought: str = Field(default="", description="Step-by-step analysis of the solutions to determine the best one.")
    solution_letter: str = Field(default="", description="The letter of the chosen best solution (only one letter).")


class ReviewOp(BaseModel):
    review_result: bool = Field(
        default=False,
        description="The Review Result (Bool). If you think this solution looks good for you, return 'true'; If not, return 'false'",
    )
    feedback: str = Field(
        default="",
        description="Your FeedBack for this problem based on the criteria. If the review result is true, you can put it 'nothing here'.",
    )


class ReviseOp(BaseModel):
    revised_solution: str = Field(default="", description="Based on the feedback, revised solution for this problem")

class DebaterOp(BaseModel):
    feedback: str = Field(default="", description="Based on the proposed solutions and your thought, return the feedback for these proposed solutions")
    solution: str = Field(default="", description="Based on the proposed solutions and your thought, return a new solution for this problem")

class JudgeOp(BaseModel):
    justification: str = Field(default="", description="The justification for your decision")
    best_solution: str = Field(default="", description="The final decision after judging the proposed solutions")

class GeneratePlanOp(BaseModel):
    detailed_plan: str = Field(default="", description="The detailed plan generated in JSON format, exactly max_scenario_len steps")

class GenerateMASOp(BaseModel):
    mas_code: str = Field(default="", description="The executable Python code for this problem")