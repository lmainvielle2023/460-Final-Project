# src/student_success/interventions/agent.py
from pydantic import BaseModel, Field
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI 

from student_success.interventions.planner import build_agent_prompt
from student_success.schemas import PredictionRecord, SimulationResult

class LLMInterventionStep(BaseModel):
    title: str = Field(description="Actionable title of the intervention step.")
    owner: str = Field(description="Who is responsible (e.g., teacher, counselor).")
    description: str = Field(description="Brief explanation of how to execute this step.")

class LLMInterventionPlan(BaseModel):
    summary: str = Field(description="A brief summary of the student's current risk and the best simulated path.")
    recommended_steps: List[LLMInterventionStep] = Field(description="2-4 recommended actions based on the scenarios.")
    monitoring_note: str = Field(description="A short monitoring plan.")

class AgenticInterventionPlanner:
    def __init__(self, model_name: str = "gemini-2.5-flash"): 
        
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.2)
        self.structured_llm = self.llm.with_structured_output(LLMInterventionPlan)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert educational counselor generating personalized intervention plans. Ensure steps are actionable and align with these targets: studytime, absences, schoolsup, famsup, internet, activities."),
            ("human", "{agent_context}")
        ])
        self.chain = self.prompt | self.structured_llm

    def build_plan(self, prediction: PredictionRecord, scenarios: list[SimulationResult]) -> LLMInterventionPlan:
        raw_prompt_text = build_agent_prompt(prediction, scenarios)
        return self.chain.invoke({"agent_context": raw_prompt_text})