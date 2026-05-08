# src/student_success/interventions/agent.py
import pandas as pd
from pydantic import BaseModel, Field
from typing import List, Callable, Any, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI 
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

class LLMInterventionStep(BaseModel):
    title: str = Field(description="Actionable title of the intervention step.")
    owner: str = Field(description="Who is responsible (e.g., teacher, counselor).")
    description: str = Field(description="Brief explanation of how to execute this step.")

class LLMInterventionPlan(BaseModel):
    summary: str = Field(description="A brief summary of the student's current risk and the best simulated path.")
    recommended_steps: List[LLMInterventionStep] = Field(description="2-4 recommended actions based on the scenarios.")
    monitoring_note: str = Field(description="A short monitoring plan.")

class AgenticInterventionPlanner:
    def __init__(self, score_fn: Callable[[pd.DataFrame], Any], model_name: str = "gemini-3.1-flash-lite"): 
        self.score_fn = score_fn
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.2)
        self.structured_llm = self.llm.with_structured_output(LLMInterventionPlan)
        
        self.system_message = (
            "You are an expert educational counselor generating personalized intervention plans. "
            "Your goal is to iteratively use the `simulate_intervention` tool to find a combination of "
            "changes (max 2) that raises the student's predicted grade as close to the passing threshold ({pass_threshold}) as possible. "
            "Focus on actionable features like studytime, absences, schoolsup, famsup, internet, activities. "
            "Once you find the best scenario, summarize your plan and explain the recommended steps."
        )
        
    def build_plan(
        self, 
        student_id: str, 
        base_features: Dict[str, Any], 
        base_prediction: float, 
        pass_threshold: float
    ) -> LLMInterventionPlan:
        
        @tool
        def simulate_intervention(
            absences: int = None, 
            studytime: int = None,
            schoolsup: str = None,
            famsup: str = None,
            internet: str = None,
            activities: str = None
        ) -> float:
            """Simulate the student's grade if one or more features are changed. 
            Pass the proposed new values. Only change up to 2 features. 
            Leave the rest as None to keep their baseline values.
            """
            candidate = base_features.copy()
            if absences is not None: candidate["absences"] = absences
            if studytime is not None: candidate["studytime"] = studytime
            if schoolsup is not None: candidate["schoolsup"] = schoolsup
            if famsup is not None: candidate["famsup"] = famsup
            if internet is not None: candidate["internet"] = internet
            if activities is not None: candidate["activities"] = activities
            
            candidate_frame = pd.DataFrame([candidate])
            prediction = self.score_fn(candidate_frame)[0]
            return float(prediction)

        tools = [simulate_intervention]
        system_prompt = self.system_message.format(pass_threshold=pass_threshold)
        agent_executor = create_react_agent(self.llm, tools, prompt=system_prompt)
        
        human_message = f"Student ID: {student_id}\nBase Grade: {base_prediction}\nBaseline Features: {base_features}"
        
        # Run the agent
        result = agent_executor.invoke({"messages": [("human", human_message)]})
        
        # Force strict formatting of the output
        final_msg = result["messages"][-1]
        if isinstance(final_msg.content, list):
            text_content = " ".join(
                str(c.get("text", "")) if isinstance(c, dict) else str(c) 
                for c in final_msg.content
            )
        else:
            text_content = str(final_msg.content)
            
        return self.structured_llm.invoke(text_content)