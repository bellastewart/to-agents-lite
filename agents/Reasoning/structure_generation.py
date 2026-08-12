import os
import sys
sys.path.append("..")
import json
import time
from copy import deepcopy


def structurePrompt(input: str, generate_pydantic):

    SYS_PROMPT_MAKER = """
        You are an expert in extracting parameters for topology optimization.
        Always return valid JSON with the following structure:
        
    {
      "physics": [...],
      "mesh": [...],
      "multigrid": {...},
      "bc": [...],
      "forces": [...],
      "filter": {
        "r_min": float 
      },
      "problem": {
        "type": "MinimumCompliance",
        "penalty_schedule": null,  
        "void": float,  // Void material stiffness
        "penalty": float,  // SIMP penalty parameter 
        "E_mul": [float],  // Young's modulus multiplier(s)
        "volume_fraction": [float],  // Target volume fraction(s)
        "heavyside": bool  // Use Heaviside projection (default: true)
      },
      "optimizer": {
        "type": "PGD" | "MMA" | "OC",  // Optimizer type 
        "change_tol": float or null,  // null means infinity 
        "fun_tol": float  // Function tolerance 
      }
    }
        
        Rules:
        - ... (existing rules) ...
        - For filter:
            - Extract r_min (filter radius) if mentioned
        - For problem:
            - Extract void material stiffness if mentioned 
            - Extract SIMP penalty if mentioned 
            - Extract volume fraction if mentioned 
            - E_mul is typically [1.0] for single material
            - heavyside should be true unless explicitly stated otherwise
        - For optimizer:
            - Extract optimizer type if mentioned (PGD, MMA, or OC)
            - change_tol should be null (infinity) unless explicitly specified
            - fun_tol (function tolerance) default is 1e-4
        - Example interpretations:
            - "filter radius of 2.0" → r_min: 2.0
            - "20% volume fraction" → volume_fraction: [0.2]
            - "volume fraction of 0.3" → volume_fraction: [0.3]
            - "SIMP penalty of 3" → penalty: 3.0
            - "use PGD optimizer" → type: "PGD"
            - "convergence tolerance of 1e-5" → fun_tol: 1e-5
    """


    USER_PROMPT = f"Context: ```{input}``` Extract the structured JSON:"

    print("Generating structure...")
    return generate_pydantic(system_prompt=SYS_PROMPT_MAKER, prompt=USER_PROMPT)




def make_structure_from_text (txt, generate_pydantic):    

        result = structurePrompt(txt,generate_pydantic)

        return result 
