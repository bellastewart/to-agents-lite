import json
import os 
import numpy as np
from typing import Dict, Optional
from autogen import UserProxyAgent, Agent
from openai import OpenAI
import instructor


def _instructor_mode():
    """See agents/pydantic_agent.py — Gemini's OpenAI-compat endpoint rejects
    JSON_SCHEMA; TOOLS/JSON/MD_JSON/JSON_O1 are verified working there.
    Default keeps the original behaviour."""
    import os
    name = os.environ.get("TO_INSTRUCTOR_MODE", "JSON_SCHEMA").strip().upper()
    mode = getattr(instructor.Mode, name, None)
    if mode is None:
        available = [m for m in dir(instructor.Mode) if m.isupper()]
        raise ValueError(
            f"TO_INSTRUCTOR_MODE={name!r} is not a valid instructor mode. "
            f"Available: {', '.join(sorted(available))}")
    return mode

# Import all Pydantic models from pydantic_agent
from agents.pydantic_agent import (
    PydanticStructure,
    Physics,
    Mesh,
    Multigrid,
    AxisRule,
    BCSelection,
    BoundaryCondition,
    ForceSelection,
    PointForce,
    Filter,
    Problem,
    Optimizer,
    Optimization_settings
)


class ReviseAgent(UserProxyAgent):
    """
    Agent responsible for parsing VLLMAgent suggestions and revising PydanticAgent JSON.
    Automatically retrieves suggestions and original JSON from message history.
    Uses Pydantic validation (via instructor) to ensure correct JSON structure.
    """
    
    def __init__(
        self,
        name="ReviseAgent",
        system_message=None,
        llm_config=None,
        llm_config_TO=None,
        generate=None,  # Extract it explicitly
        **kwargs
    ):
        # Store custom parameters BEFORE calling super()
        self.llm_config_TO = llm_config_TO
        self.generate = generate
        
        # Call super() WITHOUT custom parameters
        super().__init__(
            name=name,
            system_message=system_message,
            llm_config=llm_config,
            **kwargs
        )
        
        # Register reply handler
        self.register_reply(Agent, ReviseAgent._generate_revision_reply, position=2)
    
    # def retrieve_from_history(self):
    #     """
    #     Retrieve VLLMAgent suggestions and PydanticAgent JSON from message history.
    #     Looks through the GroupChat history instead of agent-specific messages.
        
    #     Returns:
    #         Tuple of (pydantic_json, vllm_suggestions) or (None, None) if not found
    #     """
    #     pydantic_json = None
    #     vllm_suggestions = None
        
    #     # Search through message history
    #     try:
    #         # First, try to get messages from chat_manager if available
    #         chat_manager_messages = self._oai_messages.get(list(self._oai_messages.keys())[0], [])
            
    #         # Search through all messages in reverse chronological order
    #         for msg in reversed(chat_manager_messages):
    #             sender_name = msg.get("name", "").lower()
    #             content = msg.get("content", "")
                
    #             # Look for VLLMAgent response
    #             if ("vllm" in sender_name or "vision" in sender_name) and not vllm_suggestions:
    #                 print(f"🔍 Found message from VLLMAgent")
    #                 try:
    #                     data = json.loads(content)
    #                     if "analysis" in data:
    #                         vllm_suggestions = data["analysis"]
    #                         print(f"✅ Retrieved VLLMAgent suggestions")
    #                     elif "suggestions" in data:
    #                         vllm_suggestions = data["suggestions"]
    #                         print(f"✅ Retrieved VLLMAgent suggestions")
    #                 except json.JSONDecodeError:
    #                     # Not JSON, might be plain text suggestions
    #                     if len(content) > 50:
    #                         vllm_suggestions = content
    #                         print(f"✅ Retrieved VLLMAgent suggestions (plain text)")
                
    #             # Look for PydanticAgent response
    #             if ("pydantic" in sender_name) and not pydantic_json:
    #                 print(f"🔍 Found message from PydanticAgent")
    #                 try:
    #                     data = json.loads(content)
    #                     # Verify it has the expected structure
    #                     if "physics" in data and "mesh" in data and "bc" in data:
    #                         pydantic_json = data
    #                         print(f"✅ Retrieved PydanticAgent JSON")
    #                 except json.JSONDecodeError:
    #                     continue
                
    #             # Early exit if we found both
    #             if pydantic_json and vllm_suggestions:
    #                 break
            
    #         # Debug: Print message senders we found
    #         if not pydantic_json or not vllm_suggestions:
    #             print("\n🔍 Messages found in history:")
    #             for msg in reversed(chat_manager_messages[-10:]):  # Last 10 messages
    #                 sender_name = msg.get("name", "unknown")
    #                 content_preview = msg.get("content", "")[:100]
    #                 print(f"  - {sender_name}: {content_preview}...")
                        
    #     except Exception as e:
    #         print(f"⚠️ Error retrieving from history: {e}")
    #         import traceback
    #         traceback.print_exc()
        
    #     if not pydantic_json:
    #         print("⚠️ Could not find PydanticAgent JSON in message history")
    #     if not vllm_suggestions:
    #         print("⚠️ Could not find VLLMAgent suggestions in message history")
        
    #     return pydantic_json, vllm_suggestions


    # CHANGED TO ENABLE MULTIPLE PASSES 
    # def retrieve_from_history(self):
    #     """
    #     Retrieve VLLMAgent suggestions and the most recent JSON config from message history.
    #     Prefers ReviseAgent's revised JSON over PydanticAgent's original JSON.
        
    #     Returns:
    #         Tuple of (latest_json, vllm_suggestions) or (None, None) if not found
    #     """
    #     latest_json = None
    #     latest_json_source = None
    #     vllm_suggestions = None
    
    #     try:
    #         chat_manager_messages = self._oai_messages.get(list(self._oai_messages.keys())[0], [])
    
    #         for msg in reversed(chat_manager_messages):
    #             sender_name = msg.get("name", "").lower()
    #             content = msg.get("content", "")
    
    #             # Look for VLLMAgent response
    #             if ("vllm" in sender_name or "vision" in sender_name) and not vllm_suggestions:
    #                 print(f"🔍 Found message from VLLMAgent")
    #                 try:
    #                     data = json.loads(content)
    #                     if "analysis" in data:
    #                         vllm_suggestions = data["analysis"]
    #                         print(f"✅ Retrieved VLLMAgent suggestions")
    #                     elif "suggestions" in data:
    #                         vllm_suggestions = data["suggestions"]
    #                         print(f"✅ Retrieved VLLMAgent suggestions")
    #                 except json.JSONDecodeError:
    #                     if len(content) > 50:
    #                         vllm_suggestions = content
    #                         print(f"✅ Retrieved VLLMAgent suggestions (plain text)")
    
    #             # Look for most recent JSON config (ReviseAgent first, then PydanticAgent)
    #             if not latest_json:
    #                 if "revise" in sender_name:
    #                     print(f"🔍 Found message from ReviseAgent")
    #                     try:
    #                         data = json.loads(content)
    #                         if "revised_json" in data:
    #                             latest_json = data["revised_json"]
    #                             latest_json_source = "ReviseAgent"
    #                             print(f"✅ Retrieved revised JSON from ReviseAgent")
    #                     except json.JSONDecodeError:
    #                         pass
    
    #                 elif "pydantic" in sender_name:
    #                     print(f"🔍 Found message from PydanticAgent")
    #                     try:
    #                         data = json.loads(content)
    #                         if "physics" in data and "mesh" in data and "bc" in data:
    #                             latest_json = data
    #                             latest_json_source = "PydanticAgent"
    #                             print(f"✅ Retrieved original JSON from PydanticAgent")
    #                     except json.JSONDecodeError:
    #                         continue
    
    #             # Early exit if we found both
    #             if latest_json and vllm_suggestions:
    #                 break
    
    #         # Debug
    #         if not latest_json or not vllm_suggestions:
    #             print("\n🔍 Messages found in history:")
    #             for msg in reversed(chat_manager_messages[-10:]):
    #                 sender_name = msg.get("name", "unknown")
    #                 content_preview = msg.get("content", "")[:100]
    #                 print(f"  - {sender_name}: {content_preview}...")
    
    #     except Exception as e:
    #         print(f"⚠️ Error retrieving from history: {e}")
    #         import traceback
    #         traceback.print_exc()
    
    #     if not latest_json:
    #         print("⚠️ Could not find any JSON config in message history")
    #     else:
    #         print(f"📌 Using JSON from: {latest_json_source}")
    #     if not vllm_suggestions:
    #         print("⚠️ Could not find VLLMAgent suggestions in message history")
    
    #     return latest_json, vllm_suggestions

    # CHANGED TO ENABLE BEST ITERATION: 

    def retrieve_from_history(self):
        """
        Retrieve VLLMAgent suggestions and the best JSON config from message history.
        Priority: VLLMAgent base_config (best-scoring) > ReviseAgent > PydanticAgent
        """
        latest_json = None
        latest_json_source = None
        vllm_suggestions = None
    
        try:
            chat_manager_messages = self._oai_messages.get(list(self._oai_messages.keys())[0], [])
    
            for msg in reversed(chat_manager_messages):
                sender_name = msg.get("name", "").lower()
                content = msg.get("content", "")
    
                # Look for VLLMAgent response
                if ("vllm" in sender_name or "vision" in sender_name) and not vllm_suggestions:
                    print(f"🔍 Found message from VLLMAgent")
                    try:
                        data = json.loads(content)
                        
                        # Priority 0: VLLMAgent's base_config (best-scoring)
                        if "base_config" in data and data["base_config"] and not latest_json:
                            latest_json = data["base_config"]
                            base_rev = data.get("base_revision", "?")
                            latest_json_source = f"VLLMAgent (best-scoring revision {base_rev})"
                            print(f"✅ Retrieved best-scoring config from VLLMAgent (revision {base_rev})")
                        
                        if "analysis" in data:
                            vllm_suggestions = data["analysis"]
                            print(f"✅ Retrieved VLLMAgent suggestions")
                        elif "suggestions" in data:
                            vllm_suggestions = data["suggestions"]
                            print(f"✅ Retrieved VLLMAgent suggestions")
                    except json.JSONDecodeError:
                        if len(content) > 50:
                            vllm_suggestions = content
                            print(f"✅ Retrieved VLLMAgent suggestions (plain text)")
    
                # Fallback: ReviseAgent or PydanticAgent JSON
                if not latest_json:
                    if "revise" in sender_name:
                        print(f"🔍 Found message from ReviseAgent")
                        try:
                            data = json.loads(content)
                            if "revised_json" in data:
                                latest_json = data["revised_json"]
                                latest_json_source = "ReviseAgent"
                                print(f"✅ Retrieved revised JSON from ReviseAgent")
                        except json.JSONDecodeError:
                            pass
    
                    elif "pydantic" in sender_name:
                        print(f"🔍 Found message from PydanticAgent")
                        try:
                            data = json.loads(content)
                            if "physics" in data and "mesh" in data and "bc" in data:
                                latest_json = data
                                latest_json_source = "PydanticAgent"
                                print(f"✅ Retrieved original JSON from PydanticAgent")
                        except json.JSONDecodeError:
                            continue
    
                # Early exit if we found both
                if latest_json and vllm_suggestions:
                    break
    
            if not latest_json or not vllm_suggestions:
                print("\n🔍 Messages found in history:")
                for msg in reversed(chat_manager_messages[-10:]):
                    sender_name = msg.get("name", "unknown")
                    content_preview = msg.get("content", "")[:100]
                    print(f"  - {sender_name}: {content_preview}...")
    
        except Exception as e:
            print(f"⚠️ Error retrieving from history: {e}")
            import traceback
            traceback.print_exc()
    
        if not latest_json:
            print("⚠️ Could not find any JSON config in message history")
        else:
            print(f"📌 Using JSON from: {latest_json_source}")
        if not vllm_suggestions:
            print("⚠️ Could not find VLLMAgent suggestions in message history")
    
        return latest_json, vllm_suggestions
        
    
    def parse_and_revise_json(
        self,
        original_json: Dict,
        suggestions_text: str
    ) -> Optional[Dict]:
        """
        Parse VLLMAgent suggestions and apply them to the original JSON configuration.
        Uses instructor + Pydantic for strict schema validation (like PydanticAgent).
        
        Args:
            original_json: The original structured JSON from PydanticAgent
            suggestions_text: Natural language description of changes from VLLMAgent
            
        Returns:
            Revised JSON configuration or None if parsing fails
        """
        if self.llm_config_TO is None:
            print("⚠️ No llm_config_TO configured, skipping revision")
            return None
        
        # Extract config for instructor (like PydanticAgent does)
        client = self.llm_config_TO["client_TO"]
        model = self.llm_config_TO["model_TO"]
        max_tokens = self.llm_config_TO["max_tokens_TO"]
        
        # System prompt for parsing modifications
        system_prompt = """You are an expert at parsing natural language descriptions of topology optimization modifications and applying them to structured JSON configurations.

Your task is to intelligently map suggested changes to the correct JSON fields and output a revised configuration that conforms to the PydanticStructure schema.

## JSON Structure Reference

The configuration has the following structure:

{
  "physics": [{"E": float, "nu": float}],
  "mesh": [{"nx": int, "ny": int, "nz": int, "lx": float, "ly": float, "lz": float}],
  "multigrid": {"tol": float, "maxiter": int, "n_level": int},
  "bc": [
    {
      "name": "string",
      "selection": {
        "rules": [
          {
            "axis": "x"|"y"|"z",
            "operator": "equals"|"greater_than"|"less_than"|"between",
            "value": float,
            "value_max": float  // only for "between"
          }
        ],
        "tolerance": float
      },
      "dofs": {"ux": float|null, "uy": float|null, "uz": float|null}
    }
  ],
  "forces": [
    {
      "name": "string",
      "selection": {
        "rules": [/* same as bc.selection.rules */],
        "tolerance": float
      },
      "forces": {"fx": float|null, "fy": float|null, "fz": float|null},
      "divide_by_num_nodes": bool
    }
  ],
  "filter": {"r_min": float},
  "problem": {
    "type": "MinimumCompliance",
    "penalty_schedule": null,
    "void": float,
    "penalty": float,
    "E_mul": [float],
    "volume_fraction": [float],
    "heavyside": bool
  },
  "optimizer": {
    "type": "PGD"|"MMA"|"OC",
    "change_tol": float|null,
    "fun_tol": float
  },
  "optimization_settings": {
    "num_iterations": int
  }
}

## Mapping Guidelines

When you see phrases like:
- "fix the right face" → Add/modify entry in "bc" array with selection rules for x=lx (use the lx value from mesh)
- "add support at the right face" → Add new BC with rules=[{"axis": "x", "operator": "equals", "value": lx}]
- "change force magnitude to -2.0" → Modify "forces" array, update fx/fy/fz
- "increase volume fraction to 30%" → Change "problem.volume_fraction" to [0.3]
- "use MMA optimizer" → Change "optimizer.type" to "MMA"
- "add support at bottom" → Add new entry to "bc" array with y=0 selection
- "move force to center" → Modify "forces" array selection rules to point to center coordinates
- "make mesh finer" → Increase "mesh" nx/ny/nz values proportionally
- "relax convergence" → Increase "optimizer.fun_tol" or "multigrid.tol"
- "add additional boundary constraint" → Add new entry to "bc" array
- "remove force at location X" → Remove corresponding entry from "forces" array
- "constrain all DOFs at right face" → Add BC with all dofs set to 0.0

## CRITICAL RULES

1. Parse the suggestions to understand what needs to change
2. Apply ONLY the changes mentioned in the suggestions
3. Keep all other fields exactly as they were in the original JSON
4. Maintain the exact JSON structure and field names
5. Ensure all numerical values are valid (no null for required fields)
6. When modifying arrays (bc, forces), preserve existing entries unless explicitly told to remove them
7. When adding new BCs or forces, use values from the mesh dimensions (lx, ly, lz) as appropriate
8. The output will be validated against PydanticStructure schema, so ensure compliance
9. For boundary conditions, remember that dofs can be null (meaning unconstrained) or a float (typically 0.0 for fixed)
10. **CRITICAL: Maintain physics logic. (e.g. If a force is applied in a direction (fx/fy/fz) at a location, that same DOF (ux/uy/uz) must NOT be constrained (0.0) at overlapping nodes. For example, if fy is applied at x=0, then uy must be null (free) at x=0.**

If suggestions are vague, make reasonable engineering assumptions based on standard topology optimization practices."""

        user_prompt = f"""### Original JSON Configuration
```json
{json.dumps(original_json, indent=2)}
```

### VLLMAgent Suggestions
{suggestions_text}

### Task
Parse the suggestions above and output the complete revised JSON configuration with those changes applied.

IMPORTANT: 
- Keep everything else exactly the same, only modify what's mentioned in the suggestions
- When referencing mesh boundaries (e.g., "right face"), use the actual lx/ly/lz values from the mesh
- Preserve all existing BCs and forces unless explicitly told to remove them
- Output a valid PydanticStructure configuration

Output the revised configuration:"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            # Use instructor to enforce Pydantic schema (like PydanticAgent)
            create = instructor.patch(
                create=client.chat.completions.create,
                mode=_instructor_mode(),  # LITE: was hardcoded JSON_SCHEMA
            )
            
            # Get Pydantic-validated response
            result = create(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                temperature=0.1,  # Low temperature for precise parsing
                response_model=PydanticStructure,  # Enforce schema
            )
            
            # Convert Pydantic model to dict (like PydanticAgent's TOsolve)
            revised_json = self._pydantic_to_dict(result)
            
            print("\n📝 Revised JSON Configuration:")
            print("=" * 60)
            print(json.dumps(revised_json, indent=2))
            print("=" * 60 + "\n")
            print("✅ Pydantic validation passed")
            
            return revised_json
        
        except Exception as e:
            print(f"⚠️ Revision failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _pydantic_to_dict(self, params: PydanticStructure) -> Dict:
        """
        Convert PydanticStructure to dict format (like PydanticAgent's TOsolve).
        """
        params_dict = params.model_dump()
        return {
            "physics": params_dict["physics"],
            "mesh": params_dict["mesh"],
            "multigrid": params_dict["multigrid"],
            "bc": params_dict["bc"],
            "forces": params_dict.get("forces", []),
            "filter": params_dict.get("filter", {"r_min": 1.5}),
            "problem": params_dict.get("problem", {
                "type": "MinimumCompliance",
                "penalty_schedule": None,
                "void": 1e-9,
                "penalty": 3.0,
                "E_mul": [1.0],
                "volume_fraction": [0.2],
                "heavyside": True
            }),
            "optimizer": params_dict.get("optimizer", {
                "type": "PGD",
                "change_tol": None,
                "fun_tol": 1e-4
            }),
            "optimization_settings": params_dict.get("optimization_settings", {
                "num_iterations": 100
            }),
        }
    
    def _generate_revision_reply(self, messages=None, sender=None, config=None):
        """
        Handle incoming requests for JSON revision.
        Automatically retrieves data from message history.
        Can optionally accept 'original_json' and 'suggestions' in the message to override history.
        """
        last_msg = self._oai_messages[sender][-1]
        
        try:
            # Try to parse incoming message
            try:
                data = json.loads(last_msg["content"])
                original_json = data.get("original_json")
                suggestions = data.get("suggestions")
            except:
                # Message is not JSON, retrieve from history
                original_json = None
                suggestions = None
            
            # If not provided in message, retrieve from history -- what we're actually doing here 
            if not original_json or not suggestions:
                print("\n🔍 Retrieving data from message history...")
                hist_json, hist_suggestions = self.retrieve_from_history()
                
                original_json = original_json or hist_json
                suggestions = suggestions or hist_suggestions
            
            # Validate we have both pieces
            if not original_json or not suggestions:
                return True, {
                    "role": "assistant",
                    "content": json.dumps({
                        "error": "Missing required data: could not find 'original_json' or 'suggestions' in message or history"
                    })
                }
            
            print("\n🔧 Starting JSON revision process with Pydantic validation...")
            
            # Parse and revise with Pydantic validation
            revised_json = self.parse_and_revise_json(original_json, suggestions)
            
            if revised_json is None:
                return True, {
                    "role": "assistant",
                    "content": json.dumps({
                        "error": "Failed to parse and revise JSON"
                    })
                }
            
            # Convert numpy types to JSON-safe types (like PydanticAgent)
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(v) for v in obj]
                return obj
            
            revised_json = convert_numpy(revised_json)

            ###############################################################
            # 💾 SAVE REVISED JSON USING VERSIONED revision_X.json FILES
            ###############################################################
            save_dir = "revised_TO_JSON"
            os.makedirs(save_dir, exist_ok=True)
            
            # Count existing revision_X.json files
            existing = [
                f for f in os.listdir(save_dir)
                if f.startswith("revision_") and f.endswith(".json")
            ]
            
            # Determine next revision number
            if not existing:
                next_revision = 1
            else:
                nums = [
                    int(f.replace("revision_", "").replace(".json", ""))
                    for f in existing
                    if f.replace("revision_", "").replace(".json", "").isdigit()
                ]
                next_revision = max(nums) + 1
            
            save_path = os.path.join(save_dir, f"revision_{next_revision}.json")
            
            # Write revised JSON
            with open(save_path, "w") as f:
                json.dump(revised_json, f, indent=2)
            
            ###############################################################

            
            return True, {
                "role": "assistant",
                "content": json.dumps({
                    "status": "Revision complete",
                    "revised_json": revised_json,
                    "validation_passed": True,  # Always true if we got here
                    "original_json": original_json,
                    "suggestions_applied": suggestions
                })
            }
            
        except Exception as e:
            import traceback
            return True, {
                "role": "assistant",
                "content": json.dumps({
                    "error": f"ReviseAgent failed: {str(e)}",
                    "traceback": traceback.format_exc()
                })
            }


        