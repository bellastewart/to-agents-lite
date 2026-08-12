# COMBINED VLLM AGENT — toggle PROBLEM_TYPE to switch between phone stand and cantilever beam.

import json
import os
import io
import re
import base64
from typing import Optional
from PIL import Image
from autogen import UserProxyAgent, Agent
from openai import OpenAI


# =============================================================================
# PROBLEM TYPE TOGGLE — set to "phone" or "cantilever"
# =============================================================================
PROBLEM_TYPE = "phone"   # "phone" or "cantilever"
# =============================================================================

assert PROBLEM_TYPE in ("phone", "cantilever"), (
    f"PROBLEM_TYPE must be 'phone' or 'cantilever', got {PROBLEM_TYPE!r}"
)


class VLLMAgentBoth(UserProxyAgent):
    """
    Duplicate of VLLMAgent that attaches BOTH the depth-shaded and the
    stress-field screenshot to every LLM call.

    Path convention (matches _3d.py = _3d_both.py output):
        <screenshot_dir>/depth/{view}.png
        <screenshot_dir>/stress/{view}.png
    The `screenshot_path` stored in history is the stress path; the depth
    sibling is derived from it via `_paired_path`.
    """

    def __init__(
        self,
        name="VLLMAgentBoth",
        system_message=None,
        human_input_mode="NEVER",
        code_execution_config={"use_docker": False},
        llm_config=None,
        **kwargs
    ):
        super().__init__(
            name=name,
            system_message=system_message,
            human_input_mode=human_input_mode,
            code_execution_config=code_execution_config,
            llm_config=llm_config,
            **kwargs
        )

        print(f"🔧 VLLMAgentBoth initialized for problem type: {PROBLEM_TYPE.upper()}")
        self.register_reply(Agent, VLLMAgentBoth._generate_analysis_reply, position=2)

    # =========================================================================
    # Dual-image helpers (depth + stress)
    # =========================================================================

    @staticmethod
    def _paired_path(stress_path: str) -> str:
        """Return the depth sibling of a stress screenshot path.

        Given `<dir>/stress/<view>.png` returns `<dir>/depth/<view>.png`.
        Falls back to swapping a top-level `stress` → `depth` token if the
        path doesn't end with `.../stress/<view>.png`.
        """
        if "/stress/" in stress_path:
            return stress_path.replace("/stress/", "/depth/", 1)
        return stress_path  # unknown layout — caller will handle missing file

    def _append_paired_images(self, content: list, stress_path: str,
                               label: str = "") -> int:
        """Encode and append BOTH depth and stress images to a content list.

        Returns the number of images actually attached (0, 1, or 2).
        Each image is preceded by a short text label so the LLM can tell
        them apart.
        """
        depth_path = self._paired_path(stress_path)
        n_attached = 0

        depth_b64 = self._encode_image(depth_path)
        if depth_b64:
            content.append({
                "type": "text",
                "text": f"{label} — *depth-shaded view* (`{depth_path}`):\n"
            })
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{depth_b64}"}
            })
            n_attached += 1

        stress_b64 = self._encode_image(stress_path)
        if stress_b64:
            content.append({
                "type": "text",
                "text": f"{label} — *stress field* (`{stress_path}`):\n"
            })
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{stress_b64}"}
            })
            n_attached += 1

        return n_attached

    # =========================================================================
    # History Retrieval Methods
    # =========================================================================

    def retrieve_judge_analysis_from_history(self) -> Optional[str]:
        """
        Retrieve the most recent AI_Judge analysis from message history.
        Returns the analysis text or None if no AI_Judge has spoken yet.
        """
        try:
            chat_manager_messages = self._oai_messages.get(
                list(self._oai_messages.keys())[0], []
            )

            for msg in reversed(chat_manager_messages):
                sender_name = msg.get("name", "").lower()
                content = msg.get("content", "")

                if "judge" in sender_name:
                    try:
                        data = json.loads(content)
                        if "analysis" in data and data["analysis"]:
                            print(f"✅ Retrieved AI_Judge analysis from history")
                            return data["analysis"]
                    except json.JSONDecodeError:
                        if len(content) > 50:
                            return content

        except Exception as e:
            print(f"⚠️ Error retrieving judge analysis: {e}")

        return None

    def retrieve_latest_screenshot_dir_from_history(self) -> str:
        """Retrieve the most recent screenshot_dir from TOAgent history."""
        try:
            chat_manager_messages = self._oai_messages.get(
                list(self._oai_messages.keys())[0], []
            )
            for msg in reversed(chat_manager_messages):
                sender_name = msg.get("name", "").lower()
                if "toagent" in sender_name or "to_agent" in sender_name:
                    try:
                        data = json.loads(msg.get("content", ""))
                        if "screenshot_dir" in data:
                            print(f"✅ Retrieved screenshot_dir from TOAgent: {data['screenshot_dir']}")
                            return data["screenshot_dir"]
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"⚠️ Error retrieving screenshot dir: {e}")
        return "screenshots"

    def retrieve_problem_description_from_history(self):
        """
        Search chat history in reverse for the most recent problem config JSON.
        Looks for ReviseAgent's revised_json first, falls back to PydanticAgent.
        Returns the JSON as a formatted string.
        """
        try:
            chat_manager_messages = self._oai_messages.get(
                list(self._oai_messages.keys())[0], [])

            config_json = None
            source = None

            for msg in reversed(chat_manager_messages):
                sender_name = msg.get("name", "").lower()
                content = msg.get("content", "")

                # Priority 1: ReviseAgent's revised_json
                if "revise" in sender_name and not config_json:
                    try:
                        data = json.loads(content)
                        if "revised_json" in data:
                            config_json = data["revised_json"]
                            source = "ReviseAgent"
                            break
                    except (json.JSONDecodeError, TypeError):
                        continue

                # Priority 2: PydanticAgent's original JSON
                if "pydantic" in sender_name and not config_json:
                    try:
                        data = json.loads(content)
                        if "physics" in data and "mesh" in data and "bc" in data:
                            config_json = data
                            source = "PydanticAgent"
                    except (json.JSONDecodeError, TypeError):
                        continue

            if config_json:
                print(f"✅ Retrieved problem config from {source}")
                return json.dumps(config_json, indent=2)
            else:
                print("⚠️ No problem config found in history")
                return None

        except Exception as e:
            print(f"⚠️ Error retrieving problem description: {e}")
            return None

    # =========================================================================
    # Full Iteration History
    # =========================================================================

    def retrieve_full_iteration_history(self) -> list:
        """
        Retrieve all previous iterations' data: TO JSON config, screenshot path,
        and judge score/analysis.

        Returns a list of dicts sorted by revision number:
        [
            {
                "revision": 0,
                "config_json": {...},
                "screenshot_path": "screenshots/stress/front.png",
                "judge_score": 3,
                "judge_justification": "This structure exhibits..."
            },
            ...
        ]
        """
        iterations = []

        try:
            chat_manager_messages = self._oai_messages.get(
                list(self._oai_messages.keys())[0], []
            )

            configs = {}       # revision -> config_json
            screenshots = {}   # revision -> screenshot_dir
            judge_data = {}    # revision -> {score, justification}

            revision_counter = 0

            for msg in chat_manager_messages:
                sender_name = msg.get("name", "").lower()
                content = msg.get("content", "")

                try:
                    data = json.loads(content)
                except (json.JSONDecodeError, TypeError):
                    continue

                # Collect PydanticAgent configs (revision 0)
                if "pydantic" in sender_name:
                    if "physics" in data and "mesh" in data:
                        configs[0] = data

                # Collect ReviseAgent configs (revision 1, 2, ...)
                if "revise" in sender_name:
                    if "revised_json" in data:
                        revision_counter += 1
                        configs[revision_counter] = data["revised_json"]

                # Collect TOAgent screenshot dirs and revision numbers
                if "toagent" in sender_name or "to_agent" in sender_name:
                    if "screenshot_dir" in data:
                        rev = data.get("state", {}).get("revision", len(screenshots))
                        screenshots[rev] = data["screenshot_dir"]

                # Collect AI_Judge scores
                if "judge" in sender_name:
                    if "analysis" in data and data["analysis"]:
                        analysis_text = data["analysis"]
                        self._parse_judge_scores(analysis_text, judge_data)

            # Merge everything into iteration list
            all_revisions = sorted(set(
                list(configs.keys()) + list(screenshots.keys())
            ))

            for rev in all_revisions:
                entry = {"revision": rev}

                if rev in configs:
                    entry["config_json"] = configs[rev]

                if rev in screenshots:
                    entry["screenshot_path"] = screenshots[rev] + "/stress/front.png"
                elif rev == 0:
                    entry["screenshot_path"] = "screenshots/stress/front.png"

                if rev in judge_data:
                    entry["judge_score"] = judge_data[rev].get("score")
                    entry["judge_justification"] = judge_data[rev].get("justification")

                iterations.append(entry)

            print(f"✅ Retrieved {len(iterations)} iteration(s) of history")

        except Exception as e:
            print(f"⚠️ Error retrieving iteration history: {e}")

        return iterations

    def _parse_judge_scores(self, analysis_text: str, judge_data: dict):
        """
        Parse judge analysis text to extract per-image scores.
        Maps Image A -> revision 0, Image B -> revision 1, etc.
        """
        image_to_revision = {
            "A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5,
            "G": 6, "H": 7, "I": 8, "J": 9
        }

        # Accept both "Design A:" (dual-image prompt) and "Image A:" (legacy).
        score_pattern = r"(?:Design|Image)\s+([A-J]):\s*Score\s*[-–]\s*(\d+)"
        score_matches = re.findall(score_pattern, analysis_text)

        justification_pattern = (
            r"(?:Design|Image)\s+([A-J]).*?Justification:\s*(.*?)(?=\*\*\d+\.|$)"
        )
        justification_matches = re.findall(
            justification_pattern, analysis_text, re.DOTALL
        )

        justifications = {}
        for letter, text in justification_matches:
            justifications[letter] = text.strip()

        for letter, score in score_matches:
            if letter in image_to_revision:
                rev = image_to_revision[letter]
                judge_data[rev] = {
                    "score": int(score),
                    "justification": justifications.get(letter, "")
                }

    def _find_best_iteration(self, iteration_history: list) -> Optional[dict]:
        """
        Find the iteration with the highest judge score.
        Returns the best entry dict or None if no scores exist.
        """
        scored = [e for e in iteration_history if "judge_score" in e]
        if not scored:
            return None
        return max(scored, key=lambda e: e["judge_score"])

    # =========================================================================
    # Image encoding helper
    # =========================================================================

    def _encode_image(self, image_path: str) -> Optional[str]:
        """
        Load an image, convert to grayscale, thumbnail, and return base64 JPEG.
        Returns None if file doesn't exist.
        """
        if not os.path.exists(image_path):
            return None

        try:
            with Image.open(image_path) as img:
                img = img.convert("L")
                img.thumbnail((256, 256))
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=70)
                return base64.b64encode(buffer.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"⚠️ Failed to encode image {image_path}: {e}")
            return None

    # =========================================================================
    # Task text builder (problem-type dependent)
    # =========================================================================

    def _build_task_text(self) -> str:
        """Build the trailing task description, branching on PROBLEM_TYPE."""
        if PROBLEM_TYPE == "phone":
            goal_line = (
                "Your goal is to increase structural skeletonization AND skinny branching "
                "complexity. This means:\n"
            )
            rules_block = (
                "### Rules: Strictly follow these restrictions.\n"
                "1. The design MUST remain functional as a phone stand where the phone "
                "lies along the diagonal surface.\n"
                "2. Volume fraction may be changed to meet the aesthetic requirement.\n"
                "3. Do NOT go below 1.5 in density filter radius.\n"
            )
        else:  # cantilever
            goal_line = (
                "Your goal is to make the structure look more ORGANIC with skeletonization "
                "AND branching complexity. This means:\n"
            )
            rules_block = (
                "### Rules: Strictly follow these restrictions.\n"
                "1. Volume fraction may be changed to meet the aesthetic requirement.\n"
                "2. Do NOT go below 1.5 in density filter radius.\n"
                "3. Do not change the mesh resolution or domain size"
            )

        return (
            "\n### Task\n"
            "Analyze the optimized structure shown in the image(s) and recommend specific, "
            "actionable changes to the optimization parameters that will produce a "
            "significantly more skeletal, highly branched structure with many thin members "
            "and multiple internal load paths.\n\n"
            + goal_line +
            "- promoting formation of thin structural members instead of thick plates\n"
            "- encouraging splitting into multiple load paths rather than a single "
            "dominant path\n"
            "- increasing internal structural hierarchy and connectivity\n"
            "- reducing continuous sheet-like regions and promoting truss-like behavior\n\n"
            + rules_block
        )

    # =========================================================================
    # Core Analysis Method
    # =========================================================================

    def identify_problem_type(
        self,
        screenshot_path: str = 'screenshots/stress/front.png',
        problem_description: Optional[str] = None,
        judge_analysis: Optional[str] = None,
        iteration_history: Optional[list] = None,
    ) -> Optional[str]:
        """
        Identify structural problem characteristics from optimization result screenshot.

        When iteration_history is provided, ALL previous images are included in the
        prompt so the model can visually correlate parameter changes with structural
        outcomes and judge scores.

        Args:
            screenshot_path: Path to the CURRENT (latest) screenshot
            problem_description: Optional text description of the problem setup
            judge_analysis: Optional analysis from AI_Judge (fallback when no history)
            iteration_history: Full history of all iterations with configs, scores,
                              feedback, and screenshot paths

        Returns:
            Analysis text or None if analysis fails
        """
        if self.llm_config is None:
            print("⚠️ No LLM configured, skipping analysis")
            return None

        config = self.llm_config["config_list"][0]
        max_tokens = self.llm_config.get("max_tokens", config.get("max_tokens", 300))

        client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])

        # Verify at least one of the paired screenshots exists (depth or stress).
        # Attachment is deferred to _append_paired_images so both can be sent
        # to the model with labels.
        if not (os.path.exists(screenshot_path)
                or os.path.exists(self._paired_path(screenshot_path))):
            print(f"⚠️ Neither stress nor depth screenshot found "
                  f"(checked {screenshot_path} and "
                  f"{self._paired_path(screenshot_path)})")
            return None

        # ----- Build prompt text -----
        base_prompt = (
            "You are analyzing the result of a **3D topology optimization problem**.\n\n"
        )

        if problem_description:
            base_prompt += f"### Problem Setup\n{problem_description}\n\n"
        else:
            base_prompt += (
                "### Design Problem Description\n"
                "In this setup, the finite element analysis (FEA) uses linear elasticity "
                "physics with a Young's modulus E = 1.0 and a Poisson's ratio ν = 0.3. "
                "A three-dimensional structured mesh is defined with nx = 128, ny = 64, "
                "nz = 16 elements and physical dimensions lx = 1.0, ly = 0.5, lz = 0.125. "
                "The mesh is linked to the linear elastic physics so the material properties "
                "apply uniformly throughout the domain. A multigrid solver is configured with "
                "tol = 1e-4, maxiter = 50, and n_level = 5. One Dirichlet boundary condition "
                "is imposed: the bottom face (y = 0), where all nodes are fixed in the ux, uy, "
                "and uz directions. One force is applied along a diagonal: with a force of -1.0 "
                "in the y-direction distributed among all nodes. Use a density filter with "
                "radius 2.0. Set up a minimum compliance topology optimization problem with a "
                "volume fraction of 15%, SIMP penalty of 3.0, void material stiffness of 1e-9, "
                "and enable Heaviside projection. Use the PGD optimizer with a function tolerance "
                "of 1e-4 and no change tolerance.\n\n"
            )

        # ----- Build message content (text + images interleaved) -----
        content = []

        # === PATH A: Full iteration history with images ===
        if iteration_history and len(iteration_history) > 0:
            history_text = "### Complete Iteration History\n"
            history_text += (
                "Below is the complete history of all previous optimization iterations. "
                "Each entry shows the TO parameters used, the AI Judge's score (1-5) with "
                "feedback, and the resulting structure image. Study the visual progression "
                "to understand which parameter changes improved or worsened the structure.\n\n"
            )
            base_prompt += history_text
            content.append({"type": "text", "text": base_prompt})

            # Add each iteration: text label + image
            for entry in iteration_history:
                rev = entry.get("revision", "?")

                # Build text block for this iteration
                iter_text = f"#### Revision {rev}\n"

                if "config_json" in entry:
                    iter_text += f"**Parameters**: {json.dumps(entry['config_json'], indent=2)}\n"

                if "judge_score" in entry:
                    iter_text += f"**Judge Score**: {entry['judge_score']}/5\n"
                if "judge_justification" in entry:
                    iter_text += f"**Judge Feedback**: {entry['judge_justification']}\n"

                iter_text += f"**Structure (Revision {rev})**:\n"
                content.append({"type": "text", "text": iter_text})

                # Add BOTH depth and stress images for this iteration
                img_path = entry.get("screenshot_path")
                if img_path:
                    n = self._append_paired_images(
                        content, img_path, label=f"Revision {rev}"
                    )
                    if n == 0:
                        content.append({
                            "type": "text",
                            "text": f"[Images not available for revision {rev}]\n"
                        })
                    else:
                        print(f"   📷 Attached {n} image(s) for revision {rev}: "
                              f"{img_path}")
                else:
                    content.append({
                        "type": "text",
                        "text": f"[No screenshot path for revision {rev}]\n"
                    })

                content.append({"type": "text", "text": "\n"})

            # Identify and highlight the best iteration
            best = self._find_best_iteration(iteration_history)
            if best:
                best_text = (
                    f"### Best Iteration So Far\n"
                    f"Revision {best['revision']} scored highest at "
                    f"{best['judge_score']}/5.\n"
                )

                if "config_json" in best:
                    best_text += f"**Best Parameters**: {json.dumps(best['config_json'], indent=2)}\n"


                if "judge_justification" in best:
                    best_text += (
                        f"**Best Feedback**: {best['judge_justification']}\n"
                    )
                best_text += (
                    "\n**IMPORTANT**: Base your suggested changes on this best-scoring "
                    "configuration, NOT the most recent one (unless they are the same). "
                    "Build on what worked best.\n\n"
                )

                best_text += (
                    "\n**IMPORTANT**: Do not change the mesh resolution or domain size. \n\n"
                )
                content.append({"type": "text", "text": best_text})

            # Learning instructions
            learning_text = (
                "### Learning from History\n"
                "Based on the scores, feedback, and images above:\n"
                "1. Visually compare the structures — which ones have more branching, "
                "thinner members, and greater complexity?\n"
                "2. Identify which parameter changes correlated with higher scores.\n"
                "3. Identify which changes led to worse results.\n"
                "4. Build on what worked and avoid repeating what didn't.\n"
                "5. Be specific about which parameters to change and by how much.\n\n"
            )
            content.append({"type": "text", "text": learning_text})

            # Add the CURRENT (latest) screenshot last — both depth and stress
            current_text = (
                "### Current Structure (Latest Iteration)\n"
                "This is the most recent optimization result to analyze:\n"
            )
            content.append({"type": "text", "text": current_text})
            self._append_paired_images(content, screenshot_path,
                                        label="Latest iteration")

        # === PATH B: Single judge analysis fallback ===
        elif judge_analysis:
            base_prompt += (
                "### Previous Evaluation from AI Judge\n"
                "The following is the most recent comparative analysis from the AI Judge, "
                "which evaluated the structural complexity of previous optimization results. "
                "Use this feedback to inform your suggestions — consider what worked, what "
                "didn't, and how to further improve the structure in the direction the judge "
                "is scoring for.\n\n"
                f"{judge_analysis}\n\n"
                "### Important\n"
                "Take the judge's feedback into account. If the judge indicated the structure "
                "is not complex enough, suggest more aggressive changes. If the judge noted "
                "improvement, continue in that direction but push further.\n\n"
            )
            content.append({"type": "text", "text": base_prompt})
            self._append_paired_images(content, screenshot_path,
                                        label="Latest iteration")

        # === PATH C: No history (first iteration) ===
        else:
            content.append({"type": "text", "text": base_prompt})
            self._append_paired_images(content, screenshot_path,
                                        label="Latest iteration")

        # ----- Task description (problem-type dependent, always appended at end) -----
        content.append({"type": "text", "text": self._build_task_text()})

        # ----- API call -----
        messages = [{"role": "user", "content": content}]

        try:
            response = client.chat.completions.create(
                model=config["model"],
                messages=messages,
                temperature=0.2,
                max_tokens=max_tokens
            )

            description = response.choices[0].message.content
            print("\n📝 Problem Type Identification:")
            print("-" * 60)
            print(description)
            print("-" * 60 + "\n")
            return description

        except Exception as e:
            print(f"⚠️ Identification failed: {e}")
            return None

    # =========================================================================
    # Reply Handler
    # =========================================================================

    def _generate_analysis_reply(self, messages=None, sender=None, config=None):
        last_msg = self._oai_messages[sender][-1]

        try:
            # Get screenshot_dir from TOAgent history
            screenshot_dir = self.retrieve_latest_screenshot_dir_from_history()
            screenshot_path = screenshot_dir + "/stress/front.png"

            # Get problem description from history
            problem_description = None
            try:
                problem_description = self.retrieve_problem_description_from_history()
            except json.JSONDecodeError:
                pass

            # Retrieve full iteration history
            iteration_history = self.retrieve_full_iteration_history()

            # Fallback: get latest judge analysis if no iteration history
            judge_analysis = None
            if not iteration_history:
                judge_analysis = self.retrieve_judge_analysis_from_history()
                if judge_analysis:
                    print("📋 Including AI_Judge feedback in analysis prompt")
                else:
                    print("ℹ️ No AI_Judge history found (first iteration)")
            else:
                num_images = sum(
                    1 for e in iteration_history
                    if e.get("screenshot_path") and os.path.exists(e.get("screenshot_path", ""))
                )
                print(f"📋 Including full iteration history "
                      f"({len(iteration_history)} iterations, {num_images} images)")

            # Run analysis
            analysis = self.identify_problem_type(
                screenshot_path=screenshot_path,
                problem_description=problem_description,
                judge_analysis=judge_analysis,
                iteration_history=iteration_history,
            )

            # Determine the best config to base revisions on
            best_entry = None
            base_revision = None
            base_config = None
            if iteration_history:
                best_entry = self._find_best_iteration(iteration_history)
                if best_entry:
                    base_revision = best_entry["revision"]
                    base_config = best_entry.get("config_json")
                    print(f"🏆 Best iteration: revision {base_revision} "
                          f"(score: {best_entry.get('judge_score', '?')}/5)")

            response_data = {
                "status": "Analysis complete",
                "screenshot_path": screenshot_path,
                "analysis": analysis,
            }

            # Include best config info for ReviseAgent to use
            if base_revision is not None:
                response_data["base_revision"] = base_revision
            if base_config is not None:
                response_data["base_config"] = base_config

            return True, {
                "role": "assistant",
                "content": json.dumps(response_data)
            }

        except Exception as e:
            import traceback
            return True, {
                "role": "assistant",
                "content": json.dumps({
                    "error": f"VisionAgent failed: {str(e)}",
                    "traceback": traceback.format_exc()
                })
            }
