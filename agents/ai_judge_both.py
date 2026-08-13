# # CHANGED TO ENABLE MULTIPLE PASSES

# import json
# import os
# from typing import Optional, List  # 🔄 Added List
# from PIL import Image
# from autogen import UserProxyAgent, Agent
# from google import genai
# from google.genai import types
# import time


# class AI_Judge(UserProxyAgent):
#     """
#     Tool-style AutoGen agent that evaluates topology optimization screenshots using Gemini Vision.
#     Dynamically retrieves all screenshot directories from previous TOAgent messages.
#     """

#     def __init__(
#         self,
#         name="AI_Judge",
#         system_message=None,
#         human_input_mode="NEVER",
#         code_execution_config={"use_docker": False},
#         api_key: str | None = None,
#         temperature: float = 0.0,
#         seed: int | None = None,
#         **kwargs
#     ):
#         if api_key is None:
#             raise ValueError("api_key is required for Gemini SDK usage")

#         super().__init__(
#             name=name,
#             system_message=system_message,
#             human_input_mode=human_input_mode,
#             code_execution_config=code_execution_config,
#             llm_config=False,
#             **kwargs
#         )

#         self.temperature = temperature
#         self.seed = seed
#         self.gemini_client = genai.Client(api_key=api_key)
#         self.register_reply(Agent, AI_JudgeBoth._generate_analysis_reply, position=2)

#     # 🔄 NEW METHOD
#     def retrieve_screenshot_dirs_from_history(self) -> List[str]:
#         """
#         Retrieve all screenshot directories from previous TOAgent messages in history.
#         Returns list of screenshot_dir paths in chronological order.
#         """
#         screenshot_dirs = []

#         try:
#             chat_manager_messages = self._oai_messages.get(
#                 list(self._oai_messages.keys())[0], []
#             )

#             for msg in chat_manager_messages:  # chronological order
#                 sender_name = msg.get("name", "").lower()
#                 content = msg.get("content", "")

#                 if "toagent" in sender_name or "to_agent" in sender_name:
#                     try:
#                         data = json.loads(content)
#                         if "screenshot_dir" in data:
#                             screenshot_dirs.append(data["screenshot_dir"])
#                             print(f"✅ Found screenshot_dir: {data['screenshot_dir']}")
#                     except json.JSONDecodeError:
#                         continue

#         except Exception as e:
#             print(f"⚠️ Error retrieving screenshot dirs: {e}")
#             import traceback
#             traceback.print_exc()

#         print(f"📸 Found {len(screenshot_dirs)} screenshot directories: {screenshot_dirs}")
#         return screenshot_dirs

#     def compare_structures(
#         self,
#         screenshot_dirs: List[str],  # 🔄 Changed from two hardcoded paths to list
#     ) -> Optional[str]:
#         """
#         Compare topology optimization results across all available screenshot directories.
#         """
#         # 🔄 Check we have enough directories
#         if len(screenshot_dirs) < 2:
#             print("⚠️ Need at least 2 screenshot directories to compare")
#             return None

#         # 🔄 Build image list dynamically from all directories
#         images = []
#         labels = []
#         for i, sdir in enumerate(screenshot_dirs):
#             path = os.path.join(sdir, "front.png")  # 🔄 Construct path from dir
#             if not os.path.exists(path):
#                 print(f"⚠️ Screenshot not found: {path}")
#                 continue
#             images.append(Image.open(path))
#             label = chr(65 + i)  # 🔄 A, B, C, D, ...
#             labels.append(label)
#             print(f"📷 Image {label}: {path}")

#         if len(images) < 2:
#             print("⚠️ Not enough valid screenshots found")
#             for img in images:
#                 img.close()
#             return None

#         # 🔄 Build dynamic image list for prompt
#         image_list_str = "\n".join(
#             [f"- **Image {label}**: Topology optimization result {i+1} (from `{screenshot_dirs[i]}`)"
#              for i, label in enumerate(labels)]
#         )

#         # 🔄 Build dynamic scoring list for prompt
#         scoring_list_str = "\n".join(
#             [f"{i+1}. A numeric score (1–5) for Image {label}"
#              for i, label in enumerate(labels)]
#         )

#         # 🔄 Build dynamic confidence list for prompt
#         confidence_list_str = "\n".join(
#             [f"- A confidence score (0–100%) for Image {label}"
#              for label in labels]
#         )

#         # 🔄 Prompt now references dynamic number of images
#         prompt = (
#             "You are an expert judge in topology optimization, specializing in evaluating structural morphology.\n\n"

#             "### Task\n"
#             f"You are shown **{len(images)} images**:\n"  # 🔄
#             f"{image_list_str}\n\n"  # 🔄

#             "Your task is to evaluate the **aesthetic quality only** of the structure shown in each image, "
#             "**strictly based on perceived global structural complexity and richness at the global scale**.\n\n"

#             "### Critical Instruction (Priority Rule)\n"
#             "**Global structural organization takes absolute priority over local detail.**\n"
#             "If a structure exhibits a highly branched, intricate, and hierarchically rich global form, "
#             "it should score highly **even if** some members are thick, coarse, or locally simple.\n"
#             "Do NOT reward local intricacy unless it meaningfully contributes to increased "
#             "overall perceived global structural complexity.\n\n"

#             "### What to Focus On\n"
#             "- Overall form and silhouette complexity\n"
#             "- Richness and hierarchy of load-carrying members\n"
#             "- Number and diversity of major branches at the global level\n"
#             "- Interconnection, redundancy, and visual density of the full structure\n\n"

#             "### What to De-emphasize or Ignore\n"
#             "- Fine surface texture or ornamental micro-detail\n"
#             "- Small-scale filigree that does not alter global branching\n"
#             "- Voxelization, mesh resolution, or pixel-level artifacts\n"
#             "- Local complexity that does NOT increase the global load-path network\n\n"

#             "### Evaluation Criterion: Structural Complexity\n"
#             "Assess how **complex, highly branched, and globally intricate the structure appears overall**. "
#             "A structure with a dense, richly interconnected global topology should score higher than a "
#             "minimal or sparsely branched one, **even if the complex structure appears locally coarse or heavy**.\n\n"

#             "### Scoring (1–5)\n"
#             f"For **each image**, assign a score from **1 to 5** based on **global structural complexity**:\n"
#             "- **1**: extremely simple global structure; very few branches, strong minimalism, little hierarchy.\n"
#             "- **2**: low global complexity; limited branching with clear dominant members.\n"
#             "- **3**: moderate global complexity; noticeable branching with some hierarchy.\n"
#             "- **4**: high global complexity; many major branches and a rich, multi-level hierarchy.\n"
#             "- **5**: extremely complex and intricate at the global level; dense branching network, "
#             "multiple competing load paths, and exceptional structural richness.\n\n"

#             "### Tie-Breaking Rule\n"
#             "If one image appears globally more complex but locally simpler than another, "  # 🔄 "another" instead of "the other"
#             "**the globally more complex structure MUST receive the higher score**.\n\n"

#             "Use the full range of the scale. If images differ in global complexity, "  # 🔄 generalized
#             "the scores should differ accordingly.\n\n"

#             "### Output\n"
#             f"{scoring_list_str}\n"  # 🔄
#             f"{len(labels) + 1}. A concise, specific justification focused on **global structure** "
#             "(e.g., number of major branches, richness of hierarchy, overall network density)\n"
#             f"{confidence_list_str}\n"  # 🔄
#         )

#         config_params = {"temperature": self.temperature}
#         if self.seed is not None:
#             config_params["seed"] = self.seed

#         # try:
#         #     response = self.gemini_client.models.generate_content(
#         #         model="gemini-3-flash-preview",
#         #         contents=[prompt] + images,  # 🔄 Dynamic list of images
#         #         config=types.GenerateContentConfig(**config_params),
#         #     )

#         #     analysis = response.text

#         #     print("\n📝 Comparative Analysis:")
#         #     print("-" * 60)
#         #     print(analysis)
#         #     print("-" * 60 + "\n")

#         #     return analysis

#         # except Exception as e:
#         #     print(f"⚠️ Gemini vision analysis failed: {e}")
#         #     return None

    
#         # ADD SOME DELAY TO CHILL OUT GEMINI 

#             max_retries = 3
#             for attempt in range(max_retries):
#                 try:
#                     response = self.gemini_client.models.generate_content(
#                         model="gemini-3-flash-preview",
#                         contents=[prompt] + images,
#                         config=types.GenerateContentConfig(**config_params),
#                     )
    
#                     analysis = response.text
    
#                     print("\n📝 Comparative Analysis:")
#                     print("-" * 60)
#                     print(analysis)
#                     print("-" * 60 + "\n")
    
#                     return analysis
    
#                 except Exception as e:
#                     error_str = str(e)
#                     if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
#                         wait_time = 30 * (attempt + 1)  # 30s, 60s, 90s
#                         print(f"⏳ Rate limited. Waiting {wait_time}s before retry ({attempt+1}/{max_retries})...")
#                         time.sleep(wait_time)
#                     else:
#                         print(f"⚠️ Gemini vision analysis failed: {e}")
#                         return None
    
#             print("⚠️ Exhausted all retries after rate limiting")
#             return None

#     def _generate_analysis_reply(self, messages=None, sender=None, config=None):
#         """
#         AutoGen reply hook.
#         Dynamically retrieves screenshot dirs from history and runs comparison.
#         """
#         try:
#             screenshot_dirs = self.retrieve_screenshot_dirs_from_history()

#             if len(screenshot_dirs) < 2:
#                 return True, {
#                     "role": "assistant",
#                     "content": json.dumps({
#                         "error": f"Need at least 2 screenshot directories, found {len(screenshot_dirs)}",
#                         "screenshot_dirs_found": screenshot_dirs
#                     })
#                 }

#             analysis = self.compare_structures(screenshot_dirs)

#             # 🔄 Include mode in output
#             mode = "progression" if len(screenshot_dirs) > 2 else "comparison"

#             return True, {
#                 "role": "assistant",
#                 "content": json.dumps({
#                     "status": "comparison_complete",
#                     "mode": mode,
#                     "screenshot_dirs": screenshot_dirs,
#                     "num_images_compared": len(screenshot_dirs),
#                     "analysis": analysis,
#                 })
#             }

#         except Exception as e:
#             import traceback
#             return True, {
#                 "role": "assistant",
#                 "content": json.dumps({
#                     "error": str(e),
#                     "traceback": traceback.format_exc(),
#                 })
#             }


# CHANGED FOR DIFFERENT MODELS 

import json
import os
from typing import Optional, List  # 🔄 Added List
from PIL import Image
from autogen import UserProxyAgent, Agent
from google import genai
from google.genai import types
import time
import re
from datetime import datetime


class AI_JudgeBoth(UserProxyAgent):
    """
    Duplicate of AI_Judge that compares revisions using BOTH the depth-shaded
    and the stress-field screenshots from each design.

    Each screenshot_dir is expected to contain `depth/front.png` and
    `stress/front.png` (the layout produced by _3d.py = _3d_both.py).
    For each design we pass both images to the model with clear labels so
    the judge can reason about morphology AND mechanical performance.
    """

    def __init__(
        self,
        name="AI_JudgeBoth",
        system_message=None,
        human_input_mode="NEVER",
        code_execution_config={"use_docker": False},
        api_key: str | None = None,
        temperature: float = 0.0,
        seed: int | None = None,
        backend: str = "gemini",
        **kwargs
    ):
        self.backend = backend
        self.temperature = temperature
        self.seed = seed
    
        if backend == "gemini":
            if api_key is None:
                raise ValueError("api_key is required for Gemini")
            from google import genai
            self.gemini_client = genai.Client(api_key=api_key)
            self.model = "gemini-3-flash-preview"
    
        elif backend == "local":
            from openai import OpenAI
            llm_cfg = kwargs.pop("llm_config", {})
            config = llm_cfg["config_list"][0]
            self.local_client = OpenAI(
                base_url=config["base_url"],
                api_key=config.get("api_key", "dummy"),
            )
            self.model = config["model"]
    
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
        super().__init__(
            name=name,
            system_message=system_message,
            human_input_mode=human_input_mode,
            code_execution_config=code_execution_config,
            llm_config=False,
            **kwargs
        )
    
        self.register_reply(Agent, AI_JudgeBoth._generate_analysis_reply, position=2)

    # 🔄 NEW METHOD
    def retrieve_screenshot_dirs_from_history(self) -> List[str]:
        """
        Retrieve all screenshot directories from previous TOAgent messages in history.
        Returns list of screenshot_dir paths in chronological order.
        """
        screenshot_dirs = []

        try:
            chat_manager_messages = self._oai_messages.get(
                list(self._oai_messages.keys())[0], []
            )

            for msg in chat_manager_messages:  # chronological order
                sender_name = msg.get("name", "").lower()
                content = msg.get("content", "")

                if "toagent" in sender_name or "to_agent" in sender_name:
                    try:
                        data = json.loads(content)
                        if "screenshot_dir" in data:
                            screenshot_dirs.append(data["screenshot_dir"])
                            print(f"✅ Found screenshot_dir: {data['screenshot_dir']}")
                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            print(f"⚠️ Error retrieving screenshot dirs: {e}")
            import traceback
            traceback.print_exc()

        print(f"📸 Found {len(screenshot_dirs)} screenshot directories: {screenshot_dirs}")
        return screenshot_dirs

    def compare_structures(
        self,
        screenshot_dirs: List[str],  # 🔄 Changed from two hardcoded paths to list
    ) -> Optional[str]:
        """
        Compare topology optimization results across all available screenshot directories.
        """
        # 🔄 Check we have enough directories
        if len(screenshot_dirs) < 2:
            print("⚠️ Need at least 2 screenshot directories to compare")
            return None

        # Build paired (depth + stress) image list per design.
        # `images` is a flat list of PIL images in capture order;
        # `image_meta` parallels it with the design letter + view name so we
        # can interleave labels in the LLM prompt.
        images = []
        image_meta = []   # list of {'label': 'A', 'view': 'depth'|'stress', 'path': str}
        labels = []       # one letter per design that contributed >=1 image
        for i, sdir in enumerate(screenshot_dirs):
            design_label = chr(65 + i)
            pair_count = 0
            for view in ("depth", "stress"):
                path = os.path.join(sdir, view, "front.png")
                if not os.path.exists(path):
                    print(f"⚠️ Missing {view} screenshot for design "
                          f"{design_label}: {path}")
                    continue
                images.append(Image.open(path))
                image_meta.append({'label': design_label, 'view': view, 'path': path})
                pair_count += 1
                print(f"📷 Design {design_label} ({view}): {path}")
            if pair_count > 0:
                labels.append(design_label)

        if len(labels) < 2:
            print("⚠️ Need at least 2 designs with at least one view each")
            for img in images:
                img.close()
            return None

        # Designs list for the prompt (one entry per design, both views inline).
        image_list_str = "\n".join(
            [f"- **Design {label}**: Topology optimization result {i+1} "
             f"(from `{screenshot_dirs[i]}`) — shown via both depth-shaded "
             f"and stress-field views below"
             for i, label in enumerate(labels)]
        )

        # ONE score per design, not per image.
        scoring_list_str = "\n".join(
            [f"{i+1}. A numeric score (1–5) for Design {label}"
             for i, label in enumerate(labels)]
        )

        confidence_list_str = "\n".join(
            [f"- A confidence score (0–100%) for Design {label}"
             for label in labels]
        )

        # 🔄 Prompt now references dynamic number of images
        prompt = (
            "You are an expert judge in topology optimization, specializing in evaluating structural morphology.\n\n"

            "### Task\n"
            f"You are shown **{len(images)} images**:\n"  # 🔄
            f"{image_list_str}\n\n"  # 🔄

            "Your task is to evaluate the **aesthetic quality only** of the structure shown in each image, "
            "**strictly based on perceived global structural complexity and richness at the global scale**.\n\n"

            "### Critical Instruction (Priority Rule)\n"
            "**Global structural organization takes absolute priority over local detail.**\n"
            "If a structure exhibits a highly branched, intricate, and hierarchically rich global form, "
            "it should score highly **even if** some members are thick, coarse, or locally simple.\n"
            "Do NOT reward local intricacy unless it meaningfully contributes to increased "
            "overall perceived global structural complexity.\n\n"

            "### What to Focus On\n"
            "- Overall form and silhouette complexity\n"
            "- Richness and hierarchy of load-carrying members\n"
            "- Number and diversity of major branches at the global level\n"
            "- Interconnection, redundancy, and visual density of the full structure\n\n"

            "### What to De-emphasize or Ignore\n"
            "- Fine surface texture or ornamental micro-detail\n"
            "- Small-scale filigree that does not alter global branching\n"
            "- Voxelization, mesh resolution, or pixel-level artifacts\n"
            "- Local complexity that does NOT increase the global load-path network\n\n"

            "### Evaluation Criterion: Structural Complexity\n"
            "Assess how **complex, highly branched, and globally intricate the structure appears overall**. "
            "A structure with a dense, richly interconnected global topology should score higher than a "
            "minimal or sparsely branched one, **even if the complex structure appears locally coarse or heavy**.\n\n"

            "### Scoring (1–5)\n"
            f"For **each image**, assign a score from **1 to 5** based on **global structural complexity**:\n"
            "- **1**: extremely simple global structure; very few branches, strong minimalism, little hierarchy.\n"
            "- **2**: low global complexity; limited branching with clear dominant members.\n"
            "- **3**: moderate global complexity; noticeable branching with some hierarchy.\n"
            "- **4**: high global complexity; many major branches and a rich, multi-level hierarchy.\n"
            "- **5**: extremely complex and intricate at the global level; dense branching network, "
            "multiple competing load paths, and exceptional structural richness.\n\n"

            "### Tie-Breaking Rule\n"
            "If one image appears globally more complex but locally simpler than another, "  # 🔄 "another" instead of "the other"
            "**the globally more complex structure MUST receive the higher score**.\n\n"

            "Use the full range of the scale. If images differ in global complexity, "  # 🔄 generalized
            "the scores should differ accordingly.\n\n"

            "### Output\n"
            f"{scoring_list_str}\n"  # 🔄
            f"{len(labels) + 1}. A concise, specific justification focused on **global structure** "
            "(e.g., number of major branches, richness of hierarchy, overall network density)\n"
            f"{confidence_list_str}\n"  # 🔄
        )

        max_retries = 3
        for attempt in range(max_retries):
            try:
                if self.backend == "gemini":
                    from google.genai import types
                    config_params = {"temperature": self.temperature}
                    if self.seed is not None:
                        config_params["seed"] = self.seed
                    # Interleave a short text label before each image so
                    # the model knows which design + view each picture is.
                    contents = [prompt]
                    for img, meta in zip(images, image_meta):
                        contents.append(
                            f"Design {meta['label']} — {meta['view']} view:"
                        )
                        contents.append(img)
                    response = self.gemini_client.models.generate_content(
                        model=self.model,
                        contents=contents,
                        config=types.GenerateContentConfig(**config_params),
                    )
                    analysis = response.text

                elif self.backend == "local":
                    import base64
                    from io import BytesIO

                    content_parts = [{"type": "text", "text": prompt}]
                    for img, meta in zip(images, image_meta):
                        content_parts.append({
                            "type": "text",
                            "text": f"Design {meta['label']} — {meta['view']} view:"
                        })
                        buf = BytesIO()
                        img.save(buf, format="PNG")
                        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"}
                        })

                    response = self.local_client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": content_parts}],
                        temperature=self.temperature,
                    )
                    analysis = response.choices[0].message.content

                print("\n📝 Comparative Analysis:")
                print("-" * 60)
                print(analysis)
                print("-" * 60 + "\n")

                return analysis

            except Exception as e:
                error_str = str(e)
                if self.backend == "gemini" and ("429" in error_str or "RESOURCE_EXHAUSTED" in error_str):
                    wait_time = 30 * (attempt + 1)
                    print(f"⏳ Rate limited. Waiting {wait_time}s before retry ({attempt+1}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    print(f"⚠️ Vision analysis failed: {e}")
                    return None

        print("⚠️ Exhausted all retries after rate limiting")
        return None

    def _save_judge_scores(self, analysis: str, screenshot_dirs: List[str], save_path: str = "judge_history.json"):
        """
        Parse scores from analysis text and append to a JSON history file.
        """
        # Parse per-image scores: "Image A: Score - 3" or "Image A: 3" etc.
        # Match either "Design A" (new dual-image prompt) or "Image A" (legacy
        # single-image fallback) so old transcripts still parse.
        # LITE: the original pattern was
        #     (?:Design|Image)\s+([A-J])[\s:]*(?:Score\s*[-–:]?\s*)?(\d)
        # which had two defects that between them corrupted 11 of the 21 judge
        # entries in website/runs.
        #
        # 1. `[\s:]*` does not span markdown, so the very common
        #        **Design A:** 3
        #    never matched -- the `**` between the colon and the digit stops it.
        #    The only things that then DID match were the trailing confidence
        #    percentages ("Design A: 90%"), so entries stored 9 or 8 on a scale
        #    documented as 1-5. Six entries recorded a confidence as a score.
        # 2. `(\d)` accepts any digit, and the loop OVERWROTE, so the last match
        #    anywhere in the response won. Even when the header parsed
        #    correctly, prose later in the justification replaced it:
        #        raw : "Design A: 1\nDesign B: 4\n\n**Justification:** ..."
        #        was : {A: 1, B: 1}
        #        now : {A: 1, B: 4}
        #
        # The pattern below tolerates markdown and the two observed layouts
        #     **1. Design A: Score - 3**      (score interleaved with prose)
        #     **Design A:** 3                  (score header block)
        # restricts the value to the documented 1-5, and keeps the FIRST match
        # per design. Verified against every judge_history.json in website/runs:
        # 21/21 entries parse completely, versus 15/21 before, and 11 change.
        score_pattern = (r"(?:Design|Image)\s+([A-J])\s*[:\-–]?\s*\**\s*"
                         r"(?:Score\s*[-–:]?\s*)?\**\s*([1-5])\b")
        matches = re.findall(score_pattern, analysis, re.IGNORECASE)

        scores = {}
        for letter, score in matches:
            idx = ord(letter.upper()) - 65
            if idx >= len(screenshot_dirs):
                continue
            key = screenshot_dirs[idx]
            if key not in scores:          # first value wins, not last
                scores[key] = int(score)

        missing = [d for d in screenshot_dirs if d not in scores]
        if missing:
            print(f"   ⚠️  no score parsed for {len(missing)} design(s): "
                  f"{', '.join(missing)}. They are omitted rather than guessed — "
                  f"check the judge's response format.")
    
        entry = {
            "timestamp": datetime.now().isoformat(),
            "num_images": len(screenshot_dirs),
            "screenshot_dirs": screenshot_dirs,
            "scores": scores,
            "raw_analysis": analysis,
        }
    
        # Load existing history or start fresh
        history = []
        if os.path.exists(save_path):
            try:
                with open(save_path, "r") as f:
                    history = json.load(f)
            except (json.JSONDecodeError, IOError):
                history = []
    
        history.append(entry)
    
        with open(save_path, "w") as f:
            json.dump(history, f, indent=2)
    
        print(f"💾 Saved judge scores to {save_path} ({len(history)} total evaluations)")
        return scores

    def _generate_analysis_reply(self, messages=None, sender=None, config=None):
        """
        AutoGen reply hook.
        Dynamically retrieves screenshot dirs from history and runs comparison.
        """
        try:
            screenshot_dirs = self.retrieve_screenshot_dirs_from_history()

            if len(screenshot_dirs) < 2:
                return True, {
                    "role": "assistant",
                    "content": json.dumps({
                        "error": f"Need at least 2 screenshot directories, found {len(screenshot_dirs)}",
                        "screenshot_dirs_found": screenshot_dirs
                    })
                }

            analysis = self.compare_structures(screenshot_dirs)

            # # 🔄 Include mode in output
            # mode = "progression" if len(screenshot_dirs) > 2 else "comparison"

            # return True, {
            #     "role": "assistant",
            #     "content": json.dumps({
            #         "status": "comparison_complete",
            #         "mode": mode,
            #         "screenshot_dirs": screenshot_dirs,
            #         "num_images_compared": len(screenshot_dirs),
            #         "analysis": analysis,
            #     })
            # }

            scores = {}
            if analysis:
                scores = self._save_judge_scores(analysis, screenshot_dirs)

            mode = "progression" if len(screenshot_dirs) > 2 else "comparison"

            return True, {
                "role": "assistant",
                "content": json.dumps({
                    "status": "comparison_complete",
                    "mode": mode,
                    "screenshot_dirs": screenshot_dirs,
                    "num_images_compared": len(screenshot_dirs),
                    "scores": scores,          # ← now also in the message
                    "analysis": analysis,
                })
            }

        except Exception as e:
            import traceback
            return True, {
                "role": "assistant",
                "content": json.dumps({
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                })
            }