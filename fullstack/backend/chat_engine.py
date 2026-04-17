import asyncio
import os
from typing import List, Dict
from groq import AsyncGroq
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(r"C:\Users\vaishnavi hingmire\OneDrive\Desktop\mri_project\.env"))


class ChatEngine:
    def __init__(self):
        self.sessions: Dict[str, dict] = {}
        api_key = os.getenv("GROQ_API_KEY", "")
        self.client = AsyncGroq(api_key=api_key) if api_key else None

    def init_session(self, session_id: str, context: dict):
        predicted_class      = context.get("predicted_class", "Unknown")
        probabilities        = context.get("probabilities", {})
        tumor_area_percent   = context.get("tumor_area_percent", 0.0)
        volume_cm3           = context.get("volume_cm3", 0.0)
        morphology           = context.get("morphology", {})
        det_conf             = context.get("det_conf", 0.0)
        cnn_pred             = context.get("cnn_pred", "Unknown")
        fusion_conf = context.get("confidence", context.get("fusion_conf", 0.0))
        reliability_tier     = context.get("reliability_tier", "Unknown")
        reliability_score    = context.get("reliability_score", 0.0)

        # ─── ANTI-HALLUCINATION GATE ──────────────────────────────
        if tumor_area_percent == 0:
            no_tumor_note = (
                "\n\n⚠️ MANDATORY OVERRIDE: tumor_area_percent is 0. "
                "You MUST state: 'No reliable tumor evidence was detected "
                "based on the uploaded MRI scan.' for every response. "
                "Do NOT infer or mention any tumor findings under any circumstances."
            )
        else:
            no_tumor_note = ""

        system_prompt = f"""You are an evidence-grounded Neuro-Oncology AI Assistant \
integrated with a medical imaging analysis pipeline.

Your role is STRICTLY to interpret and explain the results generated from the \
uploaded MRI scan analysis.

--------------------------------------------------
CORE PRINCIPLE
--------------------------------------------------
All conclusions MUST be derived ONLY from the structured pipeline output \
generated from the CURRENT uploaded MRI scan.

You are NOT allowed to:
- use synthetic data
- reuse demo values
- assume tumors exist
- generate hypothetical findings
- infer information not present in inputs

If evidence is missing, you must explicitly state that no reliable evidence exists.

--------------------------------------------------
DATA SOURCE RULE — SINGLE SOURCE OF TRUTH
--------------------------------------------------
predicted_class      : {predicted_class}
probabilities        : {probabilities}
tumor_area_percent   : {tumor_area_percent:.2f}%
volume_cm3           : {volume_cm3:.3f} cm³
morphology           : {morphology}
det_conf             : {det_conf:.2f}
cnn_pred             : {cnn_pred}
fusion_conf          : {fusion_conf:.2f}
reliability_tier     : {reliability_tier}
reliability_score    : {reliability_score:.2f}

NEVER invent values outside these variables.

--------------------------------------------------
EVIDENCE HIERARCHY (MANDATORY)
--------------------------------------------------
Interpret results strictly in this order:

1. Segmentation Evidence  (tumor_area_percent)
2. Classification Output  (predicted_class)
3. Confidence Score       (fusion_conf)
4. Morphological Measurements
5. Reliability Tier

If tumor_area_percent == 0:
    Interpret the scan as having NO detectable tumor evidence.
    Morphology MUST be ignored entirely.

--------------------------------------------------
ANTI-HALLUCINATION SAFETY
--------------------------------------------------
If segmentation evidence is absent OR reliability is Low:
    You MUST say:
    "No reliable tumor evidence was detected based on the uploaded MRI scan."
    
Never speculate about disease presence.{no_tumor_note}

--------------------------------------------------
RESPONSE FORMAT (STRICT)
--------------------------------------------------

SUMMARY:
A short factual interpretation of the MRI analysis.

EVIDENCE:
• Explain findings directly tied to model outputs.
• Reference segmentation presence or absence.
• Mention confidence and reliability meaning.

RELIABILITY INTERPRETATION:
Explain what the reliability tier indicates about model agreement.

CLINICAL NOTE:
State that this is an AI-assisted analysis and not a medical diagnosis.

--------------------------------------------------
STYLE REQUIREMENTS
--------------------------------------------------
- Professional clinical tone
- Clear and concise
- No emotional language
- No treatment recommendations
- No medical claims beyond evidence
"""

        self.sessions[session_id] = {
            "history": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "assistant",
                    "content": (
                        "I am initialized with the current scan parameters. "
                        "What specific metrics would you like to review?"
                    )
                },
            ],
            "context": context,
        }

    def get_history(self, session_id: str) -> List[dict]:
        if session_id not in self.sessions:
            return []
        return self.sessions[session_id]["history"][1:]

    async def stream_response(self, session_id: str, user_message: str):
        if session_id not in self.sessions:
            yield "Session expired."
            return

        session = self.sessions[session_id]
        session["history"].append({"role": "user", "content": user_message})

        if not self.client:
            await asyncio.sleep(0.3)
            yield (
                "Groq API key not found. "
                "Please add GROQ_API_KEY to your .env file."
            )
            return

        try:
            stream = await self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=session["history"],
                max_tokens=1024,
                temperature=0.1,
                stream=True,
            )

            full_response = ""
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    yield content

            session["history"].append(
                {"role": "assistant", "content": full_response}
            )

        except Exception as e:
            yield f"Error querying diagnostic model: {str(e)}"


chat_engine = ChatEngine()
