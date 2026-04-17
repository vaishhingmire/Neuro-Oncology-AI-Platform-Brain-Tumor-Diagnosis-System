import os 
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(dotenv_path=Path(r"C:\Users\vaishnavi hingmire\OneDrive\Desktop\mri_project\.env"))
from groq import Groq
from datetime import datetime
import json

def generate_neuro_report(detection_result: dict, classification: dict) -> str:
    """
    Generate clinical neuro-oncology report using LLM
    grounded to structured YOLO + classifier output.
    This is the novel contribution — structured AI reasoning
    for neuro-oncology from multi-model detection pipeline.
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))


    structured_data = {
        "scan_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "yolo_detections": detection_result["detections"],
        "tumor_found": detection_result["tumor_found"],
        "total_lesions": detection_result["total_count"],
        "primary_finding": detection_result["primary"],
        "classifier_result": {
            "predicted_type": classification["predicted_class"],
            "confidence": f"{classification['confidence']*100:.1f}%",
            "all_probabilities": {
                k: f"{v*100:.1f}%"
                for k,v in classification["probabilities"].items()
            }
        }
    }

    system_prompt = """You are an expert neuro-oncology AI assistant.
You will receive structured detection data from a multi-model AI pipeline:
1. YOLOv10 object detector — provides bounding boxes and locations
2. EfficientNet classifier — provides tumor type probabilities

Your task is to generate a structured clinical radiology report following
standard neuro-oncology reporting format (RADS-Brain).

STRICT RULES:
- Only report what the data shows. Never fabricate findings.
- Always note this is AI-generated and requires radiologist review.
- Include WHO grade estimate based on tumor type (educational only).
- Flag low confidence scores as uncertain findings.
- Use professional medical terminology.
- End with recommended next steps.
"""

    user_prompt = f"""Generate a neuro-oncology radiology report for this AI detection output:

{json.dumps(structured_data, indent=2)}

Format your report as:

NEURO-ONCOLOGY AI REPORT
=========================
Date: [date]
Pipeline: YOLOv10 + EfficientNet + LLM

DETECTION SUMMARY:
[summarize YOLO findings]

PRIMARY IMPRESSION:
[tumor type, location, size, confidence]

DIFFERENTIAL DIAGNOSIS:
[list alternatives with probabilities]

WHO GRADE ESTIMATE (Educational):
[based on tumor type literature]

UNCERTAINTY FLAGS:
[any low confidence or conflicting findings]

RECOMMENDED NEXT STEPS:
[standard clinical next steps for this finding]

DISCLAIMER:
[AI disclaimer]
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        max_tokens=1500,
        temperature=0.1,
    )

    return response.choices[0].message.content


def chat_with_neuro_report(report: str, question: str, history: list = []) -> str:
    """Chat about the neuro-oncology report."""
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    messages = [{
        "role": "system",
        "content": f"""You are a neuro-oncology AI assistant explaining a radiology report.
Only answer based on the report below. Use simple language.
Always remind the user a licensed radiologist must review findings.

REPORT:
{report}"""
    }]

    for msg in history:
        messages.append(msg)
    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        max_tokens=1024,
        temperature=0.1,
    )

    reply = response.choices[0].message.content
    history.append({"role": "user",      "content": question})
    history.append({"role": "assistant", "content": reply})
    return reply