"""
APPROACH 2: Gemini → Generate DINO Prompts → Grounding DINO → Detect
=====================================================================
Pipeline:
  1. Gemini sees the full water sample image
  2. Gemini outputs SHORT, CONCRETE noun phrases describing what it sees
     (color + shape + texture descriptors, optimized for DINO's BERT encoder)
  3. Those phrases become the text prompt for Grounding DINO
  4. DINO is the primary detector — it finds bounding boxes using those phrases
  5. All DINO detections above threshold are accepted

Key insight: DINO's text encoder (BERT) works best with short noun phrases
separated by periods, NOT with paragraph descriptions. So Gemini's role here
is pure "prompt engineering for DINO", not detection itself.

Requirements:
    pip install google-generativeai pillow transformers torch torchvision
"""

import os
import json
import base64
import requests
from io import BytesIO

import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# ─── CONFIG ────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "INSERT_YOUR_API_KEY_HERE")
GEMINI_MODEL   = "gemini-2.5-flash"
DINO_MODEL_ID  = "IDEA-Research/grounding-dino-base"

# DINO detection threshold (0–1). Lower = more detections but more false positives.
# Start at 0.25 for microplastics — they're visually subtle.
DINO_BOX_THRESHOLD  = 0.25
DINO_TEXT_THRESHOLD = 0.20

# These baseline prompts are ALWAYS included, regardless of what Gemini says.
# They anchor DINO on known microplastic morphologies.
BASELINE_DINO_PROMPTS = [
    "microplastic",
    "plastic",
    "plastic fragment",
    "plastic fiber",
    "plastic pellet",
    "plastic film",
    "plastic foam",
    "transparent plastic",
    "discolored plastic"
]

# ─── STEP 1: LOAD IMAGE ────────────────────────────────────────────────────────
def load_image_as_base64(image_path: str) -> tuple[str, Image.Image]:
    img = Image.open(image_path).convert("RGB")
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return b64, img


# ─── STEP 2: GEMINI — GENERATE DINO PROMPTS ───────────────────────────────────
GEMINI_SYSTEM_PROMPT = """
You are a prompt engineer for Grounding DINO, an open-vocabulary object detector.
Your job is to analyze a water sample image and generate TEXT PROMPTS that will
help Grounding DINO locate microplastic particles. The water sample is contained in a clear dish, and the image may contain
various particles, debris, and artifacts. Tell that to DINO in a way that helps it focus on microplastic-like features.

CRITICAL RULES for generating DINO prompts:
- Use ONLY short noun phrases (2–5 words maximum per phrase)
- Describe COLOR + SHAPE/FORM + MATERIAL, e.g.: "white angular fragment", "blue thin fiber"
- Do NOT write sentences. No verbs, no full sentences.
- Separate each phrase with a period and space in the output string
- Maximum 8 phrases total — DINO degrades with too many prompts
- Be specific to what you actually observe in the image
- Do NOT invent things you cannot see

Also classify the image conditions, as this affects DINO's expected performance.

Return ONLY valid JSON, no markdown, no explanation:
{
  "dino_prompts": [
    "white angular fragment",
    "blue thin fiber",
    "transparent irregular pellet"
  ],
  "observation_notes": "brief note on what you see and why these prompts were chosen",
  "image_conditions": {
    "background_clarity": "clear | turbid | heavily_turbid",
    "particle_visibility": "high | medium | low",
    "lighting": "even | uneven | backlit | dark"
  },
  "expected_difficulty": "easy | moderate | hard",
  "warning": "any red flag about the image quality or sample that would limit detection accuracy"
}
"""

def gemini_generate_dino_prompts(image_b64: str) -> dict:
    """
    Call Gemini with the image. Get back short noun phrases for DINO prompting.
    """
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )
    payload = {
        "system_instruction": {
            "parts": [{"text": GEMINI_SYSTEM_PROMPT}]
        },
        "contents": [
            {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_b64
                        }
                    },
                    {
                        "text": (
                            "Analyze this water sample image for microplastic detection. "
                            "Generate the Grounding DINO text prompts that best describe "
                            "any suspicious particles you see. Return JSON only."
                        )
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 8192,  # raised — thinking model needs headroom beyond the actual JSON
        }
    }

    response = requests.post(url, json=payload)
    response.raise_for_status()
    data = response.json()

    candidate = data["candidates"][0]
    finish_reason = candidate.get("finishReason", "UNKNOWN")
    if finish_reason == "MAX_TOKENS":
        tokens_used = data.get("usageMetadata", {}).get("candidatesTokenCount", "?")
        thinking    = data.get("usageMetadata", {}).get("thoughtsTokenCount", 0)
        raise RuntimeError(
            f"Gemini hit MAX_TOKENS — output was cut off.\n"
            f"  candidatesTokenCount : {tokens_used}\n"
            f"  thoughtsTokenCount   : {thinking}\n"
            f"  Fix: raise thinkingBudget or maxOutputTokens further."
        )

    raw_text = candidate["content"]["parts"][0]["text"].strip()

    # Strip any markdown fences Gemini adds despite instructions
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
    raw_text = raw_text.strip()

    return json.loads(raw_text)


# ─── STEP 3: BUILD FINAL DINO PROMPT STRING ───────────────────────────────────
def build_dino_prompt_string(gemini_prompts: list[str]) -> str:
    """
    Combine Gemini's context-specific prompts with baseline microplastic prompts.
    Deduplicate, then join with ". " as DINO expects.

    DINO's BERT encoder has a 256-token limit. Keep this string concise.
    """
    all_prompts = list(dict.fromkeys(gemini_prompts + BASELINE_DINO_PROMPTS))

    # Safety cap — too many prompts dilute DINO's attention
    all_prompts = all_prompts[:10]

    prompt_string = ". ".join(all_prompts) + "."
    return prompt_string


# ─── STEP 4: GROUNDING DINO — DETECT WITH GEMINI PROMPTS ──────────────────────
def load_dino_model(device: str = "cpu"):
    print(f"Loading Grounding DINO ({DINO_MODEL_ID}) on {device}...")
    processor = AutoProcessor.from_pretrained(DINO_MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL_ID).to(device)
    return processor, model


def dino_detect(
    pil_image: Image.Image,
    dino_prompt: str,
    processor,
    model,
    device: str = "cpu"
) -> list[dict]:
    """
    Run Grounding DINO on the FULL image using Gemini-generated prompts.
    Returns a list of detections with boxes, labels, scores.
    """
    inputs = processor(
        images=pil_image,
        text=dino_prompt,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=DINO_BOX_THRESHOLD,
        text_threshold=DINO_TEXT_THRESHOLD,
        target_sizes=[pil_image.size[::-1]]  # (H, W)
    )[0]

    detections = []
    for score, label, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        detections.append({
            "label": label,
            "score": round(score.item(), 4),
            "bbox_pixels": (x1, y1, x2, y2),
        })

    # Sort by confidence descending
    detections.sort(key=lambda d: d["score"], reverse=True)
    return detections


# ─── STEP 5: VISUALIZE ─────────────────────────────────────────────────────────
def draw_detections(
    pil_image: Image.Image,
    detections: list[dict],
    dino_prompt: str,
    output_path: str,
    max_area_ratio: float = 0.5  # detections covering >50% of image are ignored
):
    """Draw DINO detections on the original image with confidence scores."""
    vis = pil_image.copy()
    draw = ImageDraw.Draw(vis)

    img_area = pil_image.width * pil_image.height

    for det in detections:
        x1, y1, x2, y2 = det["bbox_pixels"]
        bbox_area = (x2 - x1) * (y2 - y1)

        # Skip detections that are too large
        if bbox_area / img_area > max_area_ratio:
            continue

        score = det["score"]
        color = "lime" if score >= 0.4 else "yellow" if score >= 0.3 else "orange"

        draw.rectangle(det["bbox_pixels"], outline=color, width=3)
        label_text = f"{det['label']} {score:.2f}"
        draw.text(
            (det["bbox_pixels"][0], det["bbox_pixels"][1] - 14),
            label_text,
            fill=color
        )

    vis.save(output_path)
    print(f"Visualization saved → {output_path}")


# ─── MAIN ──────────────────────────────────────────────────────────────────────
def run_approach2(image_path: str, output_path: str = "approach2_result.jpg"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n=== APPROACH 2: Gemini Prompt Engineer → DINO Detection ===\n")

    # 1. Load image
    print(f"Loading image: {image_path}")
    image_b64, pil_image = load_image_as_base64(image_path)

    # 2. Gemini: generate DINO prompts from image
    print("Calling Gemini to generate context-specific DINO prompts...")
    gemini_result = gemini_generate_dino_prompts(image_b64)

    gemini_prompts = gemini_result.get("dino_prompts", [])
    print(f"  Gemini observation: {gemini_result.get('observation_notes')}")
    print(f"  Image conditions : {gemini_result.get('image_conditions')}")
    print(f"  Expected difficulty: {gemini_result.get('expected_difficulty')}")
    if gemini_result.get("warning"):
        print(f"  ⚠ WARNING: {gemini_result['warning']}")
    print(f"  Gemini prompts   : {gemini_prompts}")

    # 3. Build final DINO prompt string
    dino_prompt = build_dino_prompt_string(gemini_prompts)
    print(f"\n  Final DINO prompt string:\n    \"{dino_prompt}\"")

    # 4. Load DINO + detect
    processor, model = load_dino_model(device)
    print(f"\nRunning Grounding DINO on full image...")
    detections = dino_detect(pil_image, dino_prompt, processor, model, device)

    # 5. Summary
    print(f"\n── RESULTS ──────────────────────────────────")
    print(f"  DINO detections: {len(detections)}")
    for i, det in enumerate(detections):
        bar = "█" * int(det["score"] * 20)
        print(f"  [{i+1}] {det['label']:<30} score={det['score']:.4f}  {bar}")

    if not detections:
        print("  No microplastics detected above threshold.")
        print("  Consider lowering DINO_BOX_THRESHOLD or improving image quality.")

    # 6. Visualize
    draw_detections(pil_image, detections, dino_prompt, output_path)

    return {
        "approach": 2,
        "gemini_prompts": gemini_prompts,
        "dino_prompt_used": dino_prompt,
        "total_detections": len(detections),
        "detections": detections,
        "image_conditions": gemini_result.get("image_conditions"),
    }


if __name__ == "__main__":
    import sys
    image_file = sys.argv[1] if len(sys.argv) > 1 else "sample.jpg"
    run_approach2(image_file)