"""
APPROACH 1: Gemini → Detect & Crop → Grounding DINO → Validate
================================================================
Pipeline:
  1. Gemini sees the full water sample image
  2. Gemini returns bounding boxes of suspicious particles
  3. We crop each candidate from the original image
  4. Grounding DINO looks at each crop and decides: "is this microplastic?"
  5. Only DINO-confirmed crops are accepted as detections

Requirements:
    pip install google-generativeai pillow transformers torch torchvision
"""

import os
import json
import base64
import requests
from pathlib import Path
from io import BytesIO

import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# ─── CONFIG ────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "INSERT_YOUR_API_KEY_HERE")
GEMINI_MODEL   = "gemini-2.5-flash"          # or gemini-2.0-flash
DINO_MODEL_ID  = "IDEA-Research/grounding-dino-base"

# How confident DINO must be (0–1) to count a crop as real microplastic
DINO_THRESHOLD = 0.30

# Padding added around each Gemini bounding box before cropping (pixels)
CROP_PADDING = 10

# ─── STEP 1: LOAD IMAGE ────────────────────────────────────────────────────────
def load_image_as_base64(image_path: str) -> tuple[str, Image.Image]:
    """Read a local image, return (base64_string, PIL_Image)."""
    img = Image.open(image_path).convert("RGB")
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return b64, img


# ─── STEP 2: GEMINI — DETECT SUSPICIOUS PARTICLES ─────────────────────────────
GEMINI_SYSTEM_PROMPT = """
You are a microplastic detection assistant analyzing water sample images.
Your task is to locate ALL suspicious particles that could be microplastics.

You will be deliberately inclusive — flag anything that is NOT clearly water.
This includes: fragments, fibers, pellets, films, foam pieces, discolored spots,
and any foreign material regardless of color or transparency.

Do NOT try to confirm whether something IS microplastic — that is done later.
Your only job is to find candidates.

Return ONLY valid JSON, no explanation, no markdown fences. And for the candidates find a max 14 objects in the image.
Format:
{
  "candidates": [
    {
      "id": 1,
      "description": "short phrase: color + shape, e.g. 'white angular fragment'",
      "bbox_normalized": [y_min, x_min, y_max, x_max],
      "confidence": "high | medium | low"
    }
  ],
  "image_quality": "good | blurry | low_light | uneven_background",
  "notes": "any observation about the sample preparation or image conditions"
}

Bounding box values must be normalized to 0–1000 (Gemini standard).
If the image is completely clear water with nothing suspicious, return:
{"candidates": [], "image_quality": "...", "notes": "..."}
"""

def gemini_detect_candidates(image_b64: str) -> dict:
    """
    Call Gemini API with the water sample image.
    Returns parsed JSON with candidate bounding boxes.
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
                            "Analyze this water sample image. "
                            "Identify all suspicious particles that may be microplastics. "
                            "Return JSON only."
                        )
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.3,       # low temp = more consistent, less hallucination
            "maxOutputTokens": 8192,
        }
    }

    response = requests.post(url, json=payload)
    response.raise_for_status()
    data = response.json()

    raw_text = data["candidates"][0]["content"]["parts"][0]["text"]

    # Strip markdown fences if Gemini adds them despite instructions
    raw_text = raw_text.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
    raw_text = raw_text.strip()

    print("Extracted Gemini text:", raw_text)

    return json.loads(raw_text)


# ─── STEP 3: CROP EACH CANDIDATE ───────────────────────────────────────────────
def crop_candidates(pil_image: Image.Image, candidates: list[dict]) -> list[dict]:
    """
    Convert Gemini's normalized bboxes to pixel coords and crop.
    Returns a list of dicts with the crop PIL image + metadata.
    """
    W, H = pil_image.size
    crops = []

    for c in candidates:
        yn, xn, yn2, xn2 = c["bbox_normalized"]   # [y_min, x_min, y_max, x_max] 0–1000

        # Convert to pixels
        x1 = int((xn  / 1000) * W) - CROP_PADDING
        y1 = int((yn  / 1000) * H) - CROP_PADDING
        x2 = int((xn2 / 1000) * W) + CROP_PADDING
        y2 = int((yn2 / 1000) * H) + CROP_PADDING

        # Clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        # Skip degenerate boxes
        if x2 - x1 < 5 or y2 - y1 < 5:
            continue

        crop = pil_image.crop((x1, y1, x2, y2))
        crops.append({
            "id": c["id"],
            "description": c["description"],
            "gemini_confidence": c.get("confidence", "unknown"),
            "pixel_bbox": (x1, y1, x2, y2),
            "crop_image": crop,
        })

    return crops


# ─── STEP 4: GROUNDING DINO — VALIDATE EACH CROP ──────────────────────────────
# These are the text prompts DINO uses to "search" inside each crop.
# Keep them short noun phrases — DINO's BERT encoder works best this way.
DINO_VALIDATION_PROMPTS = (
    "microplastic",
    "plastic",
    "plastic fragment",
    "plastic fiber",
    "plastic pellet",
    "plastic film",
    "plastic foam",
    "transparent plastic",
    "discolored plastic"
)

def load_dino_model(device: str = "cpu"):
    """Load Grounding DINO processor and model."""
    print(f"Loading Grounding DINO ({DINO_MODEL_ID}) on {device}...")
    processor = AutoProcessor.from_pretrained(DINO_MODEL_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(DINO_MODEL_ID).to(device)
    return processor, model


def dino_validate_crop(
    crop_image: Image.Image,
    processor,
    model,
    device: str = "cpu"
) -> dict:
    """
    Run Grounding DINO on a single crop.
    Returns the best detection score and label found, or score=0 if nothing detected.
    """
    inputs = processor(
        images=crop_image,
        text=DINO_VALIDATION_PROMPTS,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=0.3,
        text_threshold=DINO_THRESHOLD,
        target_sizes=[crop_image.size[::-1]]  # (H, W)
    )[0]

    if len(results["scores"]) == 0:
        return {"confirmed": False, "best_score": 0.0, "best_label": None}

    best_idx   = results["scores"].argmax().item()
    best_score = results["scores"][best_idx].item()
    best_label = results["labels"][best_idx]

    return {
        "confirmed": best_score >= DINO_THRESHOLD,
        "best_score": round(best_score, 4),
        "best_label": best_label,
    }


# ─── STEP 5: VISUALIZE RESULTS ─────────────────────────────────────────────────
def draw_results_on_image(
    pil_image: Image.Image,
    confirmed_crops: list[dict],
    rejected_crops: list[dict],
    output_path: str
):
    """Draw bounding boxes on the original image: green=confirmed, red=rejected."""
    vis = pil_image.copy()
    draw = ImageDraw.Draw(vis)

    for crop in confirmed_crops:
        draw.rectangle(crop["pixel_bbox"], outline="lime", width=3)
        label = f"✓ {crop['best_label']} ({crop['best_score']:.2f})"
        draw.text((crop["pixel_bbox"][0], crop["pixel_bbox"][1] - 14), label, fill="lime")

    for crop in rejected_crops:
        draw.rectangle(crop["pixel_bbox"], outline="red", width=2)
        label = f"✗ {crop['description']}"
        draw.text((crop["pixel_bbox"][0], crop["pixel_bbox"][1] - 14), label, fill="red")

    vis.save(output_path)
    print(f"Visualization saved → {output_path}")


# ─── MAIN ──────────────────────────────────────────────────────────────────────
def run_approach1(image_path: str, output_path: str = "approach1_result.jpg"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n=== APPROACH 1: Gemini Detect → DINO Validate ===\n")

    # 1. Load image
    print(f"Loading image: {image_path}")
    image_b64, pil_image = load_image_as_base64(image_path)

    # 2. Gemini: detect candidate particles
    print("Calling Gemini to detect candidate particles...")
    gemini_result = gemini_detect_candidates(image_b64)
    candidates = gemini_result.get("candidates", [])
    print(f"  Image quality: {gemini_result.get('image_quality')}")
    print(f"  Notes: {gemini_result.get('notes')}")
    print(f"  Gemini found {len(candidates)} candidate(s)")

    if not candidates:
        print("No candidates found by Gemini. Done.")
        return

    # 3. Crop each candidate
    crops = crop_candidates(pil_image, candidates)
    print(f"  Cropped {len(crops)} candidate(s) (skipped degenerate boxes)")

    # 4. Load DINO
    processor, model = load_dino_model(device)

    # 5. Validate each crop with DINO
    confirmed_crops = []
    rejected_crops  = []

    for crop_data in crops:
        print(f"\n  Validating crop #{crop_data['id']}: '{crop_data['description']}'")
        dino_result = dino_validate_crop(crop_data["crop_image"], processor, model, device)
        crop_data.update(dino_result)

        if dino_result["confirmed"]:
            print(f"    → CONFIRMED ✓  label='{dino_result['best_label']}'  score={dino_result['best_score']}")
            confirmed_crops.append(crop_data)
        else:
            print(f"    → REJECTED  ✗  score={dino_result['best_score']} (below {DINO_THRESHOLD})")
            rejected_crops.append(crop_data)

    # 6. Summary
    print(f"\n── RESULTS ──────────────────────────────────")
    print(f"  Gemini candidates : {len(candidates)}")
    print(f"  DINO confirmed    : {len(confirmed_crops)}")
    print(f"  DINO rejected     : {len(rejected_crops)}")
    print(f"  Validation rate   : {len(confirmed_crops)/max(len(crops),1)*100:.1f}%")

    for c in confirmed_crops:
        print(f"  [✓] {c['description']} → {c['best_label']} (score {c['best_score']})")

    # 7. Visualize
    draw_results_on_image(pil_image, confirmed_crops, rejected_crops, output_path)

    return {
        "approach": 1,
        "gemini_candidates": len(candidates),
        "dino_confirmed": len(confirmed_crops),
        "confirmed_details": confirmed_crops,
    }


if __name__ == "__main__":
    import sys
    image_file = sys.argv[1] if len(sys.argv) > 1 else "sample.jpg"
    run_approach1(image_file)