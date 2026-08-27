"""
Image generation for the demo.

  design  -> Stability SD3.5 Large (stability.sd3-5-large-v1:0, text-to-image) on Bedrock
  preview -> the generated design composited onto the product photo in backend/img/
             (local PIL paste into the product's print area, with light shading)

The design step calls Bedrock and raises on failure so the error surfaces in the
agent trace. The preview step is local (no cloud) — it shows the *actual*
generated design printed on the product.

Config (env):
  IMAGE_MODEL     default stability.sd3-5-large-v1:0
  AWS_BEDROCK_REGION / AWS_REGION         default us-west-2
"""

from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path

IMAGE_MODEL = os.getenv("IMAGE_MODEL", "stability.sd3-5-large-v1:0")
_REGION = os.getenv("AWS_BEDROCK_REGION") or os.getenv("AWS_REGION", "us-west-2")

_client = None


def _get_client():
    global _client
    if _client is None:
        import boto3
        _client = boto3.client("bedrock-runtime", region_name=_REGION)
    return _client


def _invoke(model_id: str, body: dict) -> str:
    resp = _get_client().invoke_model(
        modelId=model_id,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json",
    )
    out = json.loads(resp["body"].read())
    images = out.get("images") or []
    if not images:
        raise RuntimeError(f"{model_id} returned no image: {out}")
    return images[0]


def generate_design_png(prompt: str) -> str:
    """Text-to-image with SD3.5 Large. Returns base64 PNG."""
    return _invoke(IMAGE_MODEL, {
        "prompt": prompt,
        "mode": "text-to-image",
        "aspect_ratio": "1:1",
        "output_format": "png",
    })


def composite_preview_png(ref_path: Path, design_b64: str,
                          print_box=(0.34, 0.38, 0.70, 0.74)) -> str:
    """Paste the generated design onto the product photo. Returns base64 PNG.

    Shows the *actual* generated design printed on the product. `print_box` is the
    printable area as (left, top, right, bottom) fractions of the 1024px canvas.
    Aspect ratio is preserved and the product's local shading is multiplied over
    the design so it looks printed on the surface rather than pasted flat.
    """
    from PIL import Image, ImageChops

    size = 1024
    prod = Image.open(ref_path).convert("RGB").resize((size, size))
    design = Image.open(io.BytesIO(base64.b64decode(design_b64))).convert("RGB")

    left, top, right, bottom = [round(f * size) for f in print_box]
    box_w, box_h = right - left, bottom - top

    # Fit the design inside the print box, preserving its aspect ratio.
    dw, dh = design.size
    scale = min(box_w / dw, box_h / dh)
    nw, nh = max(1, round(dw * scale)), max(1, round(dh * scale))
    design = design.resize((nw, nh))
    ox, oy = left + (box_w - nw) // 2, top + (box_h - nh) // 2

    # Multiply the product's local shading over the design for a printed look.
    shade = prod.crop((ox, oy, ox + nw, oy + nh)).convert("L")
    shaded = ImageChops.multiply(design, Image.merge("RGB", (shade, shade, shade)))
    printed = Image.blend(design, shaded, 0.6)

    prod.paste(printed, (ox, oy))
    return _png_b64(prod)


def _png_b64(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()
