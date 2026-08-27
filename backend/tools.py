"""
Tool implementations for the Gift Design Agent demo.

Images (see imagegen.py):
  design  -> Stability SD3.5 Large (text-to-image) on Amazon Bedrock
  preview -> the generated design composited onto the product photo in backend/img/

The design step needs Bedrock credentials; missing config raises a clear error.
"""

from __future__ import annotations

from pathlib import Path

import imagegen

IMG_DIR = Path(__file__).resolve().parent / "img"


# ---------------------------------------------------------------------------
# search_products()  ->  a tiny fake catalog
# ---------------------------------------------------------------------------

CATALOG = [
    {"id": "mug", "name": "Ceramic Mug", "price": 18, "printable": True,
     "ref_image": "mug.jpg", "print_box": (0.14, 0.34, 0.56, 0.72),
     "note": "11oz, dishwasher safe, large wrap-around print area"},
    {"id": "tshirt", "name": "T-Shirt", "price": 25, "printable": True,
     "ref_image": "t-shirt.jpg", "print_box": (0.37, 0.30, 0.63, 0.60),
     "note": "100% cotton, front print"},
    {"id": "tote", "name": "Tote Bag", "price": 20, "printable": True,
     "ref_image": "tote_bag.jpg", "print_box": (0.30, 0.45, 0.70, 0.80),
     "note": "canvas, single-side print"},
]


def search_products(query: str) -> list[dict]:
    """Return the catalog (fake product search)."""
    return list(CATALOG)


# ---------------------------------------------------------------------------
# Image tools: Stability models on Bedrock.
# ---------------------------------------------------------------------------

def generate_design(prompt: str) -> str:
    """generate_image(): SD3.5 Large text-to-image. Returns a PNG data URI."""
    return _png_uri(imagegen.generate_design_png(prompt))


def create_product_preview(product: dict, design_uri: str) -> str:
    """create_product_preview(): composite the generated design onto the product photo."""
    design_b64 = design_uri.split(",", 1)[1] if design_uri.startswith("data:") else design_uri
    b64 = imagegen.composite_preview_png(
        IMG_DIR / product["ref_image"], design_b64, product["print_box"])
    return _png_uri(b64)


def _png_uri(b64: str) -> str:
    return "data:image/png;base64," + b64
