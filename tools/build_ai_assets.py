#!/usr/bin/env python3
"""Build the browser AI model and reference-sign embedding index.

Designed to run in GitHub Actions. Nothing is trained on the user's computer:
we start from torchvision's ImageNet MobileNetV3-Small weights, export a feature
extractor to ONNX, render every bundled period sign, and store its normalized
embedding in one compact float32 file.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
OUT = DOCS / "models"
PERIOD_FILES = {
    "Neo-Assyrian": "signs-na.js",
    "Old Babylonian": "signs-ob.js",
    "Middle Babylonian": "signs-mb.js",
    "Neo-Babylonian": "signs-nb.js",
    "Hittite": "signs-hit.js",
    "Modern (Noto)": "signs-data.js",
}


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        base = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.features = base.features
        self.avgpool = base.avgpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avgpool(self.features(x))
        x = torch.flatten(x, 1)
        return torch.nn.functional.normalize(x, p=2, dim=1)


def load_js_array(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^window\.[A-Z_]+=(.*);\s*$", text, re.S)
    if not match:
        raise ValueError(f"Unrecognized sign-data wrapper: {path}")
    return json.loads(match.group(1))


def render_sign(polygons: list[list[list[float]]], size: int = 224) -> Image.Image:
    image = Image.new("RGB", (size, size), "white")
    if not polygons:
        return image
    points = [(float(x), -float(y)) for poly in polygons for x, y in poly]
    min_x = min(p[0] for p in points)
    max_x = max(p[0] for p in points)
    min_y = min(p[1] for p in points)
    max_y = max(p[1] for p in points)
    margin = size * 0.10
    scale = (size - 2 * margin) / max(max_x - min_x, max_y - min_y, 1.0)
    offset_x = (size - (max_x - min_x) * scale) / 2
    offset_y = (size - (max_y - min_y) * scale) / 2
    draw = ImageDraw.Draw(image)
    for poly in polygons:
        fitted = [
            ((float(x) - min_x) * scale + offset_x,
             (-float(y) - min_y) * scale + offset_y)
            for x, y in poly
        ]
        if len(fitted) >= 3:
            draw.polygon(fitted, fill="black")
    return image


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    weights = MobileNet_V3_Small_Weights.DEFAULT
    transform = weights.transforms()
    model = Encoder().eval()

    dummy = torch.zeros(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy,
        OUT / "cuneiform-encoder.onnx",
        input_names=["image"],
        output_names=["embedding"],
        dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )

    metadata: list[dict] = []
    tensors: list[torch.Tensor] = []
    for period, filename in PERIOD_FILES.items():
        signs = load_js_array(DOCS / "js" / filename)
        for sign in signs:
            tensors.append(transform(render_sign(sign.get("g", []))))
            metadata.append({"period": period, "cp": sign["cp"]})

    batches: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(tensors), 64):
            batch = torch.stack(tensors[start : start + 64])
            batches.append(model(batch).cpu().numpy().astype("<f4"))
    embeddings = np.concatenate(batches, axis=0)
    embeddings.tofile(OUT / "sign-embeddings.f32")
    (OUT / "sign-index.json").write_text(
        json.dumps(
            {
                "version": 1,
                "architecture": "MobileNetV3-Small ImageNet bootstrap",
                "dimensions": int(embeddings.shape[1]),
                "count": len(metadata),
                "entries": metadata,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    print(f"Wrote {len(metadata)} embeddings × {embeddings.shape[1]} dimensions to {OUT}")


if __name__ == "__main__":
    main()
