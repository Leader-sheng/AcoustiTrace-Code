from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any


class PersistentGroundedSAM:
    """Grounded-SAM runtime that loads both models once per evaluator process.

    The upstream ``grounded_sam_demo.py`` is a one-image command-line demo.  A
    subprocess invocation for every keyframe reloads GroundingDINO and SAM on
    every call, which dominates runtime for short videos.  This class preserves
    the demo's ``mask.json`` contract while keeping the two models resident.
    Heavy third-party imports stay lazy so contract checks do not require the
    GPU environment.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        repo = Path(cfg["runtime"]["grounded_sam_repo"]).resolve()
        for path in (repo, repo / "GroundingDINO", repo / "segment_anything"):
            value = str(path)
            if value not in sys.path:
                sys.path.insert(0, value)

        import cv2
        import numpy as np
        import torch
        from PIL import Image
        import GroundingDINO.groundingdino.datasets.transforms as T
        from GroundingDINO.groundingdino.models import build_model
        from GroundingDINO.groundingdino.util.slconfig import SLConfig
        from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
        from segment_anything import SamPredictor, sam_model_registry

        self.cv2 = cv2
        self.np = np
        self.torch = torch
        self.Image = Image
        self.T = T
        self.get_phrases_from_posmap = get_phrases_from_posmap
        self.device = str(cfg["gsam"].get("device", "cuda"))
        self.keep_masks = bool(cfg.get("cleanup", {}).get("keep_masks", True))
        self.keep_visualizations = bool(cfg.get("cleanup", {}).get("keep_visualizations", False))

        model_args = SLConfig.fromfile(str(cfg["gsam"]["config"]))
        model_args.device = self.device
        model_args.bert_base_uncased_path = str(cfg["runtime"]["bert_base_uncased_path"])
        model = build_model(model_args)
        checkpoint = torch.load(str(cfg["gsam"]["grounded_checkpoint"]), map_location="cpu")
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        self.grounding_model = model.to(self.device).eval()

        sam = sam_model_registry[str(cfg["gsam"]["sam_version"])](
            checkpoint=str(cfg["gsam"]["sam_checkpoint"])
        ).to(self.device)
        self.predictor = SamPredictor(sam)

        self.box_threshold = float(cfg["gsam"].get("box_threshold", 0.3))
        self.text_threshold = float(cfg["gsam"].get("text_threshold", 0.25))

    def _load_image(self, image_path: Path):
        image_pil = self.Image.open(image_path).convert("RGB")
        transform = self.T.Compose(
            [
                self.T.RandomResize([800], max_size=1333),
                self.T.ToTensor(),
                self.T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_tensor, _ = transform(image_pil, None)
        return image_pil, image_tensor

    def _ground(self, image_tensor, caption: str):
        caption = caption.lower().strip()
        if not caption.endswith("."):
            caption += "."
        image_tensor = image_tensor.to(self.device)
        with self.torch.inference_mode():
            outputs = self.grounding_model(image_tensor[None], captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]
        boxes = outputs["pred_boxes"].cpu()[0]
        keep = logits.max(dim=1)[0] > self.box_threshold
        logits = logits[keep]
        boxes = boxes[keep]
        tokenizer = self.grounding_model.tokenizer
        tokenized = tokenizer(caption)
        labels: list[str] = []
        scores: list[float] = []
        for logit in logits:
            labels.append(
                self.get_phrases_from_posmap(
                    logit > self.text_threshold,
                    tokenized,
                    tokenizer,
                )
            )
            scores.append(float(logit.max().item()))
        return boxes, labels, scores

    def _write_outputs(
        self,
        out_dir: Path,
        rgb_image,
        boxes_xyxy,
        labels: list[str],
        scores: list[float],
        masks,
    ) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        json_rows: list[dict[str, Any]] = [{"value": 0, "label": "background"}]
        for index, (label, score, box) in enumerate(zip(labels, scores, boxes_xyxy), start=1):
            json_rows.append(
                {
                    "value": index,
                    "label": label,
                    "logit": score,
                    "box": [float(value) for value in box.tolist()],
                }
            )
        (out_dir / "mask.json").write_text(
            json.dumps(json_rows, ensure_ascii=False),
            encoding="utf-8",
        )

        if self.keep_masks:
            height, width = rgb_image.shape[:2]
            labelmap = self.np.zeros((height, width), dtype=self.np.uint16)
            for index, mask in enumerate(masks, start=1):
                mask_array = mask.detach().cpu().numpy()[0].astype(bool)
                labelmap[mask_array] = index
            self.np.save(out_dir / "labelmap.npy", labelmap)

        if self.keep_visualizations:
            # These files are diagnostic only; evaluator scoring consumes
            # mask.json/labelmap.npy.  Avoid matplotlib's high-DPI rendering in
            # ordinary runs while retaining compact image artifacts on demand.
            self.cv2.imwrite(str(out_dir / "raw_image.jpg"), self.cv2.cvtColor(rgb_image, self.cv2.COLOR_RGB2BGR))
            if self.keep_masks:
                scale = max(1, int(labelmap.max()))
                mask_image = (labelmap.astype(self.np.float32) * (255.0 / scale)).astype(self.np.uint8)
                self.cv2.imwrite(str(out_dir / "mask.jpg"), mask_image)

    def infer(self, frame_path: Path, prompt: str, out_dir: Path) -> None:
        frame_path = frame_path.resolve()
        out_dir = out_dir.resolve()
        image_pil, image_tensor = self._load_image(frame_path)
        boxes, labels, scores = self._ground(image_tensor, prompt)

        bgr = self.cv2.imread(str(frame_path))
        if bgr is None:
            raise RuntimeError(f"unable_to_read_frame:{frame_path}")
        rgb = self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]

        boxes_xyxy = boxes.clone()
        if boxes_xyxy.numel():
            scale = self.torch.tensor([width, height, width, height], dtype=boxes_xyxy.dtype)
            boxes_xyxy *= scale
            boxes_xyxy[:, :2] -= boxes_xyxy[:, 2:] / 2
            boxes_xyxy[:, 2:] += boxes_xyxy[:, :2]
        boxes_xyxy = boxes_xyxy.cpu()

        self.predictor.set_image(rgb)
        if boxes_xyxy.numel():
            transformed = self.predictor.transform.apply_boxes_torch(
                boxes_xyxy,
                rgb.shape[:2],
            ).to(self.device)
            with self.torch.inference_mode():
                masks, _, _ = self.predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed,
                    multimask_output=False,
                )
        else:
            masks = self.torch.zeros(
                (0, 1, height, width),
                dtype=self.torch.bool,
                device=self.device,
            )
        self._write_outputs(out_dir, rgb, boxes_xyxy, labels, scores, masks)

