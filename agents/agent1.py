# agents/agent1.py
from __future__ import annotations

import os
from dataclasses import dataclass
from io import BytesIO

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from matplotlib import cm
from torchvision import models, transforms
from torchvision.models import DenseNet121_Weights

# ── Chemin du checkpoint ────────────────────────────────────────────────────
# Modifiez ce chemin si votre .pth est ailleurs
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "checkpoints", "BEST_DENSENET121 (1).pth"
)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_SIZE      = 224
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Architecture (identique au Notebook 1) ─────────────────────────────────
class _DenseNet121(nn.Module):
    def __init__(self):
        super().__init__()
        base            = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        in_feat         = base.classifier.in_features  # 1024
        self.features   = base.features
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(in_feat, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ── Singleton : chargé une seule fois au démarrage ─────────────────────────
_model: _DenseNet121 | None = None
_threshold: float = 0.5


def _load_model() -> tuple[_DenseNet121, float]:
    global _model, _threshold
    if _model is not None:
        return _model, _threshold

    ckpt_path = os.path.abspath(CHECKPOINT_PATH)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint introuvable : {ckpt_path}\n"
            "→ Copiez BEST_DENSENET121 (1).pth dans streamlit/checkpoints/"
        )

    ckpt    = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    _model  = _DenseNet121().to(DEVICE)
    _model.load_state_dict(ckpt["model_state"])
    _model.eval()
    _threshold = float(ckpt.get("best_tau_balanced", 0.5))
    return _model, _threshold


# ── Prétraitement ──────────────────────────────────────────────────────────
_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def _preprocess(image: Image.Image) -> torch.Tensor:
    return _transform(image).unsqueeze(0).to(DEVICE)


# ── Grad-CAM ───────────────────────────────────────────────────────────────
class _GradCAM:
    def __init__(self, model: _DenseNet121):
        self._model       = model
        self._activations: torch.Tensor | None = None
        self._gradients:   torch.Tensor | None = None
        target = dict(model.features.named_children())["denseblock4"]
        self._h1 = target.register_forward_hook(
            lambda m, i, o: setattr(self, "_activations", o.detach())
        )
        self._h2 = target.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "_gradients", go[0].detach())
        )

    def generate(self, img_tensor: torch.Tensor) -> tuple[np.ndarray, float]:
        img_tensor = img_tensor.requires_grad_(True)
        logit = self._model(img_tensor)
        prob  = torch.sigmoid(logit).item()
        self._model.zero_grad()
        logit.backward()
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam     = F.relu((weights * self._activations).sum(1, keepdim=True))
        cam     = cam.squeeze().cpu().numpy()
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        heatmap = np.array(
            Image.fromarray(cam).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        )
        return heatmap, prob

    def remove(self):
        self._h1.remove()
        self._h2.remove()


# ── Sortie ─────────────────────────────────────────────────────────────────
@dataclass
class Agent1Output:
    probability: float
    report:      str
    heatmap:     Image.Image
    overlay:     Image.Image
    active_zone: str


# ── Classe publique (interface identique à l'ancienne) ─────────────────────
class DenseNet121Agent1:
    def __init__(self, model_version: str = "DenseNet121 v1.0") -> None:
        self.model_version    = model_version
        self._net, self._tau  = _load_model()   # chargement du .pth ici

    # ── helpers privés ─────────────────────────────────────────────────────
    @staticmethod
    def _to_pil(uploaded_file) -> Image.Image:
        return Image.open(BytesIO(uploaded_file.getvalue())).convert("RGB")

    @staticmethod
    def _jet_overlay(image: Image.Image, heatmap: np.ndarray) -> tuple[Image.Image, Image.Image]:
        heat_rgb = Image.fromarray((cm.jet(heatmap)[..., :3] * 255).astype(np.uint8))
        heat_rs  = heat_rgb.resize(image.size, Image.BILINEAR)
        img_arr  = np.array(image, dtype=np.float32)
        ht_arr   = np.array(heat_rs, dtype=np.float32)
        blended  = np.clip(0.58 * img_arr + 0.42 * ht_arr, 0, 255).astype(np.uint8)
        return heat_rgb, Image.fromarray(blended)

    @staticmethod
    def _active_zone(heatmap: np.ndarray) -> str:
        h, w   = heatmap.shape
        side   = "droit"   if heatmap[:, w//2:].mean() >= heatmap[:, :w//2].mean() else "gauche"
        level  = "inférieur" if heatmap[h//2:].mean() >= heatmap[:h//2].mean()     else "supérieur"
        return f"lobe {level} {side}"

    def generate_report(self, probability: float, active_zone: str, file_name: str) -> str:
        severity = "élevée" if probability >= 0.8 else "modérée" if probability >= 0.55 else "faible"
        return (
            f"Radiographie analysée: {file_name}\n"
            "Qualité technique: exploitable.\n"
            f"Zone d'activation dominante: {active_zone}.\n"
            f"Hypothèse pneumonique: probabilité {probability:.1%} (sévérité {severity}).\n"
            "A corréler avec l'examen clinique, la SpO2 et le bilan biologique."
        )

    # ── méthode publique ───────────────────────────────────────────────────
    def analyze(self, uploaded_file, symptoms: str) -> Agent1Output:
        image      = self._to_pil(uploaded_file)
        img_tensor = _preprocess(image)

        gc          = _GradCAM(self._net)
        heatmap, prob = gc.generate(img_tensor)
        gc.remove()

        heatmap_img, overlay = self._jet_overlay(image, heatmap)
        zone                 = self._active_zone(heatmap)
        report               = self.generate_report(prob, zone, uploaded_file.name)

        return Agent1Output(
            probability = prob,
            report      = report,
            heatmap     = heatmap_img,
            overlay     = overlay,
            active_zone = zone,
        )