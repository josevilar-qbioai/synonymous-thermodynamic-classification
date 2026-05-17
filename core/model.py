"""
EnergySignalCNN — Red neuronal convolucional 1D para clasificación
de variantes genéticas a partir de perfiles energéticos multicanal.

Arquitectura basada en la patente EnergoRNA P202630522 (reiv. 1-5):
  - Señal multicanal 1D (energía, GC, purinas, estructura)
  - Conv1D con batch normalization
  - Mecanismo de self-attention (reiv. 2)
  - Grad-CAM 1D para interpretabilidad (reiv. 3)

Uso:
    from core.model import EnergySignalCNN, GradCAM1D

    model = EnergySignalCNN(n_channels=12, n_classes=2)
    logits = model(x)  # x: (batch, seq_len, n_channels)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================
# Bloques constituyentes
# ============================================================

class ConvBlock(nn.Module):
    """Conv1D + BatchNorm + ReLU + Dropout."""

    def __init__(self, in_channels, out_channels, kernel_size, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=kernel_size // 2,  # same padding
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, channels, seq_len)
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Self-attention sobre la dimensión temporal (seq_len).
    Pondera la relevancia relativa de posiciones dentro de la señal.
    Patente reiv. 2.
    """

    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) debe ser divisible por n_heads ({n_heads})"
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        batch_size, seq_len, d_model = x.shape
        residual = x

        # Proyecciones Q, K, V
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k)

        # Transponer para atención: (batch, heads, seq_len, d_k)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Aplicar atención
        context = torch.matmul(attn_weights, V)

        # Concatenar cabezas
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_len, d_model)

        # Proyección de salida + residual + layer norm
        out = self.W_o(context)
        out = self.dropout(out)
        out = self.layer_norm(out + residual)

        return out, attn_weights


# ============================================================
# Modelo principal
# ============================================================

class EnergySignalCNN(nn.Module):
    """
    CNN 1D multicanal para clasificación de variantes genéticas
    a partir de perfiles energéticos.

    Arquitectura (patente reiv. 1-5):
        Input: (batch, seq_len, n_channels) — e.g. (B, 128, 12)
        → Conv1D 64 filtros, k=7
        → Conv1D 128 filtros, k=5
        → Self-Attention multi-head (4 cabezas)
        → Conv1D 256 filtros, k=3
        → Global Average Pooling
        → FC 256 → 64 → n_classes
        → Softmax (implícito en CrossEntropyLoss)

    Args:
        n_channels: Número de canales de entrada (default: 12)
        n_classes: Número de clases de salida (default: 2)
        dropout: Tasa de dropout (default: 0.2)
        n_heads: Cabezas de atención (default: 4)
    """

    def __init__(self, n_channels=12, n_classes=2, dropout=0.2, n_heads=4):
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes

        # Bloque convolucional 1: extraer patrones locales (k=7 ≈ 2 codones)
        self.conv1 = ConvBlock(n_channels, 64, kernel_size=7, dropout=dropout)

        # Bloque convolucional 2: patrones intermedios (k=5)
        self.conv2 = ConvBlock(64, 128, kernel_size=5, dropout=dropout)

        # Self-attention: ponderar relevancia de posiciones (reiv. 2)
        self.attention = MultiHeadSelfAttention(
            d_model=128, n_heads=n_heads, dropout=dropout
        )

        # Bloque convolucional 3: patrones de alto nivel (k=3)
        self.conv3 = ConvBlock(128, 256, kernel_size=3, dropout=dropout)

        # Clasificador
        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

        # Almacenar activaciones y gradientes para Grad-CAM
        self._activations = None
        self._gradients = None

    def _register_hooks(self):
        """Registra hooks para Grad-CAM en la última capa conv."""
        def forward_hook(module, input, output):
            self._activations = output

        def backward_hook(module, grad_input, grad_output):
            self._gradients = grad_output[0]

        self.conv3.conv.register_forward_hook(forward_hook)
        self.conv3.conv.register_full_backward_hook(backward_hook)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (batch, seq_len, n_channels) — nota: channels LAST

        Returns:
            logits: (batch, n_classes)
        """
        # Transponer a channels-first para Conv1d: (B, C, L)
        x = x.transpose(1, 2)

        # Bloques convolucionales
        x = self.conv1(x)   # (B, 64, L)
        x = self.conv2(x)   # (B, 128, L)

        # Self-attention (necesita channels-last)
        x = x.transpose(1, 2)              # (B, L, 128)
        x, attn_weights = self.attention(x) # (B, L, 128)
        x = x.transpose(1, 2)              # (B, 128, L)

        # Última convolución
        x = self.conv3(x)   # (B, 256, L)

        # Global Average Pooling
        x = x.mean(dim=2)   # (B, 256)

        # Clasificación
        logits = self.classifier(x)  # (B, n_classes)

        return logits

    def predict_proba(self, x):
        """Devuelve probabilidades calibradas."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
        return probs

    def count_parameters(self):
        """Cuenta parámetros entrenables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# Grad-CAM 1D — Interpretabilidad (patente reiv. 3)
# ============================================================

class GradCAM1D:
    """
    Gradient-weighted Class Activation Mapping adaptado a CNN 1D.
    Genera mapas de atención a nivel de nucleótido que identifican
    las regiones del transcrito con mayor contribución a la predicción.

    Uso:
        gradcam = GradCAM1D(model, target_layer='conv3')
        heatmap = gradcam(x, target_class=1)  # (seq_len,)
    """

    def __init__(self, model, target_layer='conv3'):
        self.model = model
        self.activations = None
        self.gradients = None

        # Registrar hooks en la capa objetivo
        layer = getattr(model, target_layer)
        if hasattr(layer, 'conv'):
            layer = layer.conv

        layer.register_forward_hook(self._save_activation)
        layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x, target_class=None):
        """
        Genera mapa de activación Grad-CAM 1D.

        Args:
            x: (1, seq_len, n_channels) — un solo ejemplo
            target_class: Clase objetivo (default: clase predicha)

        Returns:
            heatmap: np.array (seq_len,) — importancia por posición
            prediction: int — clase predicha
            confidence: float — probabilidad de la clase predicha
        """
        self.model.eval()
        x.requires_grad_(True)

        # Forward
        logits = self.model(x)
        probs = F.softmax(logits, dim=1)

        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        confidence = probs[0, target_class].item()

        # Backward respecto a la clase objetivo
        self.model.zero_grad()
        logits[0, target_class].backward()

        # Grad-CAM: pesos = media global de gradientes por canal
        # activations: (1, C, L), gradients: (1, C, L)
        weights = self.gradients.mean(dim=2, keepdim=True)  # (1, C, 1)
        cam = (weights * self.activations).sum(dim=1)        # (1, L)
        cam = F.relu(cam)                                    # Solo contribuciones positivas

        # Normalizar a [0, 1]
        cam = cam.squeeze(0).cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Interpolar al tamaño original de la secuencia si difiere
        if len(cam) != x.shape[1]:
            from scipy.interpolate import interp1d
            f = interp1d(
                np.linspace(0, 1, len(cam)),
                cam,
                kind='linear',
            )
            cam = f(np.linspace(0, 1, x.shape[1]))

        return cam, target_class, confidence


# ============================================================
# Channel Attribution — qué canales contribuyen más
# ============================================================

class ChannelAttribution:
    """
    Mide la contribución relativa de cada canal de entrada
    a la predicción, usando gradient × input (Integrated Gradients
    simplificado).

    Uso:
        attr = ChannelAttribution(model)
        scores = attr(x)  # dict {channel_name: importance}
    """

    CHANNEL_NAMES = [
        'ΔG_SantaLucia_wt', 'ΔG_Turner_wt',
        'ΔG_SantaLucia_mut', 'ΔG_Turner_mut',
        'ΔΔG_SantaLucia', 'ΔΔG_Turner',
        'GC_content', 'Purine_bias', 'Codon_position',
        'SecStruct_prob', 'ESM1v_LLR', 'PhyloP',
    ]

    def __init__(self, model):
        self.model = model

    def __call__(self, x, target_class=None, channel_names=None):
        """
        Calcula importancia de cada canal.

        Args:
            x: (1, seq_len, n_channels)
            target_class: Clase objetivo (default: predicha)
            channel_names: Lista de nombres de canales

        Returns:
            dict {nombre_canal: importancia_relativa}
        """
        if channel_names is None:
            n_ch = x.shape[2]
            if n_ch <= len(self.CHANNEL_NAMES):
                channel_names = self.CHANNEL_NAMES[:n_ch]
            else:
                channel_names = [f'channel_{i}' for i in range(n_ch)]

        self.model.eval()
        x_input = x.detach().clone().requires_grad_(True)
        x_input.retain_grad()

        logits = self.model(x_input)
        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, target_class].backward(retain_graph=True)

        # Gradient × Input por canal
        grad = x_input.grad.detach()            # (1, L, C)
        attribution = (grad * x_input).abs()     # (1, L, C)
        channel_importance = attribution.mean(dim=1).squeeze(0)  # (C,)

        # Normalizar
        total = channel_importance.sum()
        if total > 0:
            channel_importance = channel_importance / total

        channel_importance = channel_importance.detach().cpu()
        return {
            name: float(channel_importance[i].item())
            for i, name in enumerate(channel_names)
        }


# ============================================================
# Utilidades de entrenamiento
# ============================================================

class EarlyStopping:
    """Early stopping para evitar overfitting."""

    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.should_stop


class VariantDataset(torch.utils.data.Dataset):
    """
    Dataset de variantes para entrenamiento.

    Args:
        tensors: np.array (N, seq_len, n_channels) — tensores multicanal
        labels: np.array (N,) — 0=benign, 1=pathogenic
        augment: bool — aplicar data augmentation
    """

    def __init__(self, tensors, labels, augment=False):
        self.tensors = torch.FloatTensor(tensors)
        self.labels = torch.LongTensor(labels)
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.tensors[idx].clone()
        y = self.labels[idx]

        if self.augment:
            x = self._augment(x)

        return x, y

    def _augment(self, x):
        """
        Data augmentation para señales energéticas:
        - Jitter gaussiano (ruido ±0.05 kcal/mol)
        - Shift temporal (±3 posiciones)
        - Scaling aleatorio (±5%)
        """
        # Jitter
        if torch.rand(1) < 0.5:
            noise = torch.randn_like(x) * 0.05
            x = x + noise

        # Shift temporal
        if torch.rand(1) < 0.3:
            shift = torch.randint(-3, 4, (1,)).item()
            if shift != 0:
                x = torch.roll(x, shifts=shift, dims=0)

        # Scaling
        if torch.rand(1) < 0.3:
            scale = 1.0 + (torch.rand(1).item() - 0.5) * 0.1
            x = x * scale

        return x


def create_weighted_sampler(labels):
    """
    Crea un WeightedRandomSampler para manejar desbalance de clases.
    Útil porque típicamente hay más Pathogenic que Benign en ClinVar.
    """
    class_counts = np.bincount(labels)
    weights = 1.0 / class_counts[labels]
    return torch.utils.data.WeightedRandomSampler(
        weights=torch.FloatTensor(weights),
        num_samples=len(labels),
        replacement=True,
    )
