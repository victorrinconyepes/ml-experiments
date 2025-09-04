import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import torch.nn.functional as F
from satlaspretrain_models.model import Weights

class DualUNetPlusPlusGuided(nn.Module):
    """
    U-Net++ (SMP) con:
      - Cabeza de clasificación auxiliar desde f5 (GAP+FC).
      - Guía espacial: f5 -> conv1x1 -> upsample -> fusiona con decoder -> convs -> sigmoid -> gating residual.
      - Guía de canal (FiLM): GAP(f5) -> MLP -> gamma,beta -> y = (1+gamma)*x + beta.
    Devuelve (seg_map_logits, cls_logits).
    """
    def __init__(self, backbone_name='efficientnet-b0', pretrained=True, in_channels=3, seg_classes=1,
                 cls_classes=1, use_spatial_attn=True, use_film=True, film_hidden=512):
        super().__init__()
        self.seg_model = smp.UnetPlusPlus(
            encoder_name=backbone_name,
            encoder_weights='imagenet' if pretrained else None,
            in_channels=in_channels,
            classes=seg_classes,
            activation=None
        )
        self.encoder = self.seg_model.encoder
        self.decoder = self.seg_model.decoder
        self.segmentation_head = self.seg_model.segmentation_head

        c5 = self.encoder.out_channels[-1]
        # cabeza cls
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.cls_fc = nn.Linear(c5, cls_classes)

        # descubrir canales del decoder
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, 256, 256)
            feats = self.encoder(dummy)
            dfeat = self.decoder(feats)
            cdec = dfeat.shape[1]
        self.cdec = cdec

        self.use_spatial_attn = use_spatial_attn
        self.use_film = use_film

        if use_spatial_attn:
            # proyección de contexto desde f5 a cdec y fusión con decoder
            self.ctx_proj = nn.Conv2d(c5, cdec, kernel_size=1)
            self.attn_gen = nn.Sequential(
                nn.Conv2d(cdec*2, cdec, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(cdec, cdec, kernel_size=3, padding=1),
                nn.Sigmoid()
            )

        if use_film:
            self.film = nn.Sequential(
                nn.Linear(c5, film_hidden), nn.ReLU(inplace=True),
                nn.Linear(film_hidden, 2*cdec)
            )

    def forward(self, x):
        feats = self.encoder(x)
        f5 = feats[-1]                         # (B, C5, H5, W5)
        dec = self.decoder(feats)              # (B, Cdec, H, W)

        # cls auxiliar
        cls_logits = self.cls_fc(self.global_pool(f5).flatten(1))

        # FiLM (canal)
        if self.use_film:
            ctx_vec = self.global_pool(f5).flatten(1)  # (B, C5)
            gb = self.film(ctx_vec)                    # (B, 2*Cdec)
            gamma, beta = torch.chunk(gb, 2, dim=1)
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)
            beta  = beta.unsqueeze(-1).unsqueeze(-1)
            dec = (1 + gamma) * dec + beta

        # Atención espacial (pixel-wise)
        if self.use_spatial_attn:
            ctx = self.ctx_proj(f5)                                        # (B, Cdec, H5, W5)
            ctx = F.interpolate(ctx, size=dec.shape[2:], mode='bilinear', align_corners=False)
            attn_in = torch.cat([dec, ctx], dim=1)                         # (B, 2*Cdec, H, W)
            attn_map = self.attn_gen(attn_in)                              # (B, Cdec, H, W)
            dec = dec * (1 + attn_map)                                     # gating residual

        seg_logits = self.segmentation_head(dec)  # sin activación (logits)
        return seg_logits, cls_logits


class DualSatlasFPNGuided(nn.Module):
    """
    DualSatlasFPNGuided:
      - Backbone: SatlasPretrain ResNet50 + FPN + Upsample (devuelve 5 mapas multi-escala).
      - dec: usamos la salida a 1x del Upsample (feats[0]).
      - f5: usamos la salida de menor resolución (feats[-1]) como contexto.
      - Cabeza cls auxiliar: GAP(f5) -> FC.
      - FiLM (canal): GAP(f5) -> MLP -> gamma,beta -> dec = (1+gamma)*dec + beta.
      - Atención espacial: f5 -> 1x1 -> upsample -> concat con dec -> convs -> sigmoid -> gating residual.
      - Devuelve (seg_logits, cls_logits).
    Args:
      model_id (str): ID de checkpoint de SatlasPretrain (p.ej. "Sentinel2_Resnet50_SI_RGB").
      seg_classes (int): # de clases de segmentación (logits).
      cls_classes (int): # de clases para la cabeza auxiliar de clasificación.
      use_spatial_attn (bool): usa atención espacial residual.
      use_film (bool): usa FiLM (modulación por canal).
      film_hidden (int): ancho oculto de la MLP de FiLM.
      freeze_backbone (bool): si True, congela los pesos del backbone+FPN.
    """
    def __init__(
        self,
        model_id: str = "Sentinel2_Resnet50_SI_RGB",
        seg_classes: int = 1,
        cls_classes: int = 1,
        use_spatial_attn: bool = True,
        use_film: bool = True,
        film_hidden: int = 512,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        # Cargamos el modelo preentrenado (backbone + FPN + Upsample).
        # Nota: Weights.get_pretrained_model(..., fpn=True) arma backbone->FPN->Upsample
        # y forward devuelve lista de features [1x, 1/4, 1/8, 1/16, 1/32].
        wm = Weights()
        self.backbone_fpn = wm.get_pretrained_model(model_id, fpn=True)

        # Opcionalmente congelar
        if freeze_backbone:
            for p in self.backbone_fpn.parameters():
                p.requires_grad = False

        # Detectar canales de dec y f5 con un forward dummy (robusto a distintos checkpoints)
        with torch.no_grad():
            # Detectar num_channels esperado de entrada a partir del modelo_id si hace falta:
            # En la práctica, satlas valida canales; aquí usamos 3 por defecto y dejamos
            # que el usuario pase inputs coherentes con el checkpoint.
            dummy = torch.zeros(1, 3, 256, 256)
            feats = self.backbone_fpn(dummy)
            # feats: [p1x, p1/4, p1/8, p1/16, p1/32]
            dec_shape = feats[0].shape  # 1x
            f5_shape = feats[-1].shape  # 1/32
            cdec = dec_shape[1]
            c5 = f5_shape[1]

        self.cdec = cdec
        self.c5 = c5
        self.use_spatial_attn = use_spatial_attn
        self.use_film = use_film

        # Cabeza de clasificación auxiliar desde f5
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.cls_fc = nn.Linear(c5, cls_classes)

        # FiLM (gamma,beta) para modulación por canal de 'dec'
        if use_film:
            self.film = nn.Sequential(
                nn.Linear(c5, film_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(film_hidden, 2 * cdec),
            )

        # Atención espacial residual guiada por f5
        if use_spatial_attn:
            self.ctx_proj = nn.Conv2d(c5, cdec, kernel_size=1)
            self.attn_gen = nn.Sequential(
                nn.Conv2d(cdec * 2, cdec, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(cdec, cdec, kernel_size=3, padding=1),
                nn.Sigmoid(),
            )

        # Cabeza de segmentación (logits), sin activación
        # Un bloque ligero para refinar antes de proyectar a clases
        self.seg_head = nn.Sequential(
            nn.Conv2d(cdec, cdec, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(cdec, seg_classes, kernel_size=1),
        )

    def forward(self, x):
        # Extraer multi-escala: [1x, 1/4, 1/8, 1/16, 1/32], todos con C=cdec (típicamente 128)
        feats = self.backbone_fpn(x)
        dec = feats[0]     # 1x resolución, base para segmentación
        f5 = feats[-1]     # 1/32 resolución, contexto global

        # Cabeza auxiliar de clasificación
        cls_logits = self.cls_fc(self.global_pool(f5).flatten(1))

        # FiLM (modulación por canal)
        if self.use_film:
            ctx_vec = self.global_pool(f5).flatten(1)  # (B, C5)
            gb = self.film(ctx_vec)                    # (B, 2*Cdec)
            gamma, beta = torch.chunk(gb, 2, dim=1)
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)
            beta = beta.unsqueeze(-1).unsqueeze(-1)
            dec = (1.0 + gamma) * dec + beta

        # Atención espacial (pixel-wise) residual
        if self.use_spatial_attn:
            ctx = self.ctx_proj(f5)  # (B, Cdec, H5, W5)
            ctx = F.interpolate(ctx, size=dec.shape[2:], mode="bilinear", align_corners=False)
            attn_in = torch.cat([dec, ctx], dim=1)  # (B, 2*Cdec, H, W)
            attn_map = self.attn_gen(attn_in)       # (B, Cdec, H, W)
            dec = dec * (1.0 + attn_map)

        # Logits de segmentación
        seg_logits = self.seg_head(dec)
        return seg_logits, cls_logits