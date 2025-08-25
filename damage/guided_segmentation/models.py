import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import torch.nn.functional as F

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
