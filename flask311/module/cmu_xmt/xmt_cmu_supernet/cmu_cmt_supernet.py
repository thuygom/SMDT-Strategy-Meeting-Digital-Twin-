import torch
import torch.nn as nn
import torch.nn.functional as F

class SharedCrossModalTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, q, kv):
        attn_output, _ = self.attn(q, kv, kv)
        x = self.norm1(q + attn_output)
        ffn_output = self.ffn(x)
        return self.norm2(x + ffn_output)


class CrossModalSuperNet(nn.Module):
    """
    Cross-Modal SuperNet for CMU-MOSEI.
    Shared Transformer block + attention path mask + FLOPs estimation.
    """
    def __init__(self, dim_text=300, dim_audio=74, dim_visual=35, dim_model=128, n_heads=4, dropout=0.1):
        super().__init__()
        self.dim_model = dim_model

        # Projection layers: input modality → shared dim
        self.text_proj = nn.Sequential(
            nn.Linear(dim_text, 256),
            nn.ReLU(),
            nn.Linear(256, dim_model)
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(dim_audio, 128),
            nn.ReLU(),
            nn.Linear(128, dim_model)
        )
        self.visual_proj = nn.Sequential(
            nn.Linear(dim_visual, 64),
            nn.ReLU(),
            nn.Linear(64, dim_model)
        )

        # Shared Transformer block
        self.shared_block = SharedCrossModalTransformerBlock(dim_model, n_heads, dropout)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(dim_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 7)  # CMU-MOSEI 7-class
        )

        # FLOPs per modal path (custom static estimate)
        self.path_flops = self.get_default_path_flops()

    def forward(self, text, audio, visual, mask=None, return_flops=False):
        """
        Args:
            text/audio/visual: [B, T, F]
            mask: list of 6 binary flags (path 활성화)
            return_flops: True일 경우 총 연산량 반환

        Returns:
            logits: [B, 7]
            fused: [B, dim_model]
            flops (optional): float
        """
        if mask is None:
            mask = [1] * 6  # 모든 path 사용

        # Project to shared space
        text = self.text_proj(text)
        audio = self.audio_proj(audio)
        visual = self.visual_proj(visual)

        fused_outputs = []

        if mask[0]:  # text → audio
            fused_outputs.append(self.shared_block(text, audio).mean(1))
        if mask[1]:  # text → visual
            fused_outputs.append(self.shared_block(text, visual).mean(1))
        if mask[2]:  # audio → text
            fused_outputs.append(self.shared_block(audio, text).mean(1))
        if mask[3]:  # audio → visual
            fused_outputs.append(self.shared_block(audio, visual).mean(1))
        if mask[4]:  # visual → text
            fused_outputs.append(self.shared_block(visual, text).mean(1))
        if mask[5]:  # visual → audio
            fused_outputs.append(self.shared_block(visual, audio).mean(1))

        if len(fused_outputs) == 0:
            raise ValueError("At least one cross-modal path must be active.")

        fused = torch.mean(torch.stack(fused_outputs, dim=0), dim=0)
        logits = self.classifier(fused)

        if return_flops:
            return logits, fused, self.compute_flops(mask)
        else:
            return logits, fused

    def compute_flops(self, mask):
        """
        Sum estimated FLOPs for activated paths.
        """
        if len(mask) != 6:
            raise ValueError("mask must be length 6")
        return sum(self.path_flops[i] for i in range(6) if mask[i])

    @staticmethod
    def get_default_path_flops():
        """
        Approximate FLOPs per path for CMU-MOSEI config.
        Order:
          0: text→audio
          1: text→visual
          2: audio→text
          3: audio→visual
          4: visual→text
          5: visual→audio
        """
        return {
            0: 1.0,
            1: 1.0,
            2: 1.0,
            3: 1.0,
            4: 1.0,
            5: 1.0
        }
