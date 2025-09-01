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
    Cross-Modal SuperNet model with shared transformer block and FLOPs estimation for NAS.
    """
    def __init__(self, dim_text=300, dim_audio=32, dim_visual=2048, dim_model=128, n_heads=4, dropout=0.1):
        super().__init__()
        self.dim_model = dim_model

        # Projection layers: project input to shared dimension
        self.text_proj = nn.Sequential(
            nn.Linear(dim_text, 512),
            nn.ReLU(),
            nn.Linear(512, dim_model)
        )
        self.audio_proj = nn.Sequential(
            nn.Linear(dim_audio, 128),
            nn.ReLU(),
            nn.Linear(128, dim_model)
        )
        self.visual_proj = nn.Sequential(
            nn.Linear(dim_visual, 1024),
            nn.ReLU(),
            nn.Linear(1024, dim_model)
        )

        # Shared Transformer block
        self.shared_block = SharedCrossModalTransformerBlock(dim_model, n_heads, dropout)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(dim_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 7)  # MELD: 7-class emotion classification
        )

        # FLOPs per modal path (estimated unit cost)
        self.path_flops = self.get_default_path_flops()

    def forward(self, text, audio, visual, mask=None, return_flops=False):
        """
        Args:
            text, audio, visual: input tensors [B, T, F]
            mask: 6-bit list (e.g. [1,0,1,0,0,1]) for active attention paths
            return_flops: if True, also return total estimated FLOPs

        Returns:
            logits: [B, 7]
            fused: [B, H]
            flops (optional): float
        """
        if mask is None:
            mask = [1] * 6  # Use all paths by default

        text = self.text_proj(text)
        audio = self.audio_proj(audio)
        visual = self.visual_proj(visual)

        fused_outputs = []

        if mask[0]:  # T→A
            fused_outputs.append(self.shared_block(text, audio).mean(1))
        if mask[1]:  # T→V
            fused_outputs.append(self.shared_block(text, visual).mean(1))
        if mask[2]:  # A→T
            fused_outputs.append(self.shared_block(audio, text).mean(1))
        if mask[3]:  # A→V
            fused_outputs.append(self.shared_block(audio, visual).mean(1))
        if mask[4]:  # V→T
            fused_outputs.append(self.shared_block(visual, text).mean(1))
        if mask[5]:  # V→A
            fused_outputs.append(self.shared_block(visual, audio).mean(1))

        if len(fused_outputs) == 0:
            raise ValueError("At least one cross-modal path must be active in the mask.")

        fused = torch.mean(torch.stack(fused_outputs, dim=0), dim=0)
        logits = self.classifier(fused)

        if return_flops:
            flops = self.compute_flops(mask)
            return logits, fused, flops
        else:
            return logits, fused

    def compute_flops(self, mask):
        """
        Estimate total FLOPs for a given path mask (sum of active path costs)
        """
        if len(mask) != 6:
            raise ValueError("Mask must be a list of 6 binary values.")
        return sum(self.path_flops[i] for i in range(6) if mask[i])

    @staticmethod
    def get_default_path_flops():
        """
        Default static FLOPs cost per attention path.
        Index order:
            0: T→A
            1: T→V
            2: A→T
            3: A→V
            4: V→T
            5: V→A
        """
        return {
            0: 1.0,
            1: 1.1,
            2: 1.0,
            3: 1.2,
            4: 1.1,
            5: 1.0
        }
