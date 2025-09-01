import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalTransformerBlock(nn.Module):
    def __init__(self, dim_q, dim_kv, n_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim_q, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim_q)
        self.ffn = nn.Sequential(
            nn.Linear(dim_q, dim_q * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_q * 4, dim_q)
        )
        self.norm2 = nn.LayerNorm(dim_q)

    def forward(self, q, kv, kv_mask=None):
        attn_output, attn_weights = self.attn(q, kv, kv, key_padding_mask=kv_mask, need_weights=True)
        x = self.norm1(q + attn_output)
        ffn_output = self.ffn(x)
        out = self.norm2(x + ffn_output)
        return out, attn_weights


class CrossModalTransformer(nn.Module):
    def __init__(self, dim_text=300, dim_audio=32, dim_visual=2048, dim_model=128, n_heads=4, dropout=0.1):
        super().__init__()

        self.dim_model = dim_model

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

        self.cross_blocks = nn.ModuleDict({
            'text_audio': CrossModalTransformerBlock(dim_model, dim_model, n_heads, dropout),
            'text_visual': CrossModalTransformerBlock(dim_model, dim_model, n_heads, dropout),
            'audio_text': CrossModalTransformerBlock(dim_model, dim_model, n_heads, dropout),
            'audio_visual': CrossModalTransformerBlock(dim_model, dim_model, n_heads, dropout),
            'visual_text': CrossModalTransformerBlock(dim_model, dim_model, n_heads, dropout),
            'visual_audio': CrossModalTransformerBlock(dim_model, dim_model, n_heads, dropout),
        })

        self.fusion_norm = nn.LayerNorm(dim_model * 6)
        self.classifier = nn.Sequential(
            nn.Linear(dim_model * 6, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 7)  # MELD: 7 emotion classes
        )

    def forward(self, text, audio, visual, mask=None):
        log = {}
        pooled = []

        # Projection
        text = self.text_proj(text)
        audio = self.audio_proj(audio)
        visual = self.visual_proj(visual)

        # Default mask: full path 활성화
        if mask is None:
            mask = [1] * 6

        # Path 활성화 여부에 따라 분기 처리
        if mask[0]:
            ta, log['text->audio'] = self.cross_blocks['text_audio'](text, audio)
            pooled.append(ta.mean(1))
        if mask[1]:
            tv, log['text->visual'] = self.cross_blocks['text_visual'](text, visual)
            pooled.append(tv.mean(1))
        if mask[2]:
            at, log['audio->text'] = self.cross_blocks['audio_text'](audio, text)
            pooled.append(at.mean(1))
        if mask[3]:
            av, log['audio->visual'] = self.cross_blocks['audio_visual'](audio, visual)
            pooled.append(av.mean(1))
        if mask[4]:
            vt, log['visual->text'] = self.cross_blocks['visual_text'](visual, text)
            pooled.append(vt.mean(1))
        if mask[5]:
            va, log['visual->audio'] = self.cross_blocks['visual_audio'](visual, audio)
            pooled.append(va.mean(1))

        if len(pooled) == 0:
            raise ValueError("At least one path must be active in the mask.")

        # Zero-padding to fixed 6*dim_model
        while len(pooled) < 6:
            pooled.append(torch.zeros_like(pooled[0]))

        fusion = torch.cat(pooled, dim=-1)  # shape: [B, 6 * dim_model]
        fusion = self.fusion_norm(fusion)
        logits = self.classifier(fusion)

        return logits, {k: v.mean().item() for k, v in log.items()}
