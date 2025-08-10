(function(){
  window.registerExample(
    'transformer',
    { categoryName: 'Transformer', categorySummary: 'SDPA, multi-head, masks, encoder/decoder, full model', topicId: 'multi-head-attn', topicName: 'Multi-head attention' },
    {
      id: 'multi-head-attn',
      name: 'Multi-head attention (from scratch)',
      tags: ['attention','multi-head'],
      meta: 'nn.Module implementing MHA with configurable heads and masking',
      description: 'A complete multi-head self-attention module with masking and projection.',
      code: `import torch
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        assert embed_dim % num_heads == 0, 'embed_dim must be divisible by num_heads'
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, T, C)
        B, T, C = x.shape
        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        # (B, h, T, d)
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if attn_mask is not None:
            # attn_mask: True for keep, False for mask out (or use -inf fill)
            scores = scores.masked_fill(~attn_mask, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        out = attn @ v  # (B, h, T, d)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)

# Demo
B, T, C, H = 2, 5, 32, 4
x = torch.randn(B, T, C)
mha = MultiHeadSelfAttention(C, H)
print(mha(x).shape)`
    }
  );
})();


