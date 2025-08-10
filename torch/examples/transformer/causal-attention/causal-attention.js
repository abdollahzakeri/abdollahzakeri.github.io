(function(){
  window.registerExample(
    'transformer',
    { categoryName: 'Transformer', categorySummary: 'SDPA, multi-head, masks, encoder/decoder, full model', topicId: 'causal-attention', topicName: 'Masked (causal) Self-Attention and multi headed causal attention' },
    {
      id: 'causal-attention',
      name: 'Masked (causal) Self-Attention and multi headed causal attention',
      tags: ['attention','causal','multi-head'],
      meta: 'Implements causal mask and applies it in MHA',
      description: 'Lower-triangular mask with a Multi-Head attention module for autoregressive decoding.',
      code: `import torch, torch.nn as nn

def causal_mask(T, device=None):
    m = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
    return ~m  # True means masked in nn.MultiheadAttention attn_mask convention

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__(); self.mha = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln = nn.LayerNorm(dim)
    def forward(self, x):
        T = x.size(1)
        mask = causal_mask(T, x.device)
        y = self.mha(self.ln(x), self.ln(x), self.ln(x), attn_mask=mask, need_weights=False)[0]
        return x + y

x = torch.randn(2, 6, 32)
print(CausalSelfAttention(32)(x).shape)`
    }
  );
})();


