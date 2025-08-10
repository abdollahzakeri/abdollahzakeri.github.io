(function(){
  window.registerExample(
    'transformer',
    { categoryName: 'Transformer', categorySummary: 'SDPA, multi-head, masks, encoder/decoder, full model', topicId: 'encoder-prenorm', topicName: 'Transformer Encoder (Pre-Norm)' },
    {
      id: 'encoder-prenorm',
      name: 'Transformer Encoder (Pre-Norm)',
      tags: ['encoder','prenorm'],
      meta: 'Pre-Norm encoder block from scratch',
      description: 'Applies LayerNorm before attention/MLP and uses residual connections.',
      code: `import torch, torch.nn as nn

class MHA(nn.Module):
    def __init__(self, dim, heads):
        super().__init__(); self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
    def forward(self, x, attn_mask=None): return self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)[0]

class PreNormEncoder(nn.Module):
    def __init__(self, dim, heads=8, ff=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.mha = MHA(dim, heads)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim*ff), nn.GELU(), nn.Linear(dim*ff, dim))
    def forward(self, x, attn_mask=None):
        x = x + self.mha(self.ln1(x), attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x

print(PreNormEncoder(64)(torch.randn(2,10,64)).shape)`
    }
  );
})();


