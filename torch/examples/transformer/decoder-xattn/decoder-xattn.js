(function(){
  window.registerExample(
    'transformer',
    { categoryName: 'Transformer', categorySummary: 'SDPA, multi-head, masks, encoder/decoder, full model', topicId: 'decoder-xattn', topicName: 'Transformer Decoder with Cross-Attention' },
    {
      id: 'decoder-xattn',
      name: 'Transformer Decoder with Cross-Attention',
      tags: ['decoder','cross-attention'],
      meta: 'Masked self-attn + cross-attn + MLP with pre-norm',
      description: 'Builds a decoder block including causal self-attention and encoder-decoder attention.',
      code: `import torch, torch.nn as nn

def causal_mask(T, device=None):
    m = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
    return ~m

class DecoderBlock(nn.Module):
    def __init__(self, dim, heads=8, ff=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim*ff), nn.GELU(), nn.Linear(dim*ff, dim))
    def forward(self, x, enc):
        T = x.size(1)
        x = x + self.self_attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=causal_mask(T, x.device), need_weights=False)[0]
        x = x + self.cross_attn(self.ln2(x), enc, enc, need_weights=False)[0]
        x = x + self.mlp(self.ln3(x))
        return x

enc = torch.randn(2, 7, 64)
dec = torch.randn(2, 6, 64)
print(DecoderBlock(64)(dec, enc).shape)`
    }
  );
})();


