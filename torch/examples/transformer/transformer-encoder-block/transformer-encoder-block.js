(function(){
  window.registerExample(
    'transformer',
    { categoryName: 'Transformer', categorySummary: 'SDPA, multi-head, masks, encoder/decoder, full model', topicId: 'transformer-encoder-block', topicName: 'Transformer encoder block' },
    {
      id: 'transformer-encoder-block',
      name: 'Transformer encoder block',
      tags: ['encoder','block'],
      meta: 'Self-attention + MLP with LayerNorm and residuals',
      description: 'A transformer encoder block using PyTorch MultiheadAttention and MLP.',
      code: `import torch, torch.nn as nn
class EncoderBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_ratio=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim*mlp_ratio), nn.GELU(), nn.Linear(dim*mlp_ratio, dim))
    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=attn_mask, need_weights=False)[0]
        x = x + self.mlp(self.ln2(x))
        return x
print('ok')`
    }
  );
})();


