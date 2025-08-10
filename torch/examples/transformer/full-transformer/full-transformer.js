(function(){
  window.registerExample(
    'transformer',
    { categoryName: 'Transformer', categorySummary: 'SDPA, multi-head, masks, encoder/decoder, full model', topicId: 'full-transformer', topicName: 'Full Transformer LM (tiny)' },
    {
      id: 'full-transformer',
      name: 'Full Transformer LM (tiny)',
      tags: ['transformer','language-model'],
      meta: 'Tiny Transformer for autoregressive LM',
      description: 'Minimal transformer language model using encoder blocks with causal mask.',
      code: `import torch, torch.nn as nn

class EncoderBlock(nn.Module):
    def __init__(self, dim, heads=4, ff=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim*ff), nn.GELU(), nn.Linear(dim*ff, dim))
    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=attn_mask, need_weights=False)[0]
        x = x + self.mlp(self.ln2(x))
        return x

class TinyTransformerLM(nn.Module):
    def __init__(self, vocab=1000, dim=128, depth=2, heads=4, ff=4, max_len=128):
        super().__init__()
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Parameter(torch.zeros(1, max_len, dim))
        self.blocks = nn.ModuleList([EncoderBlock(dim, heads, ff) for _ in range(depth)])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab)
    def forward(self, x):
        B, T = x.shape
        h = self.tok(x) + self.pos[:, :T]
        mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
        attn_mask = ~mask
        for blk in self.blocks:
            h = blk(h, attn_mask)
        h = self.ln(h)
        return self.head(h)

model = TinyTransformerLM()
print(model(torch.randint(0, 1000, (2, 16))).shape)`
    }
  );
})();


