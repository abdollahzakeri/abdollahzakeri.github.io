(function(){
  window.registerExample(
    'transformer',
    { categoryName: 'Transformer', categorySummary: 'SDPA, multi-head, masks, encoder/decoder, full model', topicId: 'seq2seq', topicName: 'Transformer Encoder–Decoder (seq2seq)' },
    {
      id: 'seq2seq',
      name: 'Transformer Encoder–Decoder (seq2seq)',
      tags: ['seq2seq','transformer'],
      meta: 'Minimal encoder-decoder assembly from blocks',
      description: 'Assemble encoder and decoder blocks into a seq2seq model with tied embeddings optional.',
      code: `import torch, torch.nn as nn

def causal_mask(T, device=None):
    m = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
    return ~m

class EncoderBlock(nn.Module):
    def __init__(self, dim, heads=8, ff=4):
        super().__init__(); self.ln1=nn.LayerNorm(dim); self.ln2=nn.LayerNorm(dim)
        self.attn=nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp=nn.Sequential(nn.Linear(dim,dim*ff), nn.GELU(), nn.Linear(dim*ff, dim))
    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)[0]
        x = x + self.mlp(self.ln2(x)); return x

class DecoderBlock(nn.Module):
    def __init__(self, dim, heads=8, ff=4):
        super().__init__(); self.ln1=nn.LayerNorm(dim); self.ln2=nn.LayerNorm(dim); self.ln3=nn.LayerNorm(dim)
        self.self_attn=nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cross_attn=nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp=nn.Sequential(nn.Linear(dim,dim*ff), nn.GELU(), nn.Linear(dim*ff, dim))
    def forward(self, x, enc):
        T = x.size(1)
        x = x + self.self_attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=causal_mask(T, x.device), need_weights=False)[0]
        x = x + self.cross_attn(self.ln2(x), enc, enc, need_weights=False)[0]
        x = x + self.mlp(self.ln3(x)); return x

class Seq2Seq(nn.Module):
    def __init__(self, src_vocab=1000, tgt_vocab=1000, dim=128, depth=2, heads=4, ff=4, max_len=128, tie_embeddings=False):
        super().__init__()
        self.src_tok = nn.Embedding(src_vocab, dim)
        self.tgt_tok = nn.Embedding(tgt_vocab, dim)
        self.pos = nn.Parameter(torch.zeros(1, max_len, dim))
        self.enc_blocks = nn.ModuleList([EncoderBlock(dim, heads, ff) for _ in range(depth)])
        self.dec_blocks = nn.ModuleList([DecoderBlock(dim, heads, ff) for _ in range(depth)])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, tgt_vocab)
        if tie_embeddings:
            self.head.weight = self.tgt_tok.weight
    def forward(self, src, tgt):
        S, T = src.size(1), tgt.size(1)
        enc = self.src_tok(src) + self.pos[:, :S]
        for blk in self.enc_blocks: enc = blk(enc)
        dec = self.tgt_tok(tgt) + self.pos[:, :T]
        for blk in self.dec_blocks: dec = blk(dec, enc)
        return self.head(self.ln(dec))

model = Seq2Seq(tie_embeddings=True)
src = torch.randint(0, 1000, (2, 10))
tgt = torch.randint(0, 1000, (2, 8))
print(model(src, tgt).shape)`
    }
  );
})();


