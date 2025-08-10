(function(){
  window.registerExample(
    'transformer',
    { categoryName: 'Transformer', categorySummary: 'SDPA, multi-head, masks, encoder/decoder, full model', topicId: 'causal-lm-head', topicName: 'Causal LM head with tied embeddings' },
    {
      id: 'causal-lm-head',
      name: 'Causal LM head with tied input embeddings',
      tags: ['lm','tied-embeddings'],
      meta: 'Tie token embedding weights to output projection',
      description: 'Demonstrates tied input/output embeddings for a language modeling head.',
      code: `import torch, torch.nn as nn

class TinyLM(nn.Module):
    def __init__(self, vocab=1000, dim=128):
        super().__init__()
        self.tok = nn.Embedding(vocab, dim)
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab, bias=False)
        self.head.weight = self.tok.weight
    def forward(self, x):
        h = self.tok(x)
        return self.head(self.ln(h))

lm = TinyLM()
print(lm(torch.randint(0,1000,(2,6))).shape)`
    }
  );
})();


