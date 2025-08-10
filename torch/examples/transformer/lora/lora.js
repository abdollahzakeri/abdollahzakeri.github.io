(function(){
  window.registerExample(
    'transformer',
    { categoryName: 'Transformer', categorySummary: 'SDPA, multi-head, masks, encoder/decoder, full model', topicId: 'lora', topicName: 'LoRA for Linear and Attention projections' },
    {
      id: 'lora',
      name: 'LoRA for Linear and Attention projections',
      tags: ['lora','parameter-efficient'],
      meta: 'Low-Rank Adapters for linear projections',
      description: 'Wraps nn.Linear to inject low-rank adapters trained while freezing base weights.',
      code: `import torch, torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4, alpha=1.0, bias=True):
        super().__init__()
        self.base = nn.Linear(in_features, out_features, bias=bias)
        for p in self.base.parameters(): p.requires_grad = False
        self.A = nn.Linear(in_features, r, bias=False)
        self.B = nn.Linear(r, out_features, bias=False)
        self.scaling = alpha / r
        nn.init.kaiming_uniform_(self.A.weight, a=5**0.5)
        nn.init.zeros_(self.B.weight)
    def forward(self, x):
        return self.base(x) + self.B(self.A(x)) * self.scaling

# Example: apply to QKV projections
embed = 32; heads = 4; head_dim = embed // heads
q_proj = LoRALinear(embed, embed)
k_proj = LoRALinear(embed, embed)
v_proj = LoRALinear(embed, embed)
x = torch.randn(2, 5, embed)
q, k, v = q_proj(x), k_proj(x), v_proj(x)
print(q.shape, k.shape, v.shape)`
    }
  );
})();


