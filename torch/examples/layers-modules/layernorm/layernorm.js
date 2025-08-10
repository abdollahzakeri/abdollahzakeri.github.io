(function(){
  window.registerExample(
    'layers-modules',
    { categoryName: 'Layers & Modules', categorySummary: 'Implementations of common layers from scratch', topicId: 'layernorm', topicName: 'LayerNorm (from scratch)' },
    {
      id: 'layernorm',
      name: 'LayerNorm (from scratch)',
      tags: ['normalization','layernorm'],
      meta: 'Implements LayerNorm across last dimension',
      description: 'Compute mean/var on last dim and apply affine params.',
      code: `import torch
import torch.nn as nn

class MyLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        xhat = (x - mean) / torch.sqrt(var + self.eps)
        return xhat * self.weight + self.bias

x = torch.randn(2, 5, 8)
ln = MyLayerNorm(8)
print(ln(x).shape)`
    }
  );
})();


