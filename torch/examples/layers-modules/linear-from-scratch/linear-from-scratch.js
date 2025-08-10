(function(){
  window.registerExample(
    'layers-modules',
    { categoryName: 'Layers & Modules', categorySummary: 'Implementations of common layers from scratch', topicId: 'linear-from-scratch', topicName: 'Linear layer (from scratch)' },
    {
      id: 'linear-from-scratch',
      name: 'Linear layer (from scratch)',
      tags: ['modules','linear'],
      meta: 'Custom nn.Module implementing affine transformation',
      description: 'A minimal linear layer with explicit parameters and forward.',
      code: `import torch
import torch.nn as nn
import math

class MyLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    def forward(self, x):
        y = x @ self.weight.t()
        return y + self.bias if self.bias is not None else y

layer = MyLinear(4, 3)
print(layer(torch.randn(2,4)).shape)`
    }
  );
})();


