(function(){
  window.registerExample(
    'layers-modules',
    { categoryName: 'Layers & Modules', categorySummary: 'Implementations of common layers from scratch', topicId: 'residual-block', topicName: 'Residual block' },
    {
      id: 'residual-block',
      name: 'Residual block',
      tags: ['residual','skip-connection'],
      meta: 'MLP residual block with skip connection',
      description: 'A basic residual MLP block demonstrating skip connections.',
      code: `import torch
import torch.nn as nn

class Residual(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)
        )
    def forward(self, x):
        return x + self.net(x)

print(Residual(16)(torch.randn(2,16)).shape)`
    }
  );
})();


