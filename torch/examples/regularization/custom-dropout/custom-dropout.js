(function(){
  window.registerExample(
    'regularization',
    { categoryName: 'Regularization', categorySummary: 'Dropout, label smoothing, mixup, cutmix, stochastic depth', topicId: 'custom-dropout', topicName: 'Custom Dropout Layer' },
    {
      id: 'custom-dropout',
      name: 'Custom Dropout Layer',
      tags: ['dropout','regularization'],
      meta: 'Implements Dropout as an nn.Module',
      description: 'Custom Dropout that randomly zeroes inputs at train time and rescales.',
      code: `import torch
import torch.nn as nn

class MyDropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__(); self.p = p
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0: return x
        keep = 1 - self.p
        mask = torch.empty_like(x).bernoulli_(keep)
        return x * mask / keep

x = torch.ones(5)
drop = MyDropout(0.8).train()
print(drop(x))`
    }
  );
})();


