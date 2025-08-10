(function(){
  window.registerExample(
    'training',
    { categoryName: 'Training', categorySummary: 'Loops, AMP, clipping, accumulation, checkpoints', topicId: 'single-neuron-backprop', topicName: 'Single Neuron with Backpropagation' },
    {
      id: 'single-neuron-backprop',
      name: 'Single Neuron with Backpropagation',
      tags: ['basics','backprop','sgd'],
      meta: 'Single neuron nn.Module trained with gradient descent',
      description: 'Implements a single logistic neuron and trains it with backprop on a toy dataset.',
      code: `import torch
import torch.nn as nn

class SingleNeuron(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_dim))
        self.bias = nn.Parameter(torch.zeros(()))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x @ self.weight + self.bias
        return torch.sigmoid(z)

# Toy binary classification
torch.manual_seed(0)
X = torch.randn(256, 3)
true_w = torch.tensor([1.5, -2.0, 0.5])
y_prob = torch.sigmoid(X @ true_w)
y = (y_prob > 0.5).float()

model = SingleNeuron(3)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

for epoch in range(20):
    optimizer.zero_grad(set_to_none=True)
    p = model(X).squeeze()
    loss = torch.nn.functional.binary_cross_entropy(p, y)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 5 == 0:
        pred = (p.detach() > 0.5).float()
        acc = (pred == y).float().mean().item()
        print(f'epoch={epoch+1} loss={loss.item():.4f} acc={acc:.3f}')`
    }
  );
})();


