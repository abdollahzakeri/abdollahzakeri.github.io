(function(){
  window.registerExample(
    'training',
    { categoryName: 'Training', categorySummary: 'Loops, AMP, clipping, accumulation, checkpoints', topicId: 'cnn-train-backprop', topicName: 'Simple CNN Training with Backpropagation' },
    {
      id: 'cnn-train-backprop',
      name: 'Simple CNN Training with Backpropagation',
      tags: ['cnn','training','backprop'],
      meta: 'End-to-end CNN training loop with optimizer and CE loss',
      description: 'Trains a tiny CNN on random data to demonstrate the full backprop loop.',
      code: `import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class TinyCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(32*8*8, num_classes))
    def forward(self, x): return self.classifier(self.features(x))

X = torch.randn(256, 3, 32, 32)
y = torch.randint(0, 10, (256,))
loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)

model = TinyCNN(); opt = optim.AdamW(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()
for epoch in range(3):
    total = 0.0
    for xb, yb in loader:
        opt.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = crit(logits, yb)
        loss.backward(); opt.step()
        total += loss.item() * xb.size(0)
    print('epoch', epoch, 'loss', total / len(loader.dataset))`
    }
  );
})();


