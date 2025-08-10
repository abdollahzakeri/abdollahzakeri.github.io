(function(){
  window.registerExample(
    'training',
    { categoryName: 'Training', categorySummary: 'Loops, AMP, clipping, accumulation, checkpoints', topicId: 'basic-training-loop', topicName: 'Basic training loop' },
    {
      id: 'basic-training-loop',
      name: 'Basic training loop',
      tags: ['loop','optimizer','loss'],
      meta: 'Complete end-to-end mini training with nn.Module',
      description: 'A full supervised training example with a small nn.Module, optimizer, dataloader, and logging.',
      code: `import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class MLP(nn.Module):
    def __init__(self, in_dim=10, hidden=32, out_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):
        return self.net(x)

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * xb.size(0)
            pred = logits.argmax(dim=-1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
    return total_loss / len(loader.dataset), correct / max(1, total)

# Synthetic dataset
X_train = torch.randn(512, 10)
y_train = torch.randint(0, 3, (512,))
X_val   = torch.randn(128, 10)
y_val   = torch.randint(0, 3, (128,))
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=64)

model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

for epoch in range(5):
    tr_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    va_loss, va_acc = evaluate(model, val_loader, criterion)
    print(f"epoch={epoch} train_loss={tr_loss:.4f} val_loss={va_loss:.4f} val_acc={va_acc:.3f}")`
    }
  );
})();


