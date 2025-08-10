(function(){
  window.registerExample(
    'evaluation',
    { categoryName: 'Evaluation', categorySummary: 'Eval/inference loops, metrics, confusion matrix', topicId: 'eval-loop', topicName: 'Evaluation loop (no_grad)' },
    {
      id: 'eval-loop',
      name: 'Evaluation loop (no_grad)',
      tags: ['eval','no_grad'],
      meta: 'Switch to eval mode, disable grads, compute accuracy',
      description: 'A typical evaluation loop with model.eval() and torch.no_grad().',
      code: `import torch
import torch.nn as nn

model = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 3))
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for _ in range(10):
        x = torch.randn(8, 10)
        y = torch.randint(0, 3, (8,))
        logits = model(x)
        pred = logits.argmax(dim=-1)
        correct += (pred == y).sum().item(); total += y.numel()
print('acc:', correct / max(1, total))`
    }
  );
})();


