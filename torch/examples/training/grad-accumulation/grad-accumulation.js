(function(){
  window.registerExample(
    'training',
    { categoryName: 'Training', categorySummary: 'Loops, AMP, clipping, accumulation, checkpoints', topicId: 'grad-accumulation', topicName: 'Gradient accumulation' },
    {
      id: 'grad-accumulation',
      name: 'Gradient accumulation',
      tags: ['large-batch','accumulate'],
      meta: 'Accumulate grads over micro-batches before optimizer.step()',
      description: 'Simulate larger batch sizes by dividing loss and stepping every k micro-batches.',
      code: `import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 2))
opt = optim.AdamW(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()

accum_steps = 4
opt.zero_grad(set_to_none=True)
for step in range(8):
    x = torch.randn(16, 10)
    y = torch.randint(0, 2, (16,))
    logits = model(x)
    loss = crit(logits, y) / accum_steps
    loss.backward()
    if (step + 1) % accum_steps == 0:
        opt.step(); opt.zero_grad(set_to_none=True)
print('done')`
    }
  );
})();


