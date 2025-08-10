(function(){
  window.registerExample(
    'training',
    { categoryName: 'Training', categorySummary: 'Loops, AMP, clipping, accumulation, checkpoints', topicId: 'amp-autocast-gradscaler', topicName: 'AMP (autocast + GradScaler)' },
    {
      id: 'amp-autocast-gradscaler',
      name: 'AMP (autocast + GradScaler)',
      tags: ['amp','mixed-precision','fp16'],
      meta: 'Mixed precision forward, backward, and optimizer step',
      description: 'Demonstrates torch.cuda.amp.autocast and GradScaler for stable mixed-precision training.',
      code: `import torch
import torch.nn as nn
import torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 10)).to(device)
opt = optim.AdamW(model.parameters(), lr=1e-3)
scaler = torch.cuda.amp.GradScaler(enabled=(device=='cuda'))
crit = nn.CrossEntropyLoss()

x = torch.randn(64, 128, device=device)
y = torch.randint(0, 10, (64,), device=device)

opt.zero_grad(set_to_none=True)
with torch.cuda.amp.autocast(enabled=(device=='cuda')):
    logits = model(x)
    loss = crit(logits, y)
scaler.scale(loss).backward()
scaler.step(opt)
scaler.update()
print('done')`
    }
  );
})();


