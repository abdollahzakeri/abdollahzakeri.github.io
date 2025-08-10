(function(){
  window.registerExample(
    'optimization',
    { categoryName: 'Optimization', categorySummary: 'Optimizers, schedulers, param groups, freezing', topicId: 'param-groups', topicName: 'Parameter groups (LR/WD)' },
    {
      id: 'param-groups',
      name: 'Parameter groups (LR/WD)',
      tags: ['optimizer','param-groups'],
      meta: 'Different weight decay for decay/no_decay parameter sets',
      description: 'Split parameters into decay and no_decay groups by name.',
      code: `import torch.nn as nn, torch.optim as optim
model = nn.Sequential(nn.Linear(10, 10), nn.LayerNorm(10), nn.Linear(10, 2))
decay, no_decay = [], []
for name, p in model.named_parameters():
    if p.requires_grad:
        (no_decay if name.endswith('bias') or 'LayerNorm' in name else decay).append(p)
opt = optim.AdamW([
    {'params': decay, 'weight_decay': 0.01},
    {'params': no_decay, 'weight_decay': 0.0},
], lr=3e-4)
print(opt)`
    }
  );
})();


