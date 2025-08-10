(function(){
  window.registerExample(
    'optimization',
    { categoryName: 'Optimization', categorySummary: 'Optimizers, schedulers, param groups, freezing', topicId: 'adamw', topicName: 'AdamW' },
    {
      id: 'adamw',
      name: 'AdamW',
      tags: ['optimizer','adamw'],
      meta: 'AdamW with weight decay decoupled',
      description: 'Create AdamW optimizer for a small model.',
      code: `import torch.nn as nn, torch.optim as optim
net = nn.Linear(8, 2)
opt = optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.01)
print(opt)`
    }
  );
})();


