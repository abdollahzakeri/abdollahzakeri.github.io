(function(){
  window.registerExample(
    'initialization',
    { categoryName: 'Initialization', categorySummary: 'Xavier, Kaiming, custom rules', topicId: 'xavier-init', topicName: 'Xavier/Glorot init' },
    {
      id: 'xavier-init',
      name: 'Xavier/Glorot init',
      tags: ['init','xavier'],
      meta: 'Apply Xavier init to linear layers',
      description: 'Initializer function applied to all Linear layers in a network.',
      code: `import torch.nn as nn
def init_xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
net = nn.Sequential(nn.Linear(8,16), nn.ReLU(), nn.Linear(16,4))
net.apply(init_xavier)
print('ok')`
    }
  );
})();


