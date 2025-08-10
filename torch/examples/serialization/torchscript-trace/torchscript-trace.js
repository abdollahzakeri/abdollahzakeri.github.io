(function(){
  window.registerExample(
    'serialization',
    { categoryName: 'Serialization', categorySummary: 'Save/load checkpoints, TorchScript, ONNX', topicId: 'torchscript-trace', topicName: 'TorchScript: trace' },
    {
      id: 'torchscript-trace',
      name: 'TorchScript: trace',
      tags: ['torchscript','trace'],
      meta: 'Export a traced module for inference',
      description: 'Trace a simple model and save the TorchScript artifact.',
      code: `import torch, torch.nn as nn
m = nn.Sequential(nn.Linear(10, 16), nn.ReLU(), nn.Linear(16, 2))
tm = torch.jit.trace(m, torch.randn(1,10))
tm.save('m_traced.pt')
print('saved m_traced.pt')`
    }
  );
})();


