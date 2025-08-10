(function(){
  window.registerExample(
    'devices-memory',
    { categoryName: 'Devices & Memory', categorySummary: 'Device moves, pin_memory, inference_mode, precision, parallel', topicId: 'to-device-helper', topicName: 'to_device helper' },
    {
      id: 'to-device-helper',
      name: 'to_device helper',
      tags: ['device'],
      meta: 'Move nested tensors to a device recursively',
      description: 'Utility function to move arbitrarily nested tensors to a device.',
      code: `import torch
def to_device(x, device):
    if torch.is_tensor(x): return x.to(device)
    if isinstance(x, (list, tuple)): return type(x)(to_device(v, device) for v in x)
    if isinstance(x, dict): return {k: to_device(v, device) for k,v in x.items()}
    return x
print('ok')`
    }
  );
})();


