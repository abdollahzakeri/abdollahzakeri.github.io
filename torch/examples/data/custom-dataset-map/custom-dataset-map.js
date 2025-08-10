(function(){
  window.registerExample(
    'data',
    { categoryName: 'Data', categorySummary: 'Datasets, DataLoaders, samplers, collation, transforms', topicId: 'custom-dataset-map', topicName: 'Custom Dataset (map-style)' },
    {
      id: 'custom-dataset-map',
      name: 'Custom Dataset (map-style)',
      tags: ['dataset','map-style','data'],
      meta: 'Complete nn.Module usage not required here; focus is Dataset API',
      description: 'A complete map-style Dataset returning tensors, plus a quick usage demo with DataLoader.',
      code: `import torch
from torch.utils.data import Dataset, DataLoader

class ToyMapDataset(Dataset):
    """Map-style dataset returning (x, y) pairs as tensors."""
    def __init__(self, data, targets):
        self.data = torch.as_tensor(data, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.long)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        return x, y

# End-to-end usage
X = torch.randn(100, 10)
y = torch.randint(0, 3, (100,))
ds = ToyMapDataset(X, y)
loader = DataLoader(ds, batch_size=16, shuffle=True)
for xb, yb in loader:
    print(xb.shape, yb.shape)
    break`
    }
  );
})();


