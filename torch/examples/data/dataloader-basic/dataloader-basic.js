(function(){
  window.registerExample(
    'data',
    { categoryName: 'Data', categorySummary: 'Datasets, DataLoaders, samplers, collation, transforms', topicId: 'dataloader-basic', topicName: 'DataLoader (basic)' },
    {
      id: 'dataloader-basic',
      name: 'DataLoader (basic)',
      tags: ['dataloader','batching'],
      meta: 'Batch a TensorDataset and iterate minibatches',
      description: 'Create a DataLoader with shuffling and iterate over mini-batches.',
      code: `import torch
from torch.utils.data import DataLoader, TensorDataset

X = torch.randn(100, 10)
y = torch.randint(0, 3, (100,))
ds = TensorDataset(X, y)
loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=0)

for xb, yb in loader:
    print(xb.shape, yb.shape)
    break`
    }
  );
})();


