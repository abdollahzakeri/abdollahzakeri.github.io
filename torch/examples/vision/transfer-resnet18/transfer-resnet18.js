(function(){
  window.registerExample(
    'vision',
    { categoryName: 'Vision', categorySummary: 'CNNs, augmentations, transfer learning, segmentation', topicId: 'transfer-resnet18', topicName: 'Transfer learning (ResNet18)' },
    {
      id: 'transfer-resnet18',
      name: 'Transfer learning (ResNet18)',
      tags: ['transfer','torchvision'],
      meta: 'Replace final layer and fine-tune',
      description: 'Load a ResNet18, replace the classifier head for new num_classes.',
      code: `import torch.nn as nn
from torchvision.models import resnet18
model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 10)
print(model.fc)`
    }
  );
})();


