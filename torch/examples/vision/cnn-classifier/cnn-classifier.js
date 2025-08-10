(function(){
  window.registerExample(
    'vision',
    { categoryName: 'Vision', categorySummary: 'CNNs, augmentations, transfer learning, segmentation', topicId: 'cnn-classifier', topicName: 'Simple CNN classifier' },
    {
      id: 'cnn-classifier',
      name: 'Simple CNN classifier (CIFAR-like)',
      tags: ['cnn','classification'],
      meta: 'End-to-end CNN nn.Module with forward and output shapes',
      description: 'Small CNN model demonstrating complete forward pass and classifier head.',
      code: `import torch
import torch.nn as nn

class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

x = torch.randn(2, 3, 32, 32)
print(SmallCNN()(x).shape)`
    }
  );
})();


