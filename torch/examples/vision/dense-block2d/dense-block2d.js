(function(){
  window.registerExample(
    'vision',
    { categoryName: 'Vision', categorySummary: 'CNNs, augmentations, transfer learning, segmentation', topicId: 'dense-block2d', topicName: 'Dense Block with 2D Convolutions' },
    {
      id: 'dense-block2d',
      name: 'Dense Block with 2D Convolutions',
      tags: ['cnn','densenet'],
      meta: 'Dense connectivity pattern over 2D convs',
      description: 'Implements a small DenseNet-like dense block (concatenate previous features).',
      code: `import torch
import torch.nn as nn

class DenseLayer(nn.Module):
    def __init__(self, in_ch, growth):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, growth, kernel_size=3, padding=1, bias=False),
        )
    def forward(self, x):
        y = self.net(x)
        return torch.cat([x, y], dim=1)

class DenseBlock(nn.Module):
    def __init__(self, in_ch, growth, layers):
        super().__init__()
        mods = []
        ch = in_ch
        for _ in range(layers):
            mods.append(DenseLayer(ch, growth))
            ch += growth
        self.block = nn.Sequential(*mods)
        self.out_ch = ch
    def forward(self, x): return self.block(x)

x = torch.randn(2, 16, 32, 32)
db = DenseBlock(16, growth=8, layers=3)
print(db(x).shape, 'out_channels=', db.out_ch)`
    }
  );
})();


