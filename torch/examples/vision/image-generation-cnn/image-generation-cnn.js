(function(){
  window.registerExample(
    'vision',
    { categoryName: 'Vision', categorySummary: 'CNNs, augmentations, transfer learning, segmentation', topicId: 'image-generation-cnn', topicName: 'Image Generation (Autoencoder upsampling)' },
    {
      id: 'image-generation-cnn',
      name: 'Image Generation (Autoencoder upsampling)',
      tags: ['autoencoder','image-generation'],
      meta: 'Conv encoder + upsampling decoder for image reconstruction',
      description: 'Autoencoder that compresses and reconstructs images; a building block for generation.',
      code: `import torch, torch.nn as nn

class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3,32,3,2,1), nn.ReLU(),
            nn.Conv2d(32,64,3,2,1), nn.ReLU(),
            nn.Conv2d(64,128,3,2,1), nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(128,64,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(64,32,4,2,1), nn.ReLU(),
            nn.ConvTranspose2d(32,3,4,2,1), nn.Tanh(),
        )
    def forward(self, x):
        z = self.enc(x)
        return self.dec(z)

model = AE()
x = torch.randn(2,3,64,64)
print(model(x).shape)`
    }
  );
})();


