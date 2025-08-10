(function(){
  window.registerExample(
    'generative',
    { categoryName: 'Generative', categorySummary: 'VAE, GAN, diffusion step, autoregressive sampling', topicId: 'gan', topicName: 'GAN for image generation (DCGAN-style)' },
    {
      id: 'gan',
      name: 'GAN for image generation (DCGAN-style)',
      tags: ['gan','image-generation'],
      meta: 'ConvTranspose2d Generator and Conv Discriminator with one training iteration',
      description: 'Defines DCGAN-like G/D modules and runs a minimal training iteration on synthetic real images.',
      code: `import torch, torch.nn as nn

Z=100; IMG=64; CH=64

class Generator(nn.Module):
    def __init__(self, zdim=Z, ch=CH):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(zdim, ch*8, 4, 1, 0, bias=False), nn.BatchNorm2d(ch*8), nn.ReLU(True),
            nn.ConvTranspose2d(ch*8, ch*4, 4, 2, 1, bias=False), nn.BatchNorm2d(ch*4), nn.ReLU(True),
            nn.ConvTranspose2d(ch*4, ch*2, 4, 2, 1, bias=False), nn.BatchNorm2d(ch*2), nn.ReLU(True),
            nn.ConvTranspose2d(ch*2, ch,   4, 2, 1, bias=False), nn.BatchNorm2d(ch),   nn.ReLU(True),
            nn.ConvTranspose2d(ch, 3, 4, 2, 1, bias=False), nn.Tanh(),
        )
    def forward(self, z): return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, ch=CH):
        super().__init__()
        def block(i,o, bn=True):
            mods=[nn.Conv2d(i,o,4,2,1,bias=False), nn.LeakyReLU(0.2, inplace=True)]
            if bn: mods.insert(1, nn.BatchNorm2d(o))
            return nn.Sequential(*mods)
        self.net = nn.Sequential(
            block(3, ch, bn=False),
            block(ch, ch*2),
            block(ch*2, ch*4),
            block(ch*4, ch*8),
            nn.Conv2d(ch*8, 1, 4, 1, 0, bias=False)
        )
    def forward(self, x): return self.net(x).view(-1)

G, D = Generator(), Discriminator()
optG = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5,0.999))
optD = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5,0.999))

# Synthetic real images in [-1,1]
real = torch.randn(16,3,IMG,IMG).clamp(-1,1)
z = torch.randn(16, Z, 1, 1)
fake = G(z).detach()

# Update D
D.train(); G.train()
optD.zero_grad(set_to_none=True)
lossD = (nn.functional.binary_cross_entropy_with_logits(D(real), torch.ones(16)) +
         nn.functional.binary_cross_entropy_with_logits(D(fake), torch.zeros(16)))
lossD.backward(); optD.step()

# Update G
optG.zero_grad(set_to_none=True)
fake2 = G(torch.randn(16, Z, 1, 1))
lossG = nn.functional.binary_cross_entropy_with_logits(D(fake2), torch.ones(16))
lossG.backward(); optG.step()
print('GAN step ok:', float(lossD.item()), float(lossG.item()))`
    }
  );
})();


