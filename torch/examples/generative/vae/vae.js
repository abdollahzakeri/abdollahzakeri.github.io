(function(){
  window.registerExample(
    'generative',
    { categoryName: 'Generative', categorySummary: 'VAE, GAN, diffusion step, autoregressive sampling', topicId: 'vae', topicName: 'VAE (tiny)' },
    {
      id: 'vae',
      name: 'VAE (tiny)',
      tags: ['vae','generative'],
      meta: 'Minimal VAE with reparameterization and ELBO',
      description: 'A tiny VAE showing encode, reparameterize, decode, and ELBO loss.',
      code: `import torch, torch.nn as nn
class VAE(nn.Module):
    def __init__(self, d=784, h=256, z=32):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, h), nn.ReLU())
        self.mu = nn.Linear(h, z); self.logvar = nn.Linear(h, z)
        self.dec = nn.Sequential(nn.Linear(z, h), nn.ReLU(), nn.Linear(h, d))
    def reparam(self, mu, logvar):
        eps = torch.randn_like(mu)
        return mu + eps * (0.5*logvar).exp()
    def forward(self, x):
        h = self.enc(x)
        mu, logvar = self.mu(h), self.logvar(h)
        z = self.reparam(mu, logvar)
        return self.dec(z), mu, logvar
def elbo(recon, x, mu, logvar):
    recon_loss = torch.nn.functional.mse_loss(recon, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (recon_loss + kld) / x.size(0)
print('ok')`
    }
  );
})();


