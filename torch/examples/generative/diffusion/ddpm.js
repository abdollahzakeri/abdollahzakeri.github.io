(function(){
  window.registerExample(
    'generative',
    { categoryName: 'Generative', categorySummary: 'VAE, GAN, diffusion step, autoregressive sampling', topicId: 'ddpm', topicName: 'Diffusion (DDPM) — forward and reverse step' },
    {
      id: 'ddpm',
      name: 'Diffusion (DDPM) — forward and reverse step',
      tags: ['diffusion','ddpm','image-generation'],
      meta: 'Implements q(x_t|x_0) and one reverse denoise step with a UNet stub',
      description: 'Defines noise schedule, forward noising, and a minimal reverse step using a tiny ConvNet as epsilon-theta.',
      code: `import torch, torch.nn as nn

T = 1000
betas = torch.linspace(1e-4, 0.02, T)
alphas = 1 - betas
alpha_bar = torch.cumprod(alphas, dim=0)

def q_sample(x0, t):
    noise = torch.randn_like(x0)
    ab = alpha_bar[t].view(-1, 1, 1, 1)
    return (ab.sqrt() * x0) + ((1 - ab).sqrt() * noise), noise

class TinyUNet(nn.Module):
    def __init__(self, c=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, 64, 3, padding=1), nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.SiLU(),
            nn.Conv2d(64, c, 3, padding=1),
        )
    def forward(self, x, t_embed):
        return self.net(x)

def p_sample(model, xt, t):
    beta_t = betas[t]
    alpha_t = alphas[t]
    ab_t = alpha_bar[t]
    eps_theta = model(xt, None)
    mean = (1 / alpha_t.sqrt()) * (xt - (beta_t / (1 - ab_t).sqrt()) * eps_theta)
    if t == 0:
        return mean
    noise = torch.randn_like(xt)
    return mean + (beta_t.sqrt()) * noise

# Demo one step
x0 = torch.randn(4,3,32,32).clamp(-1,1)
model = TinyUNet(3)
t = torch.tensor([999, 999, 999, 999])
xt, _ = q_sample(x0, t)
x_prev = p_sample(model, xt, 999)
print('xt->x_{t-1}:', x_prev.shape)`
    }
  );
})();


