(function(){
  window.registerExample(
    'clip',
    { categoryName: 'CLIP & Multimodal', categorySummary: 'Dual encoders, contrastive loss, temperature, zero-shot', topicId: 'clip-dual-encoder', topicName: 'Dual encoders (text/image)' },
    {
      id: 'clip-dual-encoder',
      name: 'Dual encoders (text/image)',
      tags: ['clip','multimodal'],
      meta: 'Text and image encoders projecting to shared space',
      description: 'Two small encoders output embeddings for contrastive training.',
      code: `import torch, torch.nn as nn
class TextEncoder(nn.Module):
    def __init__(self, vocab=1000, dim=256):
        super().__init__(); self.emb = nn.Embedding(vocab, dim); self.pool = nn.AdaptiveAvgPool1d(1)
    def forward(self, x):
        e = self.emb(x).transpose(1,2)
        return self.pool(e).squeeze(-1)
class ImageEncoder(nn.Module):
    def __init__(self, dim=256):
        super().__init__(); self.net = nn.Sequential(nn.Conv2d(3,32,3,2,1), nn.ReLU(), nn.Conv2d(32,64,3,2,1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, dim))
    def forward(self, x): return self.net(x)
print('ok')`
    }
  );
})();


