(function(){
  window.registerExample(
    'clip',
    { categoryName: 'CLIP & Multimodal', categorySummary: 'Dual encoders, contrastive loss, temperature, zero-shot', topicId: 'infonce-contrastive', topicName: 'InfoNCE contrastive loss' },
    {
      id: 'infonce-contrastive',
      name: 'InfoNCE contrastive loss',
      tags: ['clip','loss'],
      meta: 'Symmetric cross-entropy over similarity matrix',
      description: 'Compute CLIP-style contrastive loss between image/text embeddings.',
      code: `import torch
def clip_loss(z_img, z_txt, temp=0.07):
    z_img = torch.nn.functional.normalize(z_img, dim=-1)
    z_txt = torch.nn.functional.normalize(z_txt, dim=-1)
    logits = z_img @ z_txt.t() / temp
    targets = torch.arange(z_img.size(0), device=z_img.device)
    li = torch.nn.functional.cross_entropy(logits, targets)
    lt = torch.nn.functional.cross_entropy(logits.t(), targets)
    return (li + lt) / 2
print(clip_loss(torch.randn(4,256), torch.randn(4,256)).item())`
    }
  );
})();


