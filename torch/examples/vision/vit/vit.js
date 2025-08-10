(function(){
  window.registerExample(
    'vision',
    { categoryName: 'Vision', categorySummary: 'CNNs, augmentations, transfer learning, segmentation', topicId: 'vit', topicName: 'ViT from scratch with patch embedding and attention' },
    {
      id: 'vit',
      name: 'ViT from scratch with patch embedding and attention',
      tags: ['vit','transformer','vision'],
      meta: 'Patchify, linear embed, add class token, apply encoder blocks',
      description: 'Minimal Vision Transformer with patch embedding and transformer encoder blocks.',
      code: `import torch, torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=8, in_chans=3, embed_dim=64):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        x = self.proj(x)  # (B, E, H/P, W/P)
        B, E, H, W = x.shape
        return x.flatten(2).transpose(1, 2)  # (B, N, E)

class EncoderBlock(nn.Module):
    def __init__(self, dim, heads=4, ff=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim*ff), nn.GELU(), nn.Linear(dim*ff, dim))
    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)[0]
        x = x + self.mlp(self.ln2(x))
        return x

class ViT(nn.Module):
    def __init__(self, img_size=32, patch=8, embed=64, depth=2, heads=4, num_classes=10):
        super().__init__()
        self.patch = PatchEmbed(img_size, patch, 3, embed)
        num_patches = (img_size // patch) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, embed))
        self.pos = nn.Parameter(torch.zeros(1, 1 + num_patches, embed))
        self.blocks = nn.ModuleList([EncoderBlock(embed, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed)
        self.head = nn.Linear(embed, num_classes)
    def forward(self, x):
        B = x.size(0)
        x = self.patch(x)
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos[:, : x.size(1)]
        for blk in self.blocks: x = blk(x)
        x = self.norm(x)
        cls = x[:, 0]
        return self.head(cls)

print(ViT()(torch.randn(2,3,32,32)).shape)`
    }
  );
})();


