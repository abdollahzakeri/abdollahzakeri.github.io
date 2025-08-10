(function(){
  window.registerExample(
    'transformer',
    { categoryName: 'Transformer', categorySummary: 'SDPA, multi-head, masks, encoder/decoder, full model', topicId: 'sdpa-masked', topicName: 'Scaled Dot-Product Attention (masked/causal)' },
    {
      id: 'sdpa-masked',
      name: 'Scaled Dot-Product Attention with masks and causal option',
      tags: ['attention','sdpa','causal'],
      meta: 'SDPA with boolean mask and causal flag',
      description: 'Implements SDPA that accepts an additive mask and an optional causal flag.',
      code: `import torch
def sdpa(q, k, v, mask=None, causal=False):
    B, T, D = q.shape
    scores = (q @ k.transpose(-2,-1)) / (D ** 0.5)
    if causal:
        causal_mask = torch.tril(torch.ones(T, T, device=q.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask, float('-inf'))
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    return attn @ v
q = torch.randn(2,6,8); k = torch.randn(2,6,8); v = torch.randn(2,6,8)
print(sdpa(q,k,v, causal=True).shape)`
    }
  );
})();


