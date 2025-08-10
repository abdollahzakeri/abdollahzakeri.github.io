(function(){
  window.registerExample(
    'transformer',
    { categoryName: 'Transformer', categorySummary: 'SDPA, multi-head, masks, encoder/decoder, full model', topicId: 'sdpa', topicName: 'Scaled dot-product attention (SDPA)' },
    {
      id: 'sdpa',
      name: 'Scaled dot-product attention (SDPA)',
      tags: ['attention','sdpa'],
      meta: 'Compute attention weights and context',
      description: 'Manual SDPA implementation computing attention matrix and output.',
      code: `import torch
def sdpa(Q, K, V, mask=None):
    d = Q.size(-1)
    scores = (Q @ K.transpose(-2, -1)) / d**0.5
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    return attn @ V
Q = torch.randn(2,4,8); K = torch.randn(2,4,8); V = torch.randn(2,4,8)
print(sdpa(Q,K,V).shape)`
    }
  );
})();


