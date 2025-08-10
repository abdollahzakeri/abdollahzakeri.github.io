(function(){
  window.registerExample(
    'tensors-math',
    { categoryName: 'Tensors & Math', categorySummary: 'Tensor ops, broadcasting, masking, einsum, stability', topicId: 'logsumexp-trick', topicName: 'LogSumExp trick' },
    {
      id: 'logsumexp-trick',
      name: 'LogSumExp trick',
      tags: ['stability','softmax'],
      meta: 'Stable log-softmax via subtracting max then log-sum-exp',
      description: 'Numerically stable implementation of log-softmax.',
      code: `import torch
def log_softmax(x, dim=-1):
    m = x.max(dim=dim, keepdim=True).values
    y = x - m
    return y - torch.log(torch.exp(y).sum(dim=dim, keepdim=True))

z = torch.randn(2,5) * 10
print(log_softmax(z, -1))`
    }
  );
})();


