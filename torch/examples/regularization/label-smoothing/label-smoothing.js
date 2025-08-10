(function(){
  window.registerExample(
    'regularization',
    { categoryName: 'Regularization', categorySummary: 'Dropout, label smoothing, mixup, cutmix, stochastic depth', topicId: 'label-smoothing', topicName: 'Label smoothing CE' },
    {
      id: 'label-smoothing',
      name: 'Label smoothing CE',
      tags: ['regularization','labels'],
      meta: 'Implement label smoothing cross-entropy',
      description: 'Construct a smoothed target distribution and compute CE loss.',
      code: `import torch
import torch.nn.functional as F

def label_smoothing_ce(logits, targets, smoothing=0.1):
    num_classes = logits.size(-1)
    with torch.no_grad():
        true_dist = torch.zeros_like(logits)
        true_dist.fill_(smoothing / (num_classes - 1))
        true_dist.scatter_(1, targets.unsqueeze(1), 1 - smoothing)
    logp = F.log_softmax(logits, dim=-1)
    return -(true_dist * logp).sum(dim=-1).mean()

logits = torch.randn(4, 5)
targets = torch.randint(0, 5, (4,))
print(label_smoothing_ce(logits, targets).item())`
    }
  );
})();


