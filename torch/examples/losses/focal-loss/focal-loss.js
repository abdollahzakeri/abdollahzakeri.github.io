(function(){
  window.registerExample(
    'losses',
    { categoryName: 'Losses', categorySummary: 'Classification/regression and metric learning losses', topicId: 'focal-loss', topicName: 'Focal loss (binary)' },
    {
      id: 'focal-loss',
      name: 'Focal loss (binary)',
      tags: ['classification','imbalance'],
      meta: 'Down-weight easy examples to focus on hard ones',
      description: 'Compute focal loss over binary logits to handle class imbalance.',
      code: `import torch
def focal_loss(logits, targets, gamma=2.0):
    prob = torch.sigmoid(logits)
    pt = torch.where(targets==1, prob, 1-prob)
    w = (1-pt).pow(gamma)
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, targets.float(), weight=w)
print('ok')`
    }
  );
})();


