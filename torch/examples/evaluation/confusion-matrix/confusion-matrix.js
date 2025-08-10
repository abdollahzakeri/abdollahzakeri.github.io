(function(){
  window.registerExample(
    'evaluation',
    { categoryName: 'Evaluation', categorySummary: 'Eval/inference loops, metrics, confusion matrix', topicId: 'confusion-matrix', topicName: 'Confusion matrix' },
    {
      id: 'confusion-matrix',
      name: 'Confusion matrix',
      tags: ['metric','matrix'],
      meta: 'Accumulate confusion matrix counts across predictions',
      description: 'Build a confusion matrix for multiclass classification.',
      code: `import torch
num_classes = 4
cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
preds = torch.tensor([0,1,2,2,3])
targets = torch.tensor([0,2,1,2,3])
for t, p in zip(targets, preds):
    cm[t, p] += 1
print(cm)`
    }
  );
})();


