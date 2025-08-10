(function(){
  window.registerExample(
    'evaluation',
    { categoryName: 'Evaluation', categorySummary: 'Eval/inference loops, metrics, confusion matrix', topicId: 'perplexity', topicName: 'Perplexity for language models' },
    {
      id: 'perplexity',
      name: 'Perplexity for language models',
      tags: ['nlp','metric','perplexity'],
      meta: 'Compute PPL from average cross-entropy',
      description: 'Perplexity is exp of average CrossEntropy loss for LM tasks.',
      code: `import torch
avg_loss = torch.tensor(2.0)  # e.g., mean CE over dataset
perplexity = torch.exp(avg_loss)
print(perplexity.item())`
    }
  );
})();


