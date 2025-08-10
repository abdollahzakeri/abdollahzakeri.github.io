(function(){
  window.registerExample(
    'nlp',
    { categoryName: 'NLP', categorySummary: 'Tokenization, RNNs, LSTMs, padding, seq2seq, attention', topicId: 'lstm-classifier', topicName: 'LSTM text classifier' },
    {
      id: 'lstm-classifier',
      name: 'LSTM text classifier',
      tags: ['lstm','classifier'],
      meta: 'Embed -> LSTM -> pooled logits for classification',
      description: 'Minimal text classifier using an LSTM backbone.',
      code: `import torch, torch.nn as nn
class LSTMCls(nn.Module):
    def __init__(self, vocab, dim=64, num_classes=2):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.lstm = nn.LSTM(dim, dim, batch_first=True)
        self.fc = nn.Linear(dim, num_classes)
    def forward(self, x):
        x = self.emb(x)
        x, _ = self.lstm(x)
        pooled = x.mean(dim=1)
        return self.fc(pooled)
print(LSTMCls(1000)(torch.randint(0,1000,(4,12))).shape)`
    }
  );
})();


