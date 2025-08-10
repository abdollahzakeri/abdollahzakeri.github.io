(function(){
  window.registerExample(
    'nlp',
    { categoryName: 'NLP', categorySummary: 'Tokenization, RNNs, LSTMs, padding, seq2seq, attention', topicId: 'lstm-bptt', topicName: 'LSTM with BPTT' },
    {
      id: 'lstm-bptt',
      name: 'Long Short-Term Memory (LSTM) with BPTT',
      tags: ['lstm','bptt'],
      meta: 'Train a small LSTM and backprop through time',
      description: 'Runs an LSTM over a sequence and computes a loss over all steps.',
      code: `import torch, torch.nn as nn

T = 12
model = nn.LSTM(8, 16)
proj = nn.Linear(16, 3)
opt = torch.optim.Adam(list(model.parameters()) + list(proj.parameters()), lr=1e-2)
crit = nn.CrossEntropyLoss()

for _ in range(50):
    x = torch.randn(T, 1, 8)  # (T, B, F)
    h0 = torch.zeros(1, 1, 16)
    c0 = torch.zeros(1, 1, 16)
    out, _ = model(x, (h0, c0))  # (T, B, 16)
    logits = proj(out.squeeze(1))  # (T, 3)
    target = torch.randint(0, 3, (T,))
    loss = crit(logits, target)
    opt.zero_grad(); loss.backward(); opt.step()
print('done')`
    }
  );
})();


