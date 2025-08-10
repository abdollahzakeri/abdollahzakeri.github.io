(function(){
  window.registerExample(
    'nlp',
    { categoryName: 'NLP', categorySummary: 'Tokenization, RNNs, LSTMs, padding, seq2seq, attention', topicId: 'simple-rnn-bptt', topicName: 'Simple RNN with BPTT' },
    {
      id: 'simple-rnn-bptt',
      name: 'Simple RNN with Backpropagation Through Time (BPTT)',
      tags: ['rnn','bptt'],
      meta: 'Manually unroll a simple RNN for BPTT',
      description: 'Implements a simple tanh RNN cell and trains it on a toy sequence task with BPTT.',
      code: `import torch, torch.nn as nn

class SimpleRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.Wxh = nn.Linear(input_size, hidden_size)
        self.Whh = nn.Linear(hidden_size, hidden_size)
        self.Why = nn.Linear(hidden_size, 2)
    def forward(self, x, h):
        h = torch.tanh(self.Wxh(x) + self.Whh(h))
        y = self.Why(h)
        return y, h

T = 10
cell = SimpleRNNCell(5, 16)
opt = torch.optim.Adam(cell.parameters(), lr=1e-2)
crit = nn.CrossEntropyLoss()

for step in range(50):
    x = torch.randn(T, 5)
    h = torch.zeros(16)
    logits_seq = []
    for t in range(T):
        logits, h = cell(x[t], h)
        logits_seq.append(logits)
    logits = torch.stack(logits_seq)  # (T, 2)
    target = torch.randint(0, 2, (T,))
    loss = crit(logits, target)
    opt.zero_grad(); loss.backward(); opt.step()
print('done')`
    }
  );
})();


