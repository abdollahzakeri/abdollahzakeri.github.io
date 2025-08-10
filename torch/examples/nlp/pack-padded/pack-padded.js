(function(){
  window.registerExample(
    'nlp',
    { categoryName: 'NLP', categorySummary: 'Tokenization, RNNs, LSTMs, padding, seq2seq, attention', topicId: 'pack-padded', topicName: 'Pack padded sequences' },
    {
      id: 'pack-padded',
      name: 'Pack padded sequences',
      tags: ['padding','lstm'],
      meta: 'Use pack_padded_sequence and pad_packed_sequence',
      description: 'Handle variable-length sequences efficiently in an LSTM.',
      code: `import torch, torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

x = torch.randint(0, 100, (3, 5))
lengths = torch.tensor([5, 3, 2])
emb = nn.Embedding(100, 16)
lstm = nn.LSTM(16, 32, batch_first=True)
xe = emb(x)
packed = pack_padded_sequence(xe, lengths, batch_first=True, enforce_sorted=False)
out, _ = lstm(packed)
unpacked, lens = pad_packed_sequence(out, batch_first=True)
print(unpacked.shape, lens)`
    }
  );
})();


