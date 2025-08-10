(function(){
  window.registerExample(
    'data',
    { categoryName: 'Data', categorySummary: 'Datasets, DataLoaders, samplers, collation, transforms', topicId: 'collate-pad-seq', topicName: 'Collate: pad variable sequences' },
    {
      id: 'collate-pad-seq',
      name: 'Collate: pad variable sequences',
      tags: ['collate_fn','nlp','padding'],
      meta: 'Custom collate_fn to pad sequences and create masks',
      description: 'Pads variable-length sequences, returns padded tensor, lengths, mask, and labels.',
      code: `import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

def collate_pad(batch):
    seqs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs])
    padded = pad_sequence(seqs, batch_first=True, padding_value=0)
    mask = (padded != 0)
    labels = torch.tensor(labels)
    return padded, lengths, mask, labels

data = [(torch.randint(1, 10, (torch.randint(3, 8, ()).item(),)), int(torch.randint(0,2,()).item())) for _ in range(4)]
loader = DataLoader(data, batch_size=4, collate_fn=collate_pad)
for padded, lengths, mask, labels in loader:
    print(padded.shape, lengths.tolist(), mask.shape, labels.tolist())`
    }
  );
})();


