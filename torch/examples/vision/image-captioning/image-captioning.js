(function(){
  window.registerExample(
    'vision',
    { categoryName: 'Vision', categorySummary: 'CNNs, augmentations, transfer learning, segmentation', topicId: 'image-captioning', topicName: 'Image Captioning (CNN encoder + RNN decoder)' },
    {
      id: 'image-captioning',
      name: 'Image Captioning (CNN encoder + RNN decoder)',
      tags: ['captioning','cnn','rnn'],
      meta: 'ResNet encoder to features, LSTM decoder over tokens',
      description: 'A compact image captioning pipeline: CNN encodes image to vector, LSTM decodes captions.',
      code: `import torch, torch.nn as nn
from torchvision.models import resnet18

class EncoderCNN(nn.Module):
    def __init__(self, embed=256):
        super().__init__()
        base = resnet18(weights=None)
        self.backbone = nn.Sequential(*(list(base.children())[:-1]))
        self.fc = nn.Linear(base.fc.in_features, embed)
    def forward(self, x):
        h = self.backbone(x).flatten(1)
        return self.fc(h)

class DecoderRNN(nn.Module):
    def __init__(self, vocab=1000, embed=256, hidden=256):
        super().__init__()
        self.emb = nn.Embedding(vocab, embed)
        self.lstm = nn.LSTM(embed, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, vocab)
    def forward(self, feats, captions):
        x = self.emb(captions)
        x[:,0,:] = feats  # inject visual features as first token representation
        out, _ = self.lstm(x)
        return self.fc(out)

enc = EncoderCNN(); dec = DecoderRNN()
img = torch.randn(2,3,224,224)
caps = torch.randint(0,1000,(2,16))
feats = enc(img)
logits = dec(feats, caps)
print(logits.shape)`
    }
  );
})();


