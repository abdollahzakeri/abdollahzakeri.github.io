// PyTorch Interview Cookbook Data
// Structure: categories -> topics -> { id, name, tags, description, meta, code }
// All code snippets are minimal, documented, and pure PyTorch (no external trainers)

window.PYTORCH_COOKBOOK = {
  categories: [
    {
      id: 'data',
      name: 'Data',
      summary: 'Datasets, DataLoaders, samplers, collation, transforms',
      topics: [
        {
          id: 'custom-dataset-map',
          name: 'Custom Dataset (map-style)',
          tags: ['dataset', 'map-style', 'data'],
          meta: 'Lines: ~25 — Difficulty: Easy',
          description: 'Define a map-style `Dataset` with indexing and length, returning tensors.',
          code: `import torch
from torch.utils.data import Dataset

class ToyMapDataset(Dataset):
    """Map-style dataset returning (x, y) pairs as tensors."""
    def __init__(self, data, targets):
        self.data = torch.as_tensor(data, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.long)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        return x, y

# usage
X = torch.randn(100, 10)
y = torch.randint(0, 3, (100,))
ds = ToyMapDataset(X, y)
print(len(ds), ds[0][0].shape, ds[0][1])`
        },
        {
          id: 'iterable-dataset',
          name: 'Iterable Dataset',
          tags: ['dataset', 'iterable', 'streaming'],
          meta: 'Lines: ~25 — Difficulty: Easy',
          description: 'Stream samples without random access; useful for large or generated data.',
          code: `import torch
from torch.utils.data import IterableDataset

class ToyIterableDataset(IterableDataset):
    """Yields (x, y) pairs on the fly."""
    def __init__(self, num_samples=100):
        self.num_samples = num_samples

    def __iter__(self):
        for i in range(self.num_samples):
            x = torch.randn(10)
            y = torch.randint(0, 2, ()).item()
            yield x, y

# usage
ds = ToyIterableDataset(5)
for x, y in ds:
    print(x.shape, y)`
        },
        {
          id: 'dataloader-basic',
          name: 'DataLoader (basic)',
          tags: ['dataloader', 'batching'],
          meta: 'Lines: ~20 — Difficulty: Easy',
          description: 'Batch a map-style `Dataset` using `DataLoader` with shuffling and workers.',
          code: `import torch
from torch.utils.data import DataLoader, TensorDataset

X = torch.randn(100, 10)
y = torch.randint(0, 3, (100,))
ds = TensorDataset(X, y)
loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=0)

for xb, yb in loader:
    print(xb.shape, yb.shape)
    break`
        },
        {
          id: 'collate-pad-seq',
          name: 'Collate: pad variable sequences',
          tags: ['collate_fn', 'nlp', 'padding'],
          meta: 'Lines: ~35 — Difficulty: Medium',
          description: 'Custom `collate_fn` to pad variable-length sequences and create masks.',
          code: `import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

def collate_pad(batch):
    # batch: list of (seq_tensor, label)
    seqs, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs])
    padded = pad_sequence(seqs, batch_first=True, padding_value=0)
    mask = (padded != 0)
    labels = torch.tensor(labels)
    return padded, lengths, mask, labels

data = [(torch.randint(1, 10, (torch.randint(3, 8, ()).item(),)), 1) for _ in range(4)]
loader = DataLoader(data, batch_size=4, collate_fn=collate_pad)

for padded, lengths, mask, labels in loader:
    print(padded.shape, lengths, mask.shape, labels.shape)`
        },
        {
          id: 'imagefolder-transforms',
          name: 'ImageFolder + transforms',
          tags: ['vision', 'transforms', 'torchvision'],
          meta: 'Lines: ~25 — Difficulty: Easy',
          description: 'Use `ImageFolder` with common augmentations and normalization.',
          code: `from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_ds = datasets.ImageFolder('/path/to/train', transform=transform)
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
print(len(train_ds), next(iter(train_loader))[0].shape)`
        },
        {
          id: 'weighted-sampler',
          name: 'WeightedRandomSampler',
          tags: ['sampler', 'class-imbalance'],
          meta: 'Lines: ~30 — Difficulty: Medium',
          description: 'Sample to address class imbalance by weighting examples.',
          code: `import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

X = torch.randn(8, 4)
y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
class_sample_count = torch.tensor([(y == 0).sum(), (y == 1).sum()], dtype=torch.float32)
weights = 1.0 / class_sample_count
sample_weights = weights[y]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

loader = DataLoader(TensorDataset(X, y), batch_size=4, sampler=sampler)
for xb, yb in loader:
    print(yb)`
        },
        {
          id: 'balanced-batch-sampler',
          name: 'Balanced batch sampler (labels)',
          tags: ['sampler', 'batching'],
          meta: 'Lines: ~40 — Difficulty: Medium',
          description: 'Create batches with equal counts from each class.',
          code: `import torch
from torch.utils.data import Sampler

class BalancedBatchSampler(Sampler):
    def __init__(self, targets, batch_size):
        self.targets = torch.as_tensor(targets)
        self.batch_size = batch_size
        self.classes = self.targets.unique()
        self.per_class = batch_size // len(self.classes)
        self.indices_by_class = {c.item(): (self.targets == c).nonzero(as_tuple=True)[0].tolist() for c in self.classes}

    def __iter__(self):
        import random
        indices = []
        pool = {c: lst[:] for c, lst in self.indices_by_class.items()}
        while all(len(v) >= self.per_class for v in pool.values()):
            batch = []
            for c in self.classes.tolist():
                random.shuffle(pool[c])
                batch += pool[c][:self.per_class]
                pool[c] = pool[c][self.per_class:]
            random.shuffle(batch)
            if len(batch) == self.batch_size:
                indices += batch
            else:
                break
        return iter(indices)

    def __len__(self):
        total = sum(len(v) for v in self.indices_by_class.values())
        return total // self.batch_size * self.batch_size

# usage
targets = [0,0,0,1,1,1,1,0,1,0,1,0]
sampler = BalancedBatchSampler(targets, batch_size=4)
print(len(list(iter(sampler))))`
        },
        {
          id: 'text-dataset-tokenize',
          name: 'Text dataset + tokenizer (basic)',
          tags: ['nlp', 'dataset', 'tokenizer'],
          meta: 'Lines: ~35 — Difficulty: Easy',
          description: 'Whitespace tokenizer with vocab and numericalization.',
          code: `import torch
from torch.utils.data import Dataset

class SimpleVocab:
    def __init__(self, texts, min_freq=1):
        from collections import Counter
        counter = Counter(w for t in texts for w in t.split())
        self.itos = ['<pad>', '<unk>'] + [w for w,c in counter.items() if c >= min_freq]
        self.stoi = {w:i for i,w in enumerate(self.itos)}

    def encode(self, text):
        return [self.stoi.get(w, 1) for w in text.split()]

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.vocab = SimpleVocab(texts)
        self.samples = [(torch.tensor(self.vocab.encode(t)), int(l)) for t,l in zip(texts, labels)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

texts = ["hello world", "deep learning with pytorch", "hello pytorch"]
labels = [0,1,0]
ds = TextDataset(texts, labels)
print(ds[0])`
        },
        {
          id: 'csv-streaming',
          name: 'CSV streaming (iterable)',
          tags: ['csv', 'iterable', 'io'],
          meta: 'Lines: ~30 — Difficulty: Easy',
          description: 'Stream rows from a CSV file as tensors without loading into memory.',
          code: `import csv
import torch
from torch.utils.data import IterableDataset

class CSVStream(IterableDataset):
    def __init__(self, path):
        self.path = path

    def __iter__(self):
        with open(self.path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                x = torch.tensor([float(v) for v in row[:-1]])
                y = torch.tensor(int(row[-1]))
                yield x, y

# for row in CSVStream('data.csv'): print(row)`
        },
      ],
    },

    {
      id: 'training',
      name: 'Training',
      summary: 'Loops, AMP, clipping, accumulation, checkpoints',
      topics: [
        {
          id: 'basic-training-loop',
          name: 'Basic training loop',
          tags: ['loop', 'optimizer', 'loss'],
          meta: 'Lines: ~45 — Difficulty: Easy',
          description: 'Canonical supervised training loop with cross-entropy.',
          code: `import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 3))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

X = torch.randn(200, 10)
y = torch.randint(0, 3, (200,))
loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

model.train()
for epoch in range(3):
    running_loss = 0.0
    for xb, yb in loader:
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
    print(f"epoch {epoch}: loss={running_loss/len(loader.dataset):.4f}")`
        },
        {
          id: 'grad-clipping',
          name: 'Gradient clipping',
          tags: ['stability', 'clip', 'norm'],
          meta: 'Lines: ~25 — Difficulty: Easy',
          description: 'Clip gradients by norm to stabilize training.',
          code: `import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

model = nn.Linear(10, 2)
loss = model(torch.randn(4,10)).sum()
loss.backward()
clip_grad_norm_(model.parameters(), max_norm=1.0)
for p in model.parameters():
    if p.grad is not None:
        print(p.grad.norm())`
        },
        {
          id: 'amp-autocast-gradscaler',
          name: 'AMP (autocast + GradScaler)',
          tags: ['amp', 'mixed-precision', 'fp16'],
          meta: 'Lines: ~45 — Difficulty: Medium',
          description: 'Use autocast and GradScaler for mixed-precision speedups on CUDA.',
          code: `import torch
import torch.nn as nn
import torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = nn.Linear(128, 10).to(device)
opt = optim.AdamW(model.parameters(), lr=1e-3)
scaler = torch.cuda.amp.GradScaler(enabled=(device=='cuda'))

x = torch.randn(64, 128, device=device)
y = torch.randint(0, 10, (64,), device=device)
crit = nn.CrossEntropyLoss()

opt.zero_grad(set_to_none=True)
with torch.cuda.amp.autocast(enabled=(device=='cuda')):
    logits = model(x)
    loss = crit(logits, y)
scaler.scale(loss).backward()
scaler.step(opt)
scaler.update()`
        },
        {
          id: 'grad-accumulation',
          name: 'Gradient accumulation',
          tags: ['large-batch', 'accumulate'],
          meta: 'Lines: ~40 — Difficulty: Medium',
          description: 'Accumulate gradients over micro-batches to simulate large batch size.',
          code: `import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 2))
opt = optim.AdamW(model.parameters(), lr=1e-3)
crit = nn.CrossEntropyLoss()

micro_batches = 4
opt.zero_grad(set_to_none=True)
for step in range(8):
    x = torch.randn(16, 10)
    y = torch.randint(0, 2, (16,))
    logits = model(x)
    loss = crit(logits, y) / micro_batches
    loss.backward()
    if (step + 1) % micro_batches == 0:
        opt.step(); opt.zero_grad(set_to_none=True)`
        },
        {
          id: 'checkpoint-save-load',
          name: 'Checkpoint: save/load',
          tags: ['checkpoint', 'state_dict'],
          meta: 'Lines: ~35 — Difficulty: Easy',
          description: 'Save and resume training with model and optimizer states.',
          code: `import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Linear(10, 3)
opt = optim.AdamW(model.parameters(), lr=1e-3)
state = {'epoch': 5, 'model': model.state_dict(), 'opt': opt.state_dict()}
torch.save(state, 'ckpt.pt')

# load
ckpt = torch.load('ckpt.pt', map_location='cpu')
model.load_state_dict(ckpt['model'])
opt.load_state_dict(ckpt['opt'])
print('resumed epoch', ckpt['epoch'])`
        },
        {
          id: 'early-stopping',
          name: 'Early stopping',
          tags: ['validation', 'stopping'],
          meta: 'Lines: ~45 — Difficulty: Medium',
          description: 'Stop training when validation metric stops improving.',
          code: `best = float('inf')
patience, bad = 3, 0
for epoch in range(50):
    train_loss = 1.0 / (epoch + 1)
    val_loss = train_loss + 0.05 * (epoch > 5)
    if val_loss < best:
        best, bad = val_loss, 0
        # save best
    else:
        bad += 1
        if bad >= patience:
            print('early stop at epoch', epoch)
            break`
        },
        {
          id: 'seed-reproducibility',
          name: 'Seed & reproducibility',
          tags: ['seed', 'deterministic'],
          meta: 'Lines: ~25 — Difficulty: Easy',
          description: 'Set seeds and deterministic flags for reproducible results.',
          code: `import torch, random, numpy as np
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(1337)`
        },
        {
          id: 'torch-compile',
          name: 'torch.compile() (PyTorch 2)',
          tags: ['compile', 'performance'],
          meta: 'Lines: ~25 — Difficulty: Easy',
          description: 'JIT compile model forward for speed in PyTorch 2.x.',
          code: `import torch, torch.nn as nn

class Small(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10))
    def forward(self, x):
        return self.net(x)

m = Small()
m = torch.compile(m)  # requires PyTorch 2+
print(m(torch.randn(2,64)).shape)`
        },
      ],
    },

    {
      id: 'evaluation',
      name: 'Evaluation',
      summary: 'Eval/inference loops, metrics, confusion matrix',
      topics: [
        {
          id: 'eval-loop',
          name: 'Evaluation loop (no_grad)',
          tags: ['eval', 'no_grad'],
          meta: 'Lines: ~30 — Difficulty: Easy',
          description: 'Switch to eval mode and disable grads for evaluation.',
          code: `import torch
import torch.nn as nn
model = nn.Linear(10, 2)
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for _ in range(10):
        x = torch.randn(8,10)
        y = torch.randint(0,2,(8,))
        logits = model(x)
        pred = logits.argmax(dim=-1)
        correct += (pred==y).sum().item(); total += y.numel()
print('acc:', correct/total)`
        },
        {
          id: 'accuracy-metric',
          name: 'Accuracy (running)',
          tags: ['metric', 'classification'],
          meta: 'Lines: ~25 — Difficulty: Easy',
          description: 'Compute running accuracy across batches.',
          code: `import torch
correct, total = 0, 0
for _ in range(3):
    y = torch.randint(0, 3, (5,))
    p = torch.randint(0, 3, (5,))
    correct += (y == p).sum().item(); total += y.numel()
print('acc=', correct/total)`
        },
        {
          id: 'topk-accuracy',
          name: 'Top-k accuracy',
          tags: ['metric', 'topk'],
          meta: 'Lines: ~30 — Difficulty: Easy',
          description: 'Compute top-k accuracy for multiclass classification.',
          code: `import torch
logits = torch.randn(4, 10)
targets = torch.randint(0, 10, (4,))
topk = 3
_, pred = logits.topk(topk, dim=-1)
correct = (pred == targets.unsqueeze(1)).any(dim=1).float().mean().item()
print('top-3 acc=', correct)`
        },
        {
          id: 'confusion-matrix',
          name: 'Confusion matrix',
          tags: ['metric', 'matrix'],
          meta: 'Lines: ~35 — Difficulty: Medium',
          description: 'Accumulate confusion matrix for multiclass classification.',
          code: `import torch
num_classes = 4
cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
preds = torch.tensor([0,1,2,2,3])
targets = torch.tensor([0,2,1,2,3])
for t, p in zip(targets, preds):
    cm[t, p] += 1
print(cm)`
        },
        {
          id: 'precision-recall-f1',
          name: 'Precision / Recall / F1',
          tags: ['metric', 'prf1'],
          meta: 'Lines: ~40 — Difficulty: Medium',
          description: 'Compute micro-averaged precision, recall, and F1.',
          code: `import torch
num_classes = 3
preds = torch.tensor([0,1,1,2,0,2])
targets = torch.tensor([0,1,2,2,0,1])
cm = torch.zeros(num_classes, num_classes)
for t, p in zip(targets, preds):
    cm[t, p] += 1
tp = cm.diag()
fp = cm.sum(0) - tp
fn = cm.sum(1) - tp
precision = (tp.sum() / (tp.sum() + fp.sum()).clamp(min=1))
recall = (tp.sum() / (tp.sum() + fn.sum()).clamp(min=1))
f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)
print(precision.item(), recall.item(), f1.item())`
        },
        {
          id: 'perplexity',
          name: 'Perplexity (LM)',
          tags: ['nlp', 'metric'],
          meta: 'Lines: ~25 — Difficulty: Easy',
          description: 'Compute perplexity from average cross-entropy.',
          code: `import torch
avg_loss = torch.tensor(2.0)  # example CE loss
perplexity = torch.exp(avg_loss)
print(perplexity.item())`
        },
      ],
    },

    {
      id: 'tensors-math',
      name: 'Tensors & Math',
      summary: 'Tensor ops, broadcasting, masking, einsum, stability',
      topics: [
        {
          id: 'tensor-create-dtypes',
          name: 'Tensor creation & dtypes',
          tags: ['tensor', 'dtype', 'device'],
          meta: 'Lines: ~25 — Difficulty: Easy',
          description: 'Create tensors on specific devices with dtypes.',
          code: `import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
a = torch.zeros((2,3), dtype=torch.float32, device=device)
b = torch.arange(6, dtype=torch.int64).reshape(2,3).to(device)
print(a.dtype, b.dtype, a.device)`
        },
        {
          id: 'broadcasting',
          name: 'Broadcasting',
          tags: ['broadcast', 'shape'],
          meta: 'Lines: ~20 — Difficulty: Easy',
          description: 'Leverage broadcasting to add per-feature bias.',
          code: `import torch
x = torch.randn(4, 3)
bias = torch.tensor([0.5, 0.0, -0.5])
y = x + bias  # (4,3) + (3,) -> (4,3)
print(y.shape)`
        },
        {
          id: 'boolean-masking',
          name: 'Boolean masking',
          tags: ['mask', 'indexing'],
          meta: 'Lines: ~20 — Difficulty: Easy',
          description: 'Select or fill values using boolean masks.',
          code: `import torch
x = torch.tensor([1, -2, 3, -4])
mask = x > 0
pos = x[mask]
x[~mask] = 0
print(pos, x)`
        },
        {
          id: 'logsumexp-trick',
          name: 'LogSumExp trick',
          tags: ['stability', 'softmax'],
          meta: 'Lines: ~25 — Difficulty: Medium',
          description: 'Stable log-softmax via LogSumExp.',
          code: `import torch
def log_softmax(x, dim=-1):
    m = x.max(dim=dim, keepdim=True).values
    y = x - m
    return y - torch.log(torch.exp(y).sum(dim=dim, keepdim=True))

z = torch.randn(2,5) * 10
print(log_softmax(z, -1))`
        },
        {
          id: 'einsum-attn-scores',
          name: 'einsum: attention scores',
          tags: ['einsum', 'attention'],
          meta: 'Lines: ~25 — Difficulty: Medium',
          description: 'Compute QK^T via `einsum` for attention scores.',
          code: `import torch
B, T, D = 2, 4, 8
Q = torch.randn(B, T, D)
K = torch.randn(B, T, D)
scores = torch.einsum('btd,bSd->bts', Q, K)  # (B,T,T)
print(scores.shape)`
        },
        {
          id: 'im2col-unfold',
          name: 'im2col via unfold',
          tags: ['conv', 'unfold'],
          meta: 'Lines: ~35 — Difficulty: Medium',
          description: 'Use `unfold` to extract sliding windows (im2col).',
          code: `import torch
x = torch.arange(1*1*4*4).float().reshape(1,1,4,4)
patches = x.unfold(2, 3, 1).unfold(3, 3, 1)  # (N,C,H-k+1,W-k+1,k,k)
cols = patches.contiguous().view(1, 1, -1, 3*3)
print(cols.shape)`
        },
        {
          id: 'manual-cross-entropy',
          name: 'Manual cross-entropy (stable)',
          tags: ['loss', 'stability'],
          meta: 'Lines: ~35 — Difficulty: Medium',
          description: 'Compute CE loss from logits using log-softmax.',
          code: `import torch
def cross_entropy(logits, target):
    logp = torch.log_softmax(logits, dim=-1)
    return -logp[torch.arange(target.size(0)), target].mean()

logits = torch.randn(4, 5) * 3
target = torch.randint(0, 5, (4,))
print(cross_entropy(logits, target))`
        },
      ],
    },

    {
      id: 'layers-modules',
      name: 'Layers & Modules',
      summary: 'Implementations of common layers from scratch',
      topics: [
        {
          id: 'linear-from-scratch',
          name: 'Linear layer (from scratch)',
          tags: ['modules', 'linear'],
          meta: 'Lines: ~40 — Difficulty: Medium',
          description: 'Minimal linear layer with parameters and forward.',
          code: `import torch
import torch.nn as nn

class MyLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        y = x @ self.weight.t()
        if self.bias is not None:
            y = y + self.bias
        return y

import math
layer = MyLinear(4, 3)
print(layer(torch.randn(2,4)).shape)`
        },
        {
          id: 'layernorm',
          name: 'LayerNorm (from scratch)',
          tags: ['normalization', 'layernorm'],
          meta: 'Lines: ~45 — Difficulty: Medium',
          description: 'Implement LayerNorm across last dimension.',
          code: `import torch
import torch.nn as nn

class MyLayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        xhat = (x - mean) / torch.sqrt(var + self.eps)
        return xhat * self.weight + self.bias

x = torch.randn(2, 5, 8)
ln = MyLayerNorm(8)
print(ln(x).shape)`
        },
        {
          id: 'batchnorm1d',
          name: 'BatchNorm1d (training mode)',
          tags: ['normalization', 'batchnorm'],
          meta: 'Lines: ~55 — Difficulty: Medium',
          description: 'Implement BatchNorm1d in training mode with running stats.',
          code: `import torch
import torch.nn as nn

class MyBatchNorm1d(nn.Module):
    def __init__(self, num_features, momentum=0.1, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.momentum, self.eps = momentum, eps

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            self.running_mean.lerp_(mean, self.momentum)
            self.running_var.lerp_(var, self.momentum)
        else:
            mean, var = self.running_mean, self.running_var
        xhat = (x - mean) / torch.sqrt(var + self.eps)
        return xhat * self.weight + self.bias

bn = MyBatchNorm1d(4)
print(bn(torch.randn(8,4)).shape)`
        },
        {
          id: 'positional-encoding',
          name: 'Sinusoidal positional encoding',
          tags: ['transformer', 'positional'],
          meta: 'Lines: ~40 — Difficulty: Easy',
          description: 'Compute sinusoidal positional encodings for sequences.',
          code: `import torch, math
def sinusoidal_positional_encoding(T, D):
    pe = torch.zeros(T, D)
    position = torch.arange(0, T, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, D, 2).float() * (-math.log(10000.0) / D))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

print(sinusoidal_positional_encoding(5, 8).shape)`
        },
        {
          id: 'residual-block',
          name: 'Residual block',
          tags: ['residual', 'skip-connection'],
          meta: 'Lines: ~35 — Difficulty: Easy',
          description: 'A basic residual MLP block.',
          code: `import torch
import torch.nn as nn

class Residual(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
    def forward(self, x):
        return x + self.net(x)

print(Residual(16)(torch.randn(2,16)).shape)`
        },
        {
          id: 'embedding-freeze',
          name: 'Embedding + freeze',
          tags: ['embedding', 'nlp'],
          meta: 'Lines: ~30 — Difficulty: Easy',
          description: 'Use an `Embedding` layer and freeze its weights.',
          code: `import torch
import torch.nn as nn
emb = nn.Embedding(1000, 64)
emb.weight.requires_grad = False
idx = torch.randint(0, 1000, (4, 10))
print(emb(idx).shape)`
        },
      ],
    },

    {
      id: 'initialization',
      name: 'Initialization',
      summary: 'Xavier, Kaiming, custom rules',
      topics: [
        {
          id: 'xavier-init',
          name: 'Xavier/Glorot init',
          tags: ['init', 'xavier'],
          meta: 'Lines: ~20 — Difficulty: Easy',
          description: 'Apply Xavier uniform initialization to linear layers.',
          code: `import torch.nn as nn
def init_xavier(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

net = nn.Sequential(nn.Linear(8,16), nn.ReLU(), nn.Linear(16,4))
net.apply(init_xavier)`
        },
        {
          id: 'kaiming-init',
          name: 'Kaiming/He init',
          tags: ['init', 'kaiming'],
          meta: 'Lines: ~20 — Difficulty: Easy',
          description: 'Kaiming normal init for ReLU networks.',
          code: `import torch.nn as nn
def init_kaiming(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

net = nn.Sequential(nn.Linear(8,16), nn.ReLU(), nn.Linear(16,4))
net.apply(init_kaiming)`
        },
        {
          id: 'per-name-init',
          name: 'Init by parameter name',
          tags: ['init', 'naming'],
          meta: 'Lines: ~30 — Difficulty: Easy',
          description: 'Initialize parameters by matching names.',
          code: `import torch, torch.nn as nn
net = nn.Sequential(nn.Linear(8,16), nn.ReLU(), nn.Linear(16,4))
for name, p in net.named_parameters():
    if 'weight' in name: nn.init.xavier_uniform_(p)
    if 'bias'   in name: nn.init.zeros_(p)`
        },
      ],
    },

    {
      id: 'optimization',
      name: 'Optimization',
      summary: 'Optimizers, schedulers, param groups, freezing',
      topics: [
        {
          id: 'sgd-momentum',
          name: 'SGD + momentum',
          tags: ['optimizer', 'sgd'],
          meta: 'Lines: ~20 — Difficulty: Easy',
          description: 'Stochastic gradient descent with momentum.',
          code: `import torch
import torch.optim as optim
params = [torch.randn(3, requires_grad=True)]
opt = optim.SGD(params, lr=1e-2, momentum=0.9)`
        },
        {
          id: 'adamw',
          name: 'AdamW',
          tags: ['optimizer', 'adamw'],
          meta: 'Lines: ~20 — Difficulty: Easy',
          description: 'AdamW with weight decay decoupled.',
          code: `import torch.nn as nn, torch.optim as optim
net = nn.Linear(8, 2)
opt = optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.01)`
        },
        {
          id: 'steplr',
          name: 'StepLR scheduler',
          tags: ['scheduler'],
          meta: 'Lines: ~25 — Difficulty: Easy',
          description: 'Decay LR by gamma every step_size epochs.',
          code: `import torch.optim as optim
opt = optim.SGD([{'params': []}], lr=0.1)
sch = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.1)`
        },
        {
          id: 'cosine-warm-restarts',
          name: 'CosineAnnealingWarmRestarts',
          tags: ['scheduler', 'cosine'],
          meta: 'Lines: ~25 — Difficulty: Easy',
          description: 'Cosine schedule with restarts.',
          code: `import torch.optim as optim
opt = optim.AdamW([{'params': []}], lr=3e-4)
sch = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2)`
        },
        {
          id: 'onecycle',
          name: 'OneCycleLR',
          tags: ['scheduler', 'onecycle'],
          meta: 'Lines: ~30 — Difficulty: Medium',
          description: 'OneCycle learning rate schedule.',
          code: `import torch
import torch.optim as optim
model = torch.nn.Linear(10, 2)
opt = optim.SGD(model.parameters(), lr=0.1)
sch = optim.lr_scheduler.OneCycleLR(opt, max_lr=0.5, steps_per_epoch=10, epochs=5)`
        },
        {
          id: 'param-groups',
          name: 'Parameter groups (LR/WD)',
          tags: ['optimizer', 'param-groups'],
          meta: 'Lines: ~35 — Difficulty: Medium',
          description: 'Different hyperparams per parameter group (e.g., no decay for bias/LayerNorm).',
          code: `import torch.nn as nn, torch.optim as optim
model = nn.Sequential(nn.Linear(10, 10), nn.LayerNorm(10), nn.Linear(10, 2))
decay, no_decay = [], []
for name, p in model.named_parameters():
    if p.requires_grad:
        (no_decay if name.endswith('bias') or 'LayerNorm' in name else decay).append(p)
opt = optim.AdamW([
    {'params': decay, 'weight_decay': 0.01},
    {'params': no_decay, 'weight_decay': 0.0},
], lr=3e-4)`
        },
        {
          id: 'freeze-unfreeze',
          name: 'Freeze / unfreeze parameters',
          tags: ['fine-tune', 'transfer'],
          meta: 'Lines: ~25 — Difficulty: Easy',
          description: 'Freeze backbone, train head, then unfreeze.',
          code: `import torch.nn as nn
backbone = nn.Sequential(nn.Linear(100,64), nn.ReLU())
head = nn.Linear(64, 10)
for p in backbone.parameters(): p.requires_grad = False
model = nn.Sequential(backbone, head)`
        },
      ],
    },

    {
      id: 'regularization',
      name: 'Regularization',
      summary: 'Dropout, label smoothing, mixup, cutmix, stochastic depth',
      topics: [
        {
          id: 'label-smoothing',
          name: 'Label smoothing CE',
          tags: ['regularization', 'labels'],
          meta: 'Lines: ~40 — Difficulty: Medium',
          description: 'Cross entropy with label smoothing.',
          code: `import torch
import torch.nn.functional as F

def label_smoothing_ce(logits, targets, smoothing=0.1):
    num_classes = logits.size(-1)
    with torch.no_grad():
        true_dist = torch.zeros_like(logits)
        true_dist.fill_(smoothing / (num_classes - 1))
        true_dist.scatter_(1, targets.unsqueeze(1), 1 - smoothing)
    logp = F.log_softmax(logits, dim=-1)
    return -(true_dist * logp).sum(dim=-1).mean()`
        },
        {
          id: 'mixup',
          name: 'Mixup (images)',
          tags: ['augmentation', 'mixup'],
          meta: 'Lines: ~40 — Difficulty: Medium',
          description: 'Linear interpolate inputs and targets with lambda ~ Beta.',
          code: `import torch
def mixup(x, y, alpha=0.4):
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    idx = torch.randperm(x.size(0))
    x_mix = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return x_mix, y_a, y_b, lam

# usage: loss = lam*CE(logits,y_a) + (1-lam)*CE(logits,y_b)`
        },
        {
          id: 'cutmix',
          name: 'CutMix (images)',
          tags: ['augmentation', 'cutmix'],
          meta: 'Lines: ~55 — Difficulty: Medium',
          description: 'Cut and paste a random patch from another image.',
          code: `import torch
def rand_bbox(W, H, lam):
    import random
    cut_rat = torch.sqrt(1. - lam)
    cut_w, cut_h = (W * cut_rat).int(), (H * cut_rat).int()
    cx, cy = random.randint(0, W), random.randint(0, H)
    x1, y1 = torch.clamp(cx - cut_w // 2, 0, W), torch.clamp(cy - cut_h // 2, 0, H)
    x2, y2 = torch.clamp(cx + cut_w // 2, 0, W), torch.clamp(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2

def cutmix(x, y, alpha=1.0):
    B, C, H, W = x.size()
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    idx = torch.randperm(B)
    x1, y1, x2, y2 = rand_bbox(W, H, lam)
    x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    y_a, y_b = y, y[idx]
    return x, y_a, y_b, lam`
        },
        {
          id: 'stochastic-depth',
          name: 'Stochastic depth (DropPath)',
          tags: ['regularization', 'droppath'],
          meta: 'Lines: ~40 — Difficulty: Medium',
          description: 'Randomly drop residual branch at train time.',
          code: `import torch
import torch.nn as nn

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.2):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        if not self.training or self.drop_prob == 0.: return x
        keep = 1 - self.drop_prob
        mask = torch.empty((x.size(0),) + (1,) * (x.ndim - 1), dtype=x.dtype, device=x.device).bernoulli_(keep)
        return x / keep * mask`
        },
      ],
    },

    {
      id: 'autograd',
      name: 'Autograd',
      summary: 'Custom Functions, hooks, detach, checkpointing',
      topics: [
        {
          id: 'custom-function',
          name: 'Custom autograd Function',
          tags: ['autograd', 'custom'],
          meta: 'Lines: ~45 — Difficulty: Medium',
          description: 'Define a custom forward/backward for x^2.',
          code: `import torch
from torch.autograd import Function

class Square(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * x
    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        return 2 * x * grad_out

x = torch.tensor([3.0], requires_grad=True)
y = Square.apply(x)
y.backward(torch.ones_like(y))
print(x.grad)`
        },
        {
          id: 'detach-stop-grad',
          name: 'Detach (stop gradient)',
          tags: ['autograd', 'detach'],
          meta: 'Lines: ~20 — Difficulty: Easy',
          description: 'Prevent gradients from flowing through a tensor.',
          code: `import torch
x = torch.tensor(2.0, requires_grad=True)
y = (x * 3).detach()
z = y * 4
z.backward()
print(x.grad)  # None`}
      ,
        {
          id: 'register-hook',
          name: 'Register grad hook',
          tags: ['autograd', 'hook'],
          meta: 'Lines: ~30 — Difficulty: Medium',
          description: 'Inspect/modify gradients with hooks.',
          code: `import torch
x = torch.tensor([1.,2.,3.], requires_grad=True)
def hook(grad):
    print('grad:', grad)
    return grad
x.register_hook(hook)
y = (x * x).sum()
y.backward()`
        },
        {
          id: 'grad-checkpointing',
          name: 'Gradient checkpointing',
          tags: ['memory', 'checkpoint'],
          meta: 'Lines: ~35 — Difficulty: Medium',
          description: 'Trade compute for memory by checkpointing.',
          code: `import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

block = nn.Sequential(nn.Linear(128,128), nn.ReLU(), nn.Linear(128,128))
x = torch.randn(32, 128, requires_grad=True)
out = checkpoint(block, x)
out.sum().backward()`
        },
      ],
    },

    {
      id: 'serialization',
      name: 'Serialization',
      summary: 'Save/load checkpoints, TorchScript, ONNX',
      topics: [
        {
          id: 'save-load-model',
          name: 'Save/Load model state_dict',
          tags: ['io', 'state_dict'],
          meta: 'Lines: ~25 — Difficulty: Easy',
          description: 'Persist model weights to disk and reload.',
          code: `import torch, torch.nn as nn
m = nn.Linear(10, 2)
torch.save(m.state_dict(), 'm.pt')
m2 = nn.Linear(10, 2)
m2.load_state_dict(torch.load('m.pt', map_location='cpu'))`
        },
        {
          id: 'save-load-opt-sched',
          name: 'Save/Load optimizer & scheduler',
          tags: ['io', 'optimizer'],
          meta: 'Lines: ~30 — Difficulty: Easy',
          description: 'Resume optimizer and scheduler states.',
          code: `import torch, torch.optim as optim, torch.nn as nn
m = nn.Linear(10, 2)
opt = optim.AdamW(m.parameters())
sch = optim.lr_scheduler.StepLR(opt, 10)
torch.save({'opt': opt.state_dict(), 'sch': sch.state_dict()}, 'opt.pt')
state = torch.load('opt.pt')
opt.load_state_dict(state['opt'])
sch.load_state_dict(state['sch'])`
        },
        {
          id: 'torchscript-trace',
          name: 'TorchScript: trace',
          tags: ['torchscript', 'trace'],
          meta: 'Lines: ~30 — Difficulty: Medium',
          description: 'Export a traced module for inference.',
          code: `import torch, torch.nn as nn
m = nn.Linear(10, 2)
tm = torch.jit.trace(m, torch.randn(1,10))
tm.save('m_traced.pt')`
        },
        {
          id: 'torchscript-script',
          name: 'TorchScript: script',
          tags: ['torchscript', 'script'],
          meta: 'Lines: ~35 — Difficulty: Medium',
          description: 'Script a module with control flow.',
          code: `import torch, torch.nn as nn
class M(nn.Module):
    def forward(self, x):
        if x.sum() > 0: return x * 2
        else: return -x
sm = torch.jit.script(M())
sm.save('m_scripted.pt')`
        },
      ],
    },

    {
      id: 'nlp',
      name: 'NLP',
      summary: 'Tokenization, RNNs, LSTMs, padding, seq2seq, attention',
      topics: [
        {
          id: 'lstm-classifier',
          name: 'LSTM text classifier',
          tags: ['lstm', 'classifier'],
          meta: 'Lines: ~55 — Difficulty: Medium',
          description: 'Embed -> LSTM -> pooled logits for classification.',
          code: `import torch, torch.nn as nn
class LSTMCls(nn.Module):
    def __init__(self, vocab, dim=64, num_classes=2):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.lstm = nn.LSTM(dim, dim, batch_first=True)
        self.fc = nn.Linear(dim, num_classes)
    def forward(self, x, lengths):
        x = self.emb(x)
        x, _ = self.lstm(x)
        pooled = x.mean(dim=1)
        return self.fc(pooled)
print(LSTMCls(1000)(torch.randint(0,1000,(4,12)), torch.tensor([12,11,10,9])).shape)`
        },
        {
          id: 'pack-padded',
          name: 'Pack padded sequences',
          tags: ['padding', 'lstm'],
          meta: 'Lines: ~45 — Difficulty: Medium',
          description: 'Use `pack_padded_sequence` for variable-length LSTMs.',
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
        },
        {
          id: 'bahdanau-attn',
          name: 'Bahdanau attention (score)',
          tags: ['attention', 'seq2seq'],
          meta: 'Lines: ~55 — Difficulty: Medium',
          description: 'Compute additive attention scores and context vector.',
          code: `import torch, torch.nn as nn
class AdditiveAttention(nn.Module):
    def __init__(self, qdim, kdim, h=64):
        super().__init__()
        self.Wq, self.Wk = nn.Linear(qdim, h), nn.Linear(kdim, h)
        self.v = nn.Linear(h, 1)
    def forward(self, q, K, mask=None):
        # q: (B, qdim), K: (B,T,kdim)
        qh = self.Wq(q).unsqueeze(1)
        Kh = self.Wk(K)
        scores = self.v(torch.tanh(qh + Kh)).squeeze(-1)  # (B,T)
        if mask is not None: scores = scores.masked_fill(~mask, -1e9)
        attn = torch.softmax(scores, dim=-1)
        ctx = (attn.unsqueeze(-1) * K).sum(dim=1)
        return ctx, attn
print('ok')`
        },
        {
          id: 'seq2seq-loop',
          name: 'Seq2Seq teacher forcing loop',
          tags: ['seq2seq', 'training'],
          meta: 'Lines: ~55 — Difficulty: Medium',
          description: 'Decoder consumes targets with teacher forcing ratio.',
          code: `# Pseudocode structure
teacher_forcing = 0.5
for t in range(T):
    use_tf = torch.rand(()) < teacher_forcing
    token = target[:, t-1] if (t>0 and use_tf) else pred.argmax(dim=-1)
    pred = decoder(token, hidden)`
        },
        {
          id: 'char-tokenizer',
          name: 'Char-level tokenizer',
          tags: ['tokenizer', 'nlp'],
          meta: 'Lines: ~35 — Difficulty: Easy',
          description: 'Map characters to ids and back.',
          code: `text = 'hello world'
chars = sorted(set(text))
stoi = {c:i for i,c in enumerate(chars)}
itos = {i:c for c,i in stoi.items()}
ids = [stoi[c] for c in text]
back = ''.join(itos[i] for i in ids)
print(ids, back)`
        },
        {
          id: 'bigram-lm',
          name: 'Bigram language model',
          tags: ['language-model', 'bigram'],
          meta: 'Lines: ~45 — Difficulty: Medium',
          description: 'Count-based bigram LM with sampling.',
          code: `import torch
text = 'abababaabbabbabbaba'
chars = sorted(set(text)); stoi = {c:i for i,c in enumerate(chars)}; itos = {i:c for c,i in stoi.items()}
N = torch.zeros((len(chars), len(chars)), dtype=torch.float32)
for a,b in zip(text, text[1:]): N[stoi[a], stoi[b]] += 1
P = (N + 1) / (N + 1).sum(1, keepdim=True)
idx = stoi['a']; out = ['a']
for _ in range(20):
    idx = torch.multinomial(P[idx], 1).item()
    out.append(itos[idx])
print(''.join(out))`
        },
        {
          id: 'mlm-loss',
          name: 'Masked LM loss (BERT-style)',
          tags: ['mlm', 'loss'],
          meta: 'Lines: ~35 — Difficulty: Medium',
          description: 'Compute loss on masked token positions only.',
          code: `import torch
logits = torch.randn(2, 5, 10)
targets = torch.randint(0, 10, (2,5))
mask = torch.tensor([[0,1,0,1,0],[1,0,0,0,1]], dtype=torch.bool)
loss = torch.nn.functional.cross_entropy(
    logits[mask], targets[mask]
)
print(loss.item())`
        },
        {
          id: 'greedy-decode',
          name: 'Greedy decode (autoregressive)',
          tags: ['decoding', 'lm'],
          meta: 'Lines: ~35 — Difficulty: Easy',
          description: 'Greedy next-token decoding loop.',
          code: `import torch
def greedy_decode(model, idx, max_new_tokens):
    model.eval()
    for _ in range(max_new_tokens):
        logits = model(idx)[:,-1,:]
        next_id = logits.argmax(dim=-1, keepdim=True)
        idx = torch.cat([idx, next_id], dim=1)
    return idx
print('ok')`
        },
      ],
    },

    {
      id: 'vision',
      name: 'Vision',
      summary: 'CNNs, augmentations, transfer learning, segmentation',
      topics: [
        {
          id: 'cnn-classifier',
          name: 'Simple CNN classifier',
          tags: ['cnn', 'classification'],
          meta: 'Lines: ~55 — Difficulty: Medium',
          description: 'Small CNN for image classification.',
          code: `import torch, torch.nn as nn
class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(32*8*8, 64), nn.ReLU(), nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))
print(SmallCNN()(torch.randn(2,3,32,32)).shape)`
        },
        {
          id: 'transfer-resnet18',
          name: 'Transfer learning (ResNet18)',
          tags: ['transfer', 'torchvision'],
          meta: 'Lines: ~35 — Difficulty: Easy',
          description: 'Replace final layer and fine-tune.',
          code: `import torch.nn as nn
from torchvision.models import resnet18
model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 10)`
        },
        {
          id: 'unet-block',
          name: 'UNet encoder-decoder block',
          tags: ['segmentation', 'unet'],
          meta: 'Lines: ~55 — Difficulty: Medium',
          description: 'Minimal UNet-like block composition.',
          code: `import torch, torch.nn as nn
def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(),
        nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU()
    )
down = nn.MaxPool2d(2)
up = nn.ConvTranspose2d(64, 32, 2, stride=2)
print('ok')`
        },
        {
          id: 'normalize-denormalize',
          name: 'Normalize / denormalize',
          tags: ['preprocess'],
          meta: 'Lines: ~35 — Difficulty: Easy',
          description: 'Apply and invert normalization for visualization.',
          code: `import torch
mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
x = torch.randn(3,224,224)
xn = (x - mean) / std
x_back = xn * std + mean
print(torch.allclose(x, x_back))`
        },
        {
          id: 'depthwise-separable-conv',
          name: 'Depthwise separable conv',
          tags: ['conv', 'mobile'],
          meta: 'Lines: ~45 — Difficulty: Medium',
          description: 'Depthwise conv + pointwise conv block.',
          code: `import torch, torch.nn as nn
class DWSeparable(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, padding=1, groups=cin)
        self.pw = nn.Conv2d(cin, cout, 1)
    def forward(self, x): return self.pw(self.dw(x))
print(DWSeparable(16,32)(torch.randn(1,16,32,32)).shape)`
        },
        {
          id: 'groupnorm',
          name: 'GroupNorm',
          tags: ['normalization'],
          meta: 'Lines: ~25 — Difficulty: Easy',
          description: 'Apply GroupNorm for small batch sizes.',
          code: `import torch, torch.nn as nn
gn = nn.GroupNorm(num_groups=4, num_channels=32)
print(gn(torch.randn(2,32,8,8)).shape)`
        },
        {
          id: 'squeeze-excitation',
          name: 'Squeeze-and-Excitation (SE)',
          tags: ['attention', 'se'],
          meta: 'Lines: ~45 — Difficulty: Medium',
          description: 'Channel attention via SE block.',
          code: `import torch, torch.nn as nn
class SE(nn.Module):
    def __init__(self, c, r=16):
        super().__init__(); self.fc = nn.Sequential(nn.Linear(c, c//r), nn.ReLU(), nn.Linear(c//r, c), nn.Sigmoid())
    def forward(self, x):
        w = x.mean(dim=(2,3))
        s = self.fc(w).unsqueeze(-1).unsqueeze(-1)
        return x * s
print(SE(32)(torch.randn(1,32,16,16)).shape)`
        },
        {
          id: 'random-erasing',
          name: 'RandomErasing (augmentation)',
          tags: ['augmentation'],
          meta: 'Lines: ~30 — Difficulty: Easy',
          description: 'Apply RandomErasing via torchvision transform.',
          code: `from torchvision import transforms
aug = transforms.RandomErasing(p=1.0)
print('RandomErasing ready')`
        },
        {
          id: 'ten-crop',
          name: 'TenCrop + inference averaging',
          tags: ['augmentation', 'inference'],
          meta: 'Lines: ~35 — Difficulty: Medium',
          description: 'Augment inference by averaging predictions over TenCrop.',
          code: `from torchvision import transforms
tc = transforms.TenCrop(224)
print('TenCrop ready')`
        },
      ],
    },

    {
      id: 'transformer',
      name: 'Transformer',
      summary: 'SDPA, multi-head, masks, encoder/decoder, full model',
      topics: [
        {
          id: 'sdpa',
          name: 'Scaled dot-product attention (SDPA)',
          tags: ['attention', 'sdpa'],
          meta: 'Lines: ~55 — Difficulty: Medium',
          description: 'Compute attention weights and context.',
          code: `import torch
def sdpa(Q, K, V, mask=None):
    d = Q.size(-1)
    scores = (Q @ K.transpose(-2, -1)) / d**0.5
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    return attn @ V
Q = torch.randn(2,4,8); K = torch.randn(2,4,8); V = torch.randn(2,4,8)
print(sdpa(Q,K,V).shape)`
        },
        {
          id: 'multi-head-attn',
          name: 'Multi-head attention',
          tags: ['attention', 'multi-head'],
          meta: 'Lines: ~75 — Difficulty: Hard',
          description: 'Minimal multi-head self-attention module.',
          code: `import torch, torch.nn as nn
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim*3)
        self.proj = nn.Linear(dim, dim)
    def forward(self, x, mask=None):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = (q @ k.transpose(-2,-1)) / (self.head_dim**0.5)
        if mask is not None: scores = scores.masked_fill(~mask, -1e9)
        attn = torch.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1,2).reshape(B, T, C)
        return self.proj(out)
print('ok')`
        },
        {
          id: 'causal-mask',
          name: 'Causal self-attention mask',
          tags: ['mask', 'causal'],
          meta: 'Lines: ~25 — Difficulty: Easy',
          description: 'Lower-triangular mask for autoregressive decoding.',
          code: `import torch
T = 5
mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
print(mask)`
        },
        {
          id: 'transformer-encoder-block',
          name: 'Transformer encoder block',
          tags: ['encoder', 'block'],
          meta: 'Lines: ~75 — Difficulty: Medium',
          description: 'Self-attention + MLP with LayerNorm and residuals.',
          code: `import torch, torch.nn as nn
class EncoderBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_ratio=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim*mlp_ratio), nn.GELU(), nn.Linear(dim*mlp_ratio, dim))
    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=attn_mask, need_weights=False)[0]
        x = x + self.mlp(self.ln2(x))
        return x
print('ok')`
        },
        {
          id: 'transformer-decoder-block',
          name: 'Transformer decoder block',
          tags: ['decoder', 'block'],
          meta: 'Lines: ~85 — Difficulty: Medium',
          description: 'Masked self-attn + cross-attn + MLP.',
          code: `import torch, torch.nn as nn
class DecoderBlock(nn.Module):
    def __init__(self, dim, heads=8, mlp_ratio=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln3 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim*mlp_ratio), nn.GELU(), nn.Linear(dim*mlp_ratio, dim))
    def forward(self, x, enc, attn_mask=None, key_padding_mask=None):
        x = x + self.self_attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=attn_mask, need_weights=False)[0]
        x = x + self.cross_attn(self.ln2(x), enc, enc, key_padding_mask=key_padding_mask, need_weights=False)[0]
        x = x + self.mlp(self.ln3(x))
        return x
print('ok')`
        },
        {
          id: 'full-transformer',
          name: 'Full Transformer LM (tiny)',
          tags: ['transformer', 'language-model'],
          meta: 'Lines: ~95 — Difficulty: Hard',
          description: 'Tiny Transformer for autoregressive LM.',
          code: `import torch, torch.nn as nn
class TinyTransformerLM(nn.Module):
    def __init__(self, vocab=1000, dim=128, depth=2, heads=4, ff=4, max_len=128):
        super().__init__()
        self.tok = nn.Embedding(vocab, dim)
        self.pos = nn.Parameter(torch.zeros(1, max_len, dim))
        self.blocks = nn.ModuleList([EncoderBlock(dim, heads, ff) for _ in range(depth)])
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab)
    def forward(self, x):
        B, T = x.shape
        h = self.tok(x) + self.pos[:, :T]
        mask = torch.tril(torch.ones(T, T, device=x.device)).bool()
        attn_mask = ~mask
        for blk in self.blocks:
            h = blk(h, attn_mask)
        h = self.ln(h)
        return self.head(h)
print('ok')`
        },
        {
          id: 'sdpa-functional',
          name: 'scaled_dot_product_attention (functional)',
          tags: ['attention', 'torch.nn.functional'],
          meta: 'Lines: ~30 — Difficulty: Easy',
          description: 'Use built-in SDPA for speed and stability.',
          code: `import torch
import torch.nn.functional as F
q = torch.randn(2,4,8); k = torch.randn(2,4,8); v = torch.randn(2,4,8)
out = F.scaled_dot_product_attention(q, k, v)
print(out.shape)`
        },
        {
          id: 'rope',
          name: 'Rotary positional embeddings (RoPE)',
          tags: ['positional', 'rope'],
          meta: 'Lines: ~55 — Difficulty: Medium',
          description: 'Apply RoPE to query and key vectors.',
          code: `import torch, math
def apply_rope(x):
    # x: (B, T, H, D) with even D
    B,T,H,D = x.shape; half = D // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half)/half)
    t = torch.arange(T).unsqueeze(1)
    angles = t * freqs
    cos, sin = angles.cos(), angles.sin()
    x1, x2 = x[..., :half], x[..., half:]
    x_ro = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return x_ro
print('rope ok')`
        },
        {
          id: 'alibi',
          name: 'ALiBi positional bias',
          tags: ['positional', 'bias'],
          meta: 'Lines: ~45 — Difficulty: Medium',
          description: 'Add linear bias to attention scores by distance.',
          code: `import torch
T = 6
dist = torch.arange(T).view(1,1,T) - torch.arange(T).view(1,T,1)
bias = -torch.relu(dist.float())
print(bias.shape)`
        },
        {
          id: 'glu-ffn',
          name: 'Gated FFN (GLU)',
          tags: ['ffn', 'glu'],
          meta: 'Lines: ~45 — Difficulty: Medium',
          description: 'Use gated linear units in FFN.',
          code: `import torch, torch.nn as nn
class GLUFFN(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__(); self.fc = nn.Linear(dim, hidden*2); self.out = nn.Linear(hidden, dim)
    def forward(self, x):
        a, b = self.fc(x).chunk(2, dim=-1)
        return self.out(a * torch.sigmoid(b))
print('ok')`
        },
        {
          id: 'prenorm-vs-postnorm',
          name: 'Pre-Norm vs Post-Norm',
          tags: ['norm', 'stability'],
          meta: 'Lines: ~45 — Difficulty: Medium',
          description: 'Pattern showing pre-norm residual structure.',
          code: `import torch, torch.nn as nn
class PreNormBlock(nn.Module):
    def __init__(self, dim, fn): super().__init__(); self.ln=nn.LayerNorm(dim); self.fn=fn
    def forward(self, x): return x + self.fn(self.ln(x))
print('ok')`
        },
        {
          id: 'topk-sampling',
          name: 'Top-k / nucleus sampling',
          tags: ['decoding', 'sampling'],
          meta: 'Lines: ~55 — Difficulty: Medium',
          description: 'Filter logits to top-k or top-p before sampling.',
          code: `import torch
def sample_filter(logits, top_k=0, top_p=0.0):
    logits = logits.clone()
    if top_k > 0:
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[..., -1, None]] = -float('inf')
    if top_p > 0.0:
        sorted_logits, idx = torch.sort(logits, descending=True)
        cumprobs = torch.softmax(sorted_logits, -1).cumsum(-1)
        mask = cumprobs > top_p
        mask[..., 1:] = mask[..., :-1].clone(); mask[..., 0] = False
        sorted_logits[mask] = -float('inf')
        logits = torch.gather(sorted_logits, -1, torch.argsort(idx))
    probs = torch.softmax(logits, -1)
    return torch.multinomial(probs, 1)
print('ok')`
        },
        {
          id: 'beam-search',
          name: 'Beam search (sketch)',
          tags: ['decoding', 'beam'],
          meta: 'Lines: ~45 — Difficulty: Medium',
          description: 'Maintain top beams by sum of log-probs.',
          code: `# pseudocode sketch for clarity in interviews
beams = [(seq0, 0.0)]
for step in range(T):
    cand = []
    for seq, score in beams:
        logits = model(seq)[:,-1]
        logp = torch.log_softmax(logits, -1)
        topv, topi = logp.topk(5)
        for v,i in zip(topv, topi):
            cand.append((seq+[i], score+v.item()))
    beams = sorted(cand, key=lambda x: x[1], reverse=True)[:beam_size]`
        },
      ],
    },

    {
      id: 'clip',
      name: 'CLIP & Multimodal',
      summary: 'Dual encoders, contrastive loss, temperature, zero-shot',
      topics: [
        {
          id: 'clip-dual-encoder',
          name: 'Dual encoders (text/image)',
          tags: ['clip', 'multimodal'],
          meta: 'Lines: ~75 — Difficulty: Medium',
          description: 'Encode text and image to a joint embedding space.',
          code: `import torch, torch.nn as nn
class TextEncoder(nn.Module):
    def __init__(self, vocab=1000, dim=256):
        super().__init__(); self.emb = nn.Embedding(vocab, dim); self.pool = nn.AdaptiveAvgPool1d(1)
    def forward(self, x):
        e = self.emb(x).transpose(1,2)
        return self.pool(e).squeeze(-1)
class ImageEncoder(nn.Module):
    def __init__(self, dim=256):
        super().__init__(); self.net = nn.Sequential(nn.Conv2d(3,32,3,2,1), nn.ReLU(), nn.Conv2d(32,64,3,2,1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, dim))
    def forward(self, x): return self.net(x)
print('ok')`
        },
        {
          id: 'infonce-contrastive',
          name: 'InfoNCE contrastive loss',
          tags: ['clip', 'loss'],
          meta: 'Lines: ~55 — Difficulty: Medium',
          description: 'Symmetric cross-entropy over similarity matrix.',
          code: `import torch
def clip_loss(z_img, z_txt, temp=0.07):
    z_img = torch.nn.functional.normalize(z_img, dim=-1)
    z_txt = torch.nn.functional.normalize(z_txt, dim=-1)
    logits = z_img @ z_txt.t() / temp
    targets = torch.arange(z_img.size(0), device=z_img.device)
    li = torch.nn.functional.cross_entropy(logits, targets)
    lt = torch.nn.functional.cross_entropy(logits.t(), targets)
    return (li + lt) / 2
print('ok')`
        },
        {
          id: 'zero-shot',
          name: 'Zero-shot classification',
          tags: ['clip', 'zero-shot'],
          meta: 'Lines: ~45 — Difficulty: Easy',
          description: 'Compare image embeddings with text prompts.',
          code: `import torch
img = torch.randn(1, 256)
texts = torch.randn(5, 256)
img = torch.nn.functional.normalize(img, dim=-1)
texts = torch.nn.functional.normalize(texts, dim=-1)
logits = img @ texts.t()
pred = logits.argmax(dim=-1)
print(pred)`
        },
        {
          id: 'learnable-temperature',
          name: 'Learnable temperature (logit scale)',
          tags: ['clip', 'temperature'],
          meta: 'Lines: ~35 — Difficulty: Easy',
          description: 'Parameterize temperature as exp(logit_scale).',
          code: `import torch, torch.nn as nn
logit_scale = nn.Parameter(torch.tensor(0.07).log())
z1, z2 = torch.randn(4,256), torch.randn(4,256)
z1 = torch.nn.functional.normalize(z1, dim=-1); z2 = torch.nn.functional.normalize(z2, dim=-1)
logits = (z1 @ z2.t()) * logit_scale.exp()`
        },
        {
          id: 'projection-heads',
          name: 'Projection heads',
          tags: ['clip', 'projection'],
          meta: 'Lines: ~35 — Difficulty: Easy',
          description: 'Linear projection to shared space before contrastive loss.',
          code: `import torch, torch.nn as nn
img_enc = nn.Linear(512, 256)
txt_enc = nn.Linear(768, 256)
print(img_enc(torch.randn(2,512)).shape, txt_enc(torch.randn(2,768)).shape)`
        },
      ],
    },

    {
      id: 'generative',
      name: 'Generative',
      summary: 'VAE, GAN, diffusion step, autoregressive sampling',
      topics: [
        {
          id: 'vae',
          name: 'VAE (tiny)',
          tags: ['vae', 'generative'],
          meta: 'Lines: ~85 — Difficulty: Hard',
          description: 'Minimal VAE with reparameterization.',
          code: `import torch, torch.nn as nn
class VAE(nn.Module):
    def __init__(self, d=784, h=256, z=32):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, h), nn.ReLU())
        self.mu = nn.Linear(h, z); self.logvar = nn.Linear(h, z)
        self.dec = nn.Sequential(nn.Linear(z, h), nn.ReLU(), nn.Linear(h, d))
    def reparam(self, mu, logvar):
        eps = torch.randn_like(mu)
        return mu + eps * (0.5*logvar).exp()
    def forward(self, x):
        h = self.enc(x)
        mu, logvar = self.mu(h), self.logvar(h)
        z = self.reparam(mu, logvar)
        return self.dec(z), mu, logvar
def elbo(recon, x, mu, logvar):
    recon_loss = torch.nn.functional.mse_loss(recon, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (recon_loss + kld) / x.size(0)
print('ok')`
        },
        {
          id: 'gan-minimal',
          name: 'GAN (minimal)',
          tags: ['gan', 'generative'],
          meta: 'Lines: ~85 — Difficulty: Hard',
          description: 'Tiny GAN architecture and loss.',
          code: `import torch, torch.nn as nn
G = nn.Sequential(nn.Linear(32,64), nn.ReLU(), nn.Linear(64,784), nn.Tanh())
D = nn.Sequential(nn.Linear(784,64), nn.LeakyReLU(0.2), nn.Linear(64,1))
optG = torch.optim.Adam(G.parameters(), 2e-4); optD = torch.optim.Adam(D.parameters(), 2e-4)
real = torch.randn(16, 784)
z = torch.randn(16, 32)
fake = G(z)
# D loss
lossD = (torch.nn.functional.binary_cross_entropy_with_logits(D(real), torch.ones(16,1)) +
         torch.nn.functional.binary_cross_entropy_with_logits(D(fake.detach()), torch.zeros(16,1)))
optD.zero_grad(); lossD.backward(); optD.step()
# G loss
lossG = torch.nn.functional.binary_cross_entropy_with_logits(D(fake), torch.ones(16,1))
optG.zero_grad(); lossG.backward(); optG.step()`
        },
        {
          id: 'gumbel-softmax',
          name: 'Gumbel-Softmax sampling',
          tags: ['discrete', 'relaxation'],
          meta: 'Lines: ~35 — Difficulty: Medium',
          description: 'Sample from categorical with Gumbel-Softmax.',
          code: `import torch
logits = torch.randn(3, 5)
g = -torch.log(-torch.log(torch.rand_like(logits)))
tau = 0.5
sample = torch.softmax((logits + g) / tau, dim=-1)
print(sample.shape)`
        },
        {
          id: 'ddpm-forward-step',
          name: 'DDPM forward (q(x_t|x_0))',
          tags: ['diffusion'],
          meta: 'Lines: ~45 — Difficulty: Medium',
          description: 'Add noise according to schedule to get x_t.',
          code: `import torch
T = 1000
betas = torch.linspace(1e-4, 0.02, T)
alphas = 1 - betas
alpha_bar = torch.cumprod(alphas, dim=0)
def q_sample(x0, t):
    noise = torch.randn_like(x0)
    ab = alpha_bar[t].view(-1,1,1,1)
    return (ab.sqrt() * x0) + ((1 - ab).sqrt() * noise)
print('ok')`
        },
        {
          id: 'autoregressive-sampling',
          name: 'Autoregressive sampling loop',
          tags: ['sampling', 'lm'],
          meta: 'Lines: ~35 — Difficulty: Easy',
          description: 'Generic next-token loop for AR models.',
          code: `import torch
def sample(model, idx, steps):
    for _ in range(steps):
        logits = model(idx)[:,-1]
        idx_next = torch.multinomial(torch.softmax(logits, -1), 1)
        idx = torch.cat([idx, idx_next], dim=1)
    return idx
print('ok')`
        },
      ],
    },

    {
      id: 'utilities',
      name: 'Utilities & Tricks',
      summary: 'Meters, profiling, progress, parameter counts',
      topics: [
        {
          id: 'average-meter',
          name: 'AverageMeter',
          tags: ['meter', 'logging'],
          meta: 'Lines: ~30 — Difficulty: Easy',
          description: 'Track running averages for metrics.',
          code: `class AverageMeter:
    def __init__(self): self.sum = 0.0; self.n = 0
    def update(self, val, k=1): self.sum += val * k; self.n += k
    @property
    def avg(self): return self.sum / max(1, self.n)
meter = AverageMeter(); meter.update(2, k=4); print(meter.avg)`
        },
        {
          id: 'count-params',
          name: 'Count parameters',
          tags: ['utils'],
          meta: 'Lines: ~20 — Difficulty: Easy',
          description: 'Count trainable parameters in a model.',
          code: `import torch.nn as nn
net = nn.Sequential(nn.Linear(10,10), nn.ReLU(), nn.Linear(10,2))
num = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(num)`
        },
        {
          id: 'profiler',
          name: 'Profiler (simple)',
          tags: ['performance', 'profile'],
          meta: 'Lines: ~35 — Difficulty: Medium',
          description: 'Use PyTorch profiler to time code regions.',
          code: `import torch, torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
model = nn.Linear(1000, 1000)
x = torch.randn(64, 1000)
with profile(activities=[ProfilerActivity.CPU]) as prof:
    with record_function('forward'):
        model(x)
print(prof.key_averages().table(sort_by='cpu_time_total'))`
        },
        {
          id: 'set-grad-enabled',
          name: 'set_grad_enabled',
          tags: ['autograd', 'context'],
          meta: 'Lines: ~20 — Difficulty: Easy',
          description: 'Globally enable/disable autograd in a context.',
          code: `import torch
with torch.set_grad_enabled(False):
    x = torch.ones(3, requires_grad=True)
    y = x * 2
print(y.requires_grad)`
        },
        {
          id: 'tqdm-progress',
          name: 'tqdm progress bar',
          tags: ['progress'],
          meta: 'Lines: ~25 — Difficulty: Easy',
          description: 'Wrap a loop with tqdm for progress (if available).',
          code: `from tqdm import tqdm
for _ in tqdm(range(1000)):
    pass  # work`
        },
        {
          id: 'model-ema',
          name: 'Model EMA (exponential moving average)',
          tags: ['ema', 'stability'],
          meta: 'Lines: ~45 — Difficulty: Medium',
          description: 'Keep an EMA copy of model parameters.',
          code: `import torch
def update_ema(ema, model, decay=0.999):
    with torch.no_grad():
        for p_ema, p in zip(ema.parameters(), model.parameters()):
            p_ema.data.mul_(decay).add_(p.data, alpha=1-decay)
print('ok')`
        },
        {
          id: 'grad-norm-logger',
          name: 'Gradient norm logger',
          tags: ['logging', 'grad'],
          meta: 'Lines: ~30 — Difficulty: Easy',
          description: 'Compute global grad norm for debugging.',
          code: `import torch
def grad_global_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return total ** 0.5
print('ok')`
        },
        {
          id: 'save-best-checkpoint',
          name: 'Save best checkpoint by metric',
          tags: ['checkpoint'],
          meta: 'Lines: ~30 — Difficulty: Easy',
          description: 'Track best validation score and save.',
          code: `best = -1e9
metric = 0.75
if metric > best:
    best = metric
    # torch.save(model.state_dict(), 'best.pt')
print('ok')`
        },
        {
          id: 'seed-worker',
          name: 'DataLoader worker_init_fn',
          tags: ['seed', 'repro'],
          meta: 'Lines: ~35 — Difficulty: Easy',
          description: 'Seed numpy and random in DataLoader workers.',
          code: `import numpy as np, random, torch
def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed); random.seed(seed)
print('ok')`
        },
        {
          id: 'model-summary',
          name: 'Model summary (shapes)',
          tags: ['summary'],
          meta: 'Lines: ~40 — Difficulty: Medium',
          description: 'Forward a dummy input to print layer output shapes.',
          code: `import torch, torch.nn as nn
model = nn.Sequential(nn.Linear(10,20), nn.ReLU(), nn.Linear(20,3))
x = torch.randn(2,10)
for i, layer in enumerate(model):
    x = layer(x)
    print(f'layer {i}:', tuple(x.shape))`
        },
      ],
    },

    {
      id: 'devices-memory',
      name: 'Devices & Memory',
      summary: 'Device moves, pin_memory, inference_mode, precision, parallel',
      topics: [
        {
          id: 'to-device-helper',
          name: 'to_device helper',
          tags: ['device'],
          meta: 'Lines: ~30 — Difficulty: Easy',
          description: 'Move nested tensors to a device recursively.',
          code: `import torch
def to_device(x, device):
    if torch.is_tensor(x): return x.to(device)
    if isinstance(x, (list, tuple)): return type(x)(to_device(v, device) for v in x)
    if isinstance(x, dict): return {k: to_device(v, device) for k,v in x.items()}
    return x
print('ok')`
        },
        {
          id: 'pin-memory-nonblocking',
          name: 'pin_memory + non_blocking',
          tags: ['dataloader', 'cuda'],
          meta: 'Lines: ~35 — Difficulty: Easy',
          description: 'Speed up host->device transfers.',
          code: `import torch
from torch.utils.data import DataLoader, TensorDataset
X = torch.randn(128, 10); y = torch.randint(0,2,(128,))
loader = DataLoader(TensorDataset(X,y), batch_size=32, pin_memory=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
for xb, yb in loader:
    xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
print('ok')`
        },
        {
          id: 'inference-mode',
          name: 'torch.inference_mode()',
          tags: ['eval', 'no_grad'],
          meta: 'Lines: ~25 — Difficulty: Easy',
          description: 'Faster than no_grad for inference.',
          code: `import torch
with torch.inference_mode():
    y = (torch.randn(3,3) @ torch.randn(3,3))
print(y.requires_grad)`
        },
        {
          id: 'cuda-memory-summary',
          name: 'CUDA memory summary',
          tags: ['cuda', 'memory'],
          meta: 'Lines: ~20 — Difficulty: Easy',
          description: 'Print memory allocator statistics.',
          code: `import torch
if torch.cuda.is_available():
    print(torch.cuda.memory_summary())`
        },
        {
          id: 'matmul-precision',
          name: 'matmul precision (fp32)',
          tags: ['precision'],
          meta: 'Lines: ~25 — Difficulty: Easy',
          description: 'Control float32 matmul precision (pytorch 2).',
          code: `import torch
torch.set_float32_matmul_precision('high')
print('precision set')`
        },
        {
          id: 'persistent-workers',
          name: 'DataLoader persistent_workers',
          tags: ['dataloader'],
          meta: 'Lines: ~25 — Difficulty: Easy',
          description: 'Keep workers alive between epochs for speed.',
          code: `from torch.utils.data import DataLoader, TensorDataset
import torch
X = torch.randn(64, 4); y = torch.randint(0,2,(64,))
loader = DataLoader(TensorDataset(X,y), batch_size=16, num_workers=2, persistent_workers=True)`
        },
        {
          id: 'dataparallel',
          name: 'nn.DataParallel (simple)',
          tags: ['parallel'],
          meta: 'Lines: ~35 — Difficulty: Easy',
          description: 'Use DataParallel across GPUs (single process).',
          code: `import torch, torch.nn as nn
if torch.cuda.device_count() > 1:
    model = nn.Linear(10, 2)
    model = nn.DataParallel(model)
    print('wrapped')`
        },
        {
          id: 'bfloat16',
          name: 'bfloat16 inference',
          tags: ['precision', 'bf16'],
          meta: 'Lines: ~30 — Difficulty: Easy',
          description: 'Cast model and inputs to bfloat16 for inference.',
          code: `import torch, torch.nn as nn
model = nn.Linear(10, 2).eval().to(dtype=torch.bfloat16)
x = torch.randn(2,10, dtype=torch.bfloat16)
print(model(x).dtype)`
        },
      ],
    },

    {
      id: 'losses',
      name: 'Losses',
      summary: 'Classification/regression and metric learning losses',
      topics: [
        {
          id: 'focal-loss',
          name: 'Focal loss (binary)',
          tags: ['classification', 'imbalance'],
          meta: 'Lines: ~45 — Difficulty: Medium',
          description: 'Down-weight easy examples.',
          code: `import torch
def focal_loss(logits, targets, gamma=2.0):
    prob = torch.sigmoid(logits)
    pt = torch.where(targets==1, prob, 1-prob)
    w = (1-pt).pow(gamma)
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, targets.float(), weight=w)
print('ok')`
        },
        {
          id: 'triplet-loss',
          name: 'Triplet loss (usage)',
          tags: ['metric-learning'],
          meta: 'Lines: ~35 — Difficulty: Easy',
          description: 'Use `TripletMarginLoss` with embeddings.',
          code: `import torch
from torch.nn import TripletMarginLoss
emb = torch.randn(4, 16)
loss = TripletMarginLoss(margin=1.0)(emb[0], emb[1], emb[2])
print(loss.item())`
        },
        {
          id: 'margin-ranking',
          name: 'MarginRankingLoss',
          tags: ['ranking'],
          meta: 'Lines: ~30 — Difficulty: Easy',
          description: 'Encourage score(x1) > score(x2).',
          code: `import torch
loss = torch.nn.MarginRankingLoss(margin=1.0)
x1, x2 = torch.randn(3), torch.randn(3)
target = torch.tensor([1.0, 1.0, 1.0])
print(loss(x1, x2, target))`
        },
        {
          id: 'cosine-embedding',
          name: 'CosineEmbeddingLoss',
          tags: ['similarity'],
          meta: 'Lines: ~30 — Difficulty: Easy',
          description: 'Similarity/dissimilarity supervision.',
          code: `import torch
loss = torch.nn.CosineEmbeddingLoss()
x1, x2 = torch.randn(2,5), torch.randn(2,5)
target = torch.tensor([1, -1])
print(loss(x1, x2, target))`
        },
        {
          id: 'kldiv',
          name: 'KLDivLoss (log_softmax + soft targets)',
          tags: ['distillation'],
          meta: 'Lines: ~35 — Difficulty: Medium',
          description: 'KL divergence between distributions.',
          code: `import torch
p = torch.log_softmax(torch.randn(3,5), dim=-1)
q = torch.softmax(torch.randn(3,5), dim=-1)
loss = torch.nn.KLDivLoss(reduction='batchmean')(p, q)
print(loss)`
        },
        {
          id: 'bce-pos-weight',
          name: 'BCEWithLogits (pos_weight)',
          tags: ['imbalance'],
          meta: 'Lines: ~30 — Difficulty: Easy',
          description: 'Handle class imbalance with pos_weight.',
          code: `import torch
logits = torch.randn(4,1)
targets = torch.tensor([[1.],[0.],[1.],[0.]])
loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]))(logits, targets)
print(loss)`
        },
        {
          id: 'ctc-loss',
          name: 'CTCLoss (toy)',
          tags: ['sequence', 'ctc'],
          meta: 'Lines: ~45 — Difficulty: Hard',
          description: 'Connectionist Temporal Classification loss.',
          code: `import torch
T, N, C = 4, 2, 3
logits = torch.randn(T, N, C).log_softmax(2)
target = torch.tensor([1,2,1], dtype=torch.long)
input_lengths = torch.full((N,), T, dtype=torch.long)
target_lengths = torch.tensor([2,1], dtype=torch.long)
loss = torch.nn.CTCLoss(blank=0)(logits, target, input_lengths, target_lengths)
print(loss)`
        },
        {
          id: 'huber-smoothl1',
          name: 'Huber / SmoothL1Loss',
          tags: ['regression'],
          meta: 'Lines: ~25 — Difficulty: Easy',
          description: 'Robust regression loss.',
          code: `import torch
loss = torch.nn.SmoothL1Loss(beta=1.0)
print(loss(torch.randn(5), torch.randn(5)))`
        },
      ],
    },
  ],
};


