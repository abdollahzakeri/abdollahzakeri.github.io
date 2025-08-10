(function(){
  window.registerExample(
    'utilities',
    { categoryName: 'Utilities & Tricks', categorySummary: 'Meters, profiling, progress, parameter counts', topicId: 'average-meter', topicName: 'AverageMeter' },
    {
      id: 'average-meter',
      name: 'AverageMeter',
      tags: ['meter','logging'],
      meta: 'Track running averages for metrics',
      description: 'Simple utility to maintain running averages of values.',
      code: `class AverageMeter:
    def __init__(self): self.sum = 0.0; self.n = 0
    def update(self, val, k=1): self.sum += float(val) * k; self.n += k
    @property
    def avg(self): return self.sum / max(1, self.n)
meter = AverageMeter(); meter.update(2, k=4); print(meter.avg)`
    }
  );
})();


