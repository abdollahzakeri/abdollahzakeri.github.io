(function(){
  window.registerExample(
    'autograd',
    { categoryName: 'Autograd', categorySummary: 'Custom Functions, hooks, detach, checkpointing', topicId: 'custom-function', topicName: 'Custom autograd Function' },
    {
      id: 'custom-function',
      name: 'Custom autograd Function',
      tags: ['autograd','custom'],
      meta: 'Forward/backward for a square operation',
      description: 'Defines a custom autograd Function computing y=x^2 with manual backward.',
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
    }
  );
})();


