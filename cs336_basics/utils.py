import torch
import math
from typing import Optional
from collections import Callable, Iterable

from mpmath import beta


def cross_entropy_loss(inputs, labels):
    """
    inputs: [batch_size, vocab_size]
    labels: [batch_size]
    """

    log_softmax = inputs - inputs.logsumexp(dim=1, keepdim=True)

    target_log_probs = log_softmax.gather(1, labels.unsqueeze(1)).squeeze(1)

    loss = -target_log_probs.mean()

    return loss

class SGDOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        assert lr > 0
        self.params = params
        self.defaults = dict(lr=lr)
        super(SGDOptimizer, self).__init__(params, defaults=self.defaults)

    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                gard = p.grad.data
                p.data = p.data - lr / math.sqrt(t + 1)  * gard
                state["t"] = t + 1

        return loss


class AdamWOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        if not lr >= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0 <= betas[0] < 1:
            raise ValueError("Invalid beta parameter: {}".format(betas[0]))
        if not 0 <= betas[1] < 1:
            raise ValueError("Invalid beta parameter: {}".format(betas[1]))
        if not eps >= 0:
            raise ValueError("Invalid epsilon value: {}".format(eps))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        super(AdamWOptimizer, self).__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        loss = None if closure is None else closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group['betas']
                lr = group['lr']
                eps = group['eps']
                weight_decay = group['weight_decay']

                state["step"] += 1
                grad = p.grad.data

                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denom_sqrt_exp_avg_sq = exp_avg_sq.sqrt().add_(eps)
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                step_size = lr * (bias_correction2 ** 0.5) / bias_correction1
                p.data.addcdiv_(exp_avg, denom_sqrt_exp_avg_sq, value=-step_size)
                p.data.mul_(1 - lr * weight_decay)
        return loss









