import os
import typing

import torch
import math
from typing import Optional
from collections import Callable, Iterable
import numpy as np
import numpy.typing as npt

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
                grad = p.grad.data
                p.data = p.data - lr / math.sqrt(t + 1)  * grad
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

def cosine_learning_rate_schedule_with_warmup(
        it: int,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_iters: int,
        cosine_cycle_iters: int,
    ):

    warmup_learning_rate: float = 1e-3
    if it < warmup_iters:
        warmup_learning_rate = max_learning_rate * it / warmup_iters
    elif warmup_iters <= it <= cosine_cycle_iters:
        warmup_learning_rate = min_learning_rate + 0.5 * (1 + math.cos(math.pi * (it - warmup_iters)
                                                                       / (cosine_cycle_iters - warmup_iters))) * (
                                           max_learning_rate - min_learning_rate)
    elif it > cosine_cycle_iters:
        warmup_learning_rate = min_learning_rate

    return warmup_learning_rate

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    parameters_with_grad = [p for p in parameters if p.grad is not None]

    if len(parameters_with_grad) == 0:
        return

    norm = torch.sqrt(sum(torch.sum(p.grad.pow(2)) for p in parameters_with_grad))
    coef = max_l2_norm / (norm + 1e-6)
    if norm >= max_l2_norm:
        for p in parameters_with_grad:
            p.grad.mul_(coef)

def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str):
    starts = np.random.randint(
        low=0,
        high=len(dataset) - context_length,
        size=(batch_size,),
    )
    inputs = np.stack([dataset[start: start+context_length] for start in starts])
    labels = np.stack([dataset[start+1: start+context_length+1] for start in starts])

    return (
        torch.from_numpy(inputs).long().to(device),
        torch.from_numpy(labels).long().to(device),
    )

def save_checkpoint(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        iteration: int,
        out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    if isinstance(out, (os.PathLike, str)):
        with open(out, "wb") as f:
            torch.save(checkpoint, f)
    else:
        torch.save(checkpoint, out)

def load_checkpoint(
        src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
    ):
    if isinstance(src, (os.PathLike, str)):
        with open(src, "rb") as f:
            checkpoint = torch.load(f)
    else:
        checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint["iteration"]












