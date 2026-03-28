from __future__ import annotations

import numpy as np

from .backend import get_backend

try:
	import torch
except Exception:
	torch = None


if get_backend() == "torch" and torch is not None:
	AdamW = torch.optim.AdamW
	SGD = torch.optim.SGD
else:
	class SGD:
		def __init__(self, params, lr: float = 1e-2, weight_decay: float = 0.0):
			self.params = list(params)
			self.lr = float(lr)
			self.weight_decay = float(weight_decay)

		def zero_grad(self, set_to_none: bool = False):
			for p in self.params:
				if set_to_none:
					p.grad = np.zeros_like(p.data)
				else:
					p.grad.fill(0.0)

		def step(self):
			for p in self.params:
				grad = p.grad
				if self.weight_decay != 0.0:
					grad = grad + self.weight_decay * p.data
				p.data -= self.lr * grad


	class AdamW:
		def __init__(
			self,
			params,
			lr: float = 1e-3,
			betas: tuple[float, float] = (0.9, 0.999),
			eps: float = 1e-8,
			weight_decay: float = 1e-2,
		):
			self.params = list(params)
			self.lr = float(lr)
			self.beta1, self.beta2 = betas
			self.eps = float(eps)
			self.weight_decay = float(weight_decay)
			self.step_count = 0
			self.m = [np.zeros_like(p.data) for p in self.params]
			self.v = [np.zeros_like(p.data) for p in self.params]

		def zero_grad(self, set_to_none: bool = False):
			for p in self.params:
				if set_to_none:
					p.grad = np.zeros_like(p.data)
				else:
					p.grad.fill(0.0)

		def step(self):
			self.step_count += 1
			for i, p in enumerate(self.params):
				grad = p.grad
				if self.weight_decay != 0.0:
					p.data *= (1.0 - self.lr * self.weight_decay)

				self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grad
				self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (grad * grad)

				m_hat = self.m[i] / (1.0 - self.beta1**self.step_count)
				v_hat = self.v[i] / (1.0 - self.beta2**self.step_count)
				p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
