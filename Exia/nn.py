from __future__ import annotations

import numpy as np

from .backend import get_backend

try:
	import torch
except Exception:
	torch = None


if get_backend() == "torch" and torch is not None:
	Module = torch.nn.Module
	Sequential = torch.nn.Sequential
	Linear = torch.nn.Linear
	Conv2d = torch.nn.Conv2d
	ReLU = torch.nn.ReLU
	GELU = torch.nn.GELU
	Dropout = torch.nn.Dropout
	LayerNorm = torch.nn.LayerNorm
	Embedding = torch.nn.Embedding
	CrossEntropyLoss = torch.nn.CrossEntropyLoss
else:
	class Parameter:
		def __init__(self, data):
			self.data = np.asarray(data, dtype=np.float32)
			self.grad = np.zeros_like(self.data)


	class Module:
		def __init__(self):
			self.training = True

		def forward(self, x):
			raise NotImplementedError

		def backward(self, grad_output):
			raise NotImplementedError

		def __call__(self, x):
			return self.forward(x)

		def parameters(self):
			return []

		def zero_grad(self):
			for p in self.parameters():
				p.grad.fill(0.0)

		def train(self):
			self.training = True
			return self

		def eval(self):
			self.training = False
			return self


	class Sequential(Module):
		def __init__(self, *layers):
			super().__init__()
			self.layers = list(layers)

		def forward(self, x):
			out = np.asarray(x, dtype=np.float32)
			for layer in self.layers:
				out = layer(out)
			return out

		def backward(self, grad_output):
			grad = grad_output
			for layer in reversed(self.layers):
				grad = layer.backward(grad)
			return grad

		def parameters(self):
			params = []
			for layer in self.layers:
				if hasattr(layer, "parameters"):
					params.extend(layer.parameters())
			return params


	class Linear(Module):
		def __init__(self, in_features: int, out_features: int):
			super().__init__()
			scale = np.sqrt(2.0 / max(1, in_features))
			self.weight = Parameter(np.random.randn(in_features, out_features).astype(np.float32) * scale)
			self.bias = Parameter(np.zeros((out_features,), dtype=np.float32))
			self._x = None

		def forward(self, x):
			x_arr = np.asarray(x, dtype=np.float32)
			self._x = x_arr
			return x_arr @ self.weight.data + self.bias.data

		def backward(self, grad_output):
			grad = np.asarray(grad_output, dtype=np.float32)
			self.weight.grad += self._x.T @ grad
			self.bias.grad += grad.sum(axis=0)
			return grad @ self.weight.data.T

		def parameters(self):
			return [self.weight, self.bias]


	class ReLU(Module):
		def __init__(self):
			super().__init__()
			self._mask = None

		def forward(self, x):
			x_arr = np.asarray(x, dtype=np.float32)
			self._mask = x_arr > 0.0
			return np.where(self._mask, x_arr, 0.0)

		def backward(self, grad_output):
			grad = np.asarray(grad_output, dtype=np.float32)
			return grad * self._mask


	class GELU(Module):
		def __init__(self):
			super().__init__()
			self._x = None

		def forward(self, x):
			x_arr = np.asarray(x, dtype=np.float32)
			self._x = x_arr
			c = np.sqrt(2.0 / np.pi)
			t = c * (x_arr + 0.044715 * (x_arr**3))
			return 0.5 * x_arr * (1.0 + np.tanh(t))

		def backward(self, grad_output):
			x = self._x
			c = np.sqrt(2.0 / np.pi)
			t = c * (x + 0.044715 * (x**3))
			tanh_t = np.tanh(t)
			left = 0.5 * (1.0 + tanh_t)
			right = 0.5 * x * (1.0 - tanh_t**2) * c * (1.0 + 3.0 * 0.044715 * (x**2))
			return np.asarray(grad_output, dtype=np.float32) * (left + right)


	class Dropout(Module):
		def __init__(self, p: float = 0.5):
			super().__init__()
			if not (0.0 <= p < 1.0):
				raise ValueError("dropout p must satisfy 0 <= p < 1")
			self.p = float(p)
			self._mask = None

		def forward(self, x):
			x_arr = np.asarray(x, dtype=np.float32)
			if not self.training or self.p == 0.0:
				self._mask = np.ones_like(x_arr, dtype=np.float32)
				return x_arr
			keep = 1.0 - self.p
			self._mask = (np.random.rand(*x_arr.shape) < keep).astype(np.float32) / keep
			return x_arr * self._mask

		def backward(self, grad_output):
			return np.asarray(grad_output, dtype=np.float32) * self._mask


	class LayerNorm(Module):
		def __init__(self, normalized_shape: int, eps: float = 1e-5):
			super().__init__()
			self.eps = float(eps)
			self.gamma = Parameter(np.ones((normalized_shape,), dtype=np.float32))
			self.beta = Parameter(np.zeros((normalized_shape,), dtype=np.float32))
			self._x_hat = None

		def forward(self, x):
			x_arr = np.asarray(x, dtype=np.float32)
			mean = x_arr.mean(axis=-1, keepdims=True)
			var = x_arr.var(axis=-1, keepdims=True)
			self._x_hat = (x_arr - mean) / np.sqrt(var + self.eps)
			return self._x_hat * self.gamma.data + self.beta.data

		def backward(self, grad_output):
			grad = np.asarray(grad_output, dtype=np.float32)
			self.gamma.grad += (grad * self._x_hat).sum(axis=0)
			self.beta.grad += grad.sum(axis=0)
			return grad

		def parameters(self):
			return [self.gamma, self.beta]


	class Embedding(Module):
		def __init__(self, num_embeddings: int, embedding_dim: int):
			super().__init__()
			self.weight = Parameter((np.random.randn(num_embeddings, embedding_dim) * 0.02).astype(np.float32))
			self._indices = None

		def forward(self, x):
			indices = np.asarray(x, dtype=np.int64)
			self._indices = indices
			return self.weight.data[indices]

		def backward(self, grad_output):
			grad = np.asarray(grad_output, dtype=np.float32)
			self.weight.grad.fill(0.0)
			np.add.at(self.weight.grad, self._indices, grad)
			return np.zeros_like(self._indices, dtype=np.float32)

		def parameters(self):
			return [self.weight]


	class CrossEntropyLoss:
		def __init__(self):
			self._probs = None
			self._targets = None

		def __call__(self, logits, targets) -> float:
			z = np.asarray(logits, dtype=np.float32)
			y = np.asarray(targets, dtype=np.int64).reshape(-1)
			if z.ndim != 2:
				raise ValueError("logits must be 2D [batch, num_classes]")
			if y.shape[0] != z.shape[0]:
				raise ValueError("targets must match batch size")
			z_shift = z - z.max(axis=1, keepdims=True)
			exp = np.exp(z_shift)
			probs = exp / exp.sum(axis=1, keepdims=True)
			self._probs = probs
			self._targets = y
			loss = -np.log(np.clip(probs[np.arange(y.shape[0]), y], 1e-12, 1.0)).mean()
			return float(loss)

		def backward(self):
			grad = self._probs.copy()
			grad[np.arange(self._targets.shape[0]), self._targets] -= 1.0
			grad /= self._targets.shape[0]
			return grad.astype(np.float32)


	class _NotImplementedLayer(Module):
		def __init__(self, *args, **kwargs):
			super().__init__()
			raise NotImplementedError("This layer is not implemented in standalone mode yet")


	Conv2d = _NotImplementedLayer
