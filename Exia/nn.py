try:
	import torch
except Exception:
	torch = None


class _TorchRequired:
	def __init__(self, *args, **kwargs):
		raise RuntimeError("This API requires torch. Install with: pip install Exia[torch]")


if torch is not None:
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
	Module = _TorchRequired
	Sequential = _TorchRequired
	Linear = _TorchRequired
	Conv2d = _TorchRequired
	ReLU = _TorchRequired
	GELU = _TorchRequired
	Dropout = _TorchRequired
	LayerNorm = _TorchRequired
	Embedding = _TorchRequired
	CrossEntropyLoss = _TorchRequired
