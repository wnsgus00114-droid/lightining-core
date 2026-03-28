try:
	import torch
except Exception:
	torch = None


class _TorchRequired:
	def __init__(self, *args, **kwargs):
		raise RuntimeError("This API requires torch. Install with: pip install Exia[torch]")


if torch is not None:
	AdamW = torch.optim.AdamW
	SGD = torch.optim.SGD
else:
	AdamW = _TorchRequired
	SGD = _TorchRequired
