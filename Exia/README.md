# Exia

Exia is a torch-style Python library for machine learning and deep learning, built on top of Lightning Core.
It is designed for fast experimentation with:

- standalone Lightning Core workflows (no torch required)
- torch-native training workflows (optional)
- Hugging Face model and pipeline loading (optional)

## Install

From local workspace root:

```bash
python -m pip install -e ./Exia
```

Standalone core install from package index:

```bash
python -m pip install Exia
```

Torch mode install:

```bash
python -m pip install "Exia[torch]"
```

Torch + Hugging Face mode install:

```bash
python -m pip install "Exia[full]"
```

## Quick Start

```python
import Exia as ex

ex.set_backend("lightning")
x = ex.tensor([[1.0, 2.0], [3.0, 4.0]])
print(type(x), x.shape)

trainer = ex.Trainer(ex.TrainerConfig(epochs=100, log_every=20))
w, b = trainer.fit_linear_regression([[1.0], [2.0], [3.0]], [2.0, 4.0, 6.0])
print(w, b)
```

## Lightning Core Integration

```python
import Exia as ex

out = ex.lightning_vector_add([1, 2, 3], [4, 5, 6], device="metal")
print(out)
```

## Hugging Face Integration

Requires `Exia[hf]`.

```python
import Exia as ex

pipe = ex.load_hf_pipeline("sentiment-analysis", "distilbert-base-uncased-finetuned-sst-2-english")
print(pipe("Exia looks practical."))
```

## Minimal Deep Learning Training Example

Requires `Exia[torch]`.

See:

- examples/train_mlp.py
- examples/hf_inference.py
- examples/standalone_linear.py
- examples/standalone_classification.py

## Standalone ML Utilities

Core Exia (without torch) includes:

- train_test_split
- fit_linear_regression (with optional L2 regularization)
- fit_logistic_regression (binary classification)
- mse, mae, r2_score, accuracy

Example:

```python
import Exia as ex

ex.set_backend("lightning")

x = [[1.0], [2.0], [3.0], [4.0]]
y = [2.1, 3.9, 6.2, 8.1]

model = ex.fit_linear_regression(x, y, lr=0.05, epochs=600)
pred = model.predict(x)
print(ex.mse(y, pred), ex.r2_score(y, pred))
```

## Notes

- Exia defaults to `lightning` backend and works without torch.
- Use `ex.set_backend("torch")` to switch to torch mode when installed.
- You can also set backend by environment variable: `EXIA_BACKEND=lightning` or `EXIA_BACKEND=torch`.
- On macOS Apple Silicon, torch MPS and Lightning Core Metal paths can both be used.

Environment variable example:

```bash
EXIA_BACKEND=torch python your_script.py
```

If the value is invalid or torch is unavailable, Exia falls back to `lightning` backend.
