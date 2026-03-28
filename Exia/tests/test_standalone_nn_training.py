import numpy as np
import os

import Exia as ex


def _make_batches(x, y, batch_size=32):
    batches = []
    for i in range(0, x.shape[0], batch_size):
        batches.append((x[i : i + batch_size], y[i : i + batch_size]))
    return batches


def test_standalone_nn_classifier_learns_without_torch():
    if os.getenv("EXIA_BACKEND", "").strip().lower() == "torch":
        return
    if ex.get_backend() != "lightning":
        return

    rng = np.random.default_rng(13)
    x0 = rng.normal(loc=-1.2, scale=0.6, size=(180, 2)).astype(np.float32)
    x1 = rng.normal(loc=1.2, scale=0.6, size=(180, 2)).astype(np.float32)
    x = np.concatenate([x0, x1], axis=0)
    y = np.concatenate([np.zeros(180, dtype=np.int64), np.ones(180, dtype=np.int64)], axis=0)

    idx = rng.permutation(x.shape[0])
    x = x[idx]
    y = y[idx]

    split = int(0.8 * x.shape[0])
    x_train, y_train = x[:split], y[:split]
    x_test, y_test = x[split:], y[split:]

    dataloader = _make_batches(x_train, y_train, batch_size=48)

    model = ex.Sequential(ex.Linear(2, 16), ex.ReLU(), ex.Linear(16, 2))
    optimizer = ex.AdamW(model.parameters(), lr=0.03, weight_decay=1e-4)
    loss_fn = ex.CrossEntropyLoss()

    trainer = ex.Trainer(ex.TrainerConfig(epochs=55, log_every=1000))
    trainer.fit(model, dataloader, optimizer, loss_fn)

    logits = model(x_test)
    pred = np.argmax(logits, axis=1)
    acc = ex.accuracy(y_test, pred)

    assert acc > 0.93
