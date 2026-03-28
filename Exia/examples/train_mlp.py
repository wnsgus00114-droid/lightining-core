import torch
from torch.utils.data import DataLoader, TensorDataset

import Exia as ex


class MLP(ex.Module):
    def __init__(self):
        super().__init__()
        self.net = ex.Sequential(
            ex.Linear(8, 32),
            ex.GELU(),
            ex.Linear(32, 2),
        )

    def forward(self, x):
        return self.net(x)


def main():
    x = torch.randn(512, 8)
    y = torch.randint(0, 2, (512,))
    ds = TensorDataset(x, y)
    dl = DataLoader(ds, batch_size=32, shuffle=True)

    model = MLP()
    optimizer = ex.AdamW(model.parameters(), lr=1e-3)
    loss_fn = ex.CrossEntropyLoss()

    trainer = ex.Trainer(ex.TrainerConfig(epochs=2, log_every=10))
    trainer.fit(model, dl, optimizer, loss_fn)


if __name__ == "__main__":
    main()
