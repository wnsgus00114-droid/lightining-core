import Exia as ex


def main():
    ex.set_backend("lightning")

    x = [[1.0], [2.0], [3.0], [4.0], [5.0]]
    y = [2.0, 4.0, 6.0, 8.0, 10.0]

    trainer = ex.Trainer(ex.TrainerConfig(epochs=200, log_every=50))
    w, b = trainer.fit_linear_regression(x, y, lr=0.05)

    print("weights:", w)
    print("bias:", b)
    print("vector add:", ex.lightning_vector_add([1, 2, 3], [4, 5, 6]))


if __name__ == "__main__":
    main()
