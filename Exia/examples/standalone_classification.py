import numpy as np
import Exia as ex


def main():
    ex.set_backend("lightning")

    rng = np.random.default_rng(7)
    x0 = rng.normal(loc=-1.5, scale=0.7, size=(200, 2))
    x1 = rng.normal(loc=1.5, scale=0.7, size=(200, 2))
    x = np.concatenate([x0, x1], axis=0)
    y = np.concatenate([np.zeros(200), np.ones(200)], axis=0)

    x_train, x_test, y_train, y_test = ex.train_test_split(x, y, test_size=0.25, seed=7)
    clf = ex.fit_logistic_regression(x_train, y_train, lr=0.1, epochs=700)

    pred = clf.predict(x_test)
    acc = ex.accuracy(y_test, pred)

    print("test_accuracy:", round(acc, 4))
    print("sample_prob:", np.round(clf.predict_proba([[0.0, 0.0]]), 4))


if __name__ == "__main__":
    main()
