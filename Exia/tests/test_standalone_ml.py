import numpy as np
import Exia as ex


def test_linear_regression_fits_simple_line():
    ex.set_backend("lightning")
    x = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=np.float32)
    y = np.array([2.0, 4.1, 6.0, 8.1, 10.0], dtype=np.float32)

    model = ex.fit_linear_regression(x, y, lr=0.05, epochs=800)
    pred = model.predict(x)

    assert ex.mse(y, pred) < 0.03
    assert ex.r2_score(y, pred) > 0.995


def test_logistic_regression_binary_classification():
    ex.set_backend("lightning")
    rng = np.random.default_rng(11)
    x0 = rng.normal(loc=-1.5, scale=0.8, size=(150, 2))
    x1 = rng.normal(loc=1.5, scale=0.8, size=(150, 2))
    x = np.concatenate([x0, x1], axis=0)
    y = np.concatenate([np.zeros(150), np.ones(150)], axis=0)

    x_train, x_test, y_train, y_test = ex.train_test_split(x, y, test_size=0.3, seed=11)
    model = ex.fit_logistic_regression(x_train, y_train, lr=0.1, epochs=900)
    pred = model.predict(x_test)

    assert ex.accuracy(y_test, pred) > 0.95
