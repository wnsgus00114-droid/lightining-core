import importlib

import Exia as ex
import Exia.backend as backend


def test_env_selects_lightning_backend(monkeypatch):
    monkeypatch.setenv("EXIA_BACKEND", "lightning")
    importlib.reload(backend)
    assert backend.get_backend() == "lightning"


def test_env_invalid_value_falls_back_to_lightning(monkeypatch):
    monkeypatch.setenv("EXIA_BACKEND", "invalid-value")
    importlib.reload(backend)
    assert backend.get_backend() == "lightning"


def test_env_selects_torch_when_available(monkeypatch):
    if not ex.has_torch():
        return
    monkeypatch.setenv("EXIA_BACKEND", "torch")
    importlib.reload(backend)
    assert backend.get_backend() == "torch"
