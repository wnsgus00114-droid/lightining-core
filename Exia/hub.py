from __future__ import annotations

try:
    import torch
except Exception:
    torch = None


def _auto_device_for_pipeline() -> int:
    if torch is not None and torch.cuda.is_available():
        return 0
    return -1


def load_hf_pipeline(task: str, model_id: str, device: int | None = None):
    try:
        from transformers import pipeline
    except Exception as exc:
        raise RuntimeError("Hugging Face integration requires transformers. Install with: pip install Exia[hf]") from exc
    selected_device = _auto_device_for_pipeline() if device is None else device
    return pipeline(task=task, model=model_id, device=selected_device)


def load_hf_model_and_tokenizer(model_id: str, num_labels: int | None = None):
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as exc:
        raise RuntimeError("Hugging Face integration requires transformers. Install with: pip install Exia[hf]") from exc
    kwargs = {"num_labels": num_labels} if num_labels is not None else {}
    model = AutoModelForSequenceClassification.from_pretrained(model_id, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer
