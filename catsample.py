import torch
import torch.nn.functional as F


def gumbel_softmax(categorical_probs, hard=False, eps=1e-9):
    logits = categorical_probs.clamp(min=1e-9).log()
    return F.gumbel_softmax(logits, hard=hard)


def sample_categorical(categorical_probs, method="hard",generator=None):
    if method == "hard":
        shape = categorical_probs.shape
        device = categorical_probs.device
        dtype = categorical_probs.dtype
        if generator is None:
            uniform = torch.rand(shape, device=device, dtype=dtype)
        else:
            uniform = torch.rand(shape, device=device, dtype=dtype, generator=generator)
        gumbel_norm = 1e-10 - (uniform + 1e-10).log()
        return (categorical_probs / gumbel_norm).argmax(dim=-1)
    else:
        raise ValueError(f"Method {method} for sampling categorical variables is not valid.")
    