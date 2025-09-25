
import torch

def tensor_stats(x: torch.Tensor) -> str:
    n_nan = torch.isnan(x).sum().item()
    n_inf = torch.isinf(x).sum().item()
    max_abs = float(torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).abs().max().item()) if x.numel() > 0 else 0.0
    return f"shape={tuple(x.shape)}, nan={n_nan}, inf={n_inf}, max|x|={max_abs: .3e}"
