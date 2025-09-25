
import time
from typing import Optional

def _now_str() -> str:
    return time.strftime("%H:%M:%S", time.localtime()) + f".{int((time.time()%1)*1000):03d}"

def log(msg: str, step: Optional[int] = None) -> None:
    if step is not None:
        print(f"[{_now_str()}] Batch {step}: {msg}")
    else:
        print(f"[{_now_str()}] {msg}")
