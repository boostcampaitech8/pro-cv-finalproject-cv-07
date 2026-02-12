import os
import numpy as np
import random
import torch

def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    # Favor speed; deterministic mode slows training and isn't fully enforced here.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Avoid hard crash on unsupported deterministic ops
    try:
        # Deterministic algorithms can spam CuBLAS warnings and slow training.
        # Leave deterministic disabled by default; keep seeding for repeatability.
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass
    np.random.seed(seed)
    random.seed(seed)
