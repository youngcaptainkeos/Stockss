import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def config():
    return json.loads((ROOT / "experiment.json").read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(x):
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, (np.ndarray,)):
            return x.tolist()
        return str(x)

    path.write_text(json.dumps(data, indent=2, default=convert, allow_nan=False))
