import json
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import openai


def main() -> None:
    info = {
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "openai": openai.__version__,
        "timestamp": datetime.utcnow().isoformat(),
    }
    with open("results/env.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


if __name__ == "__main__":
    main()
