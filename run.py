import subprocess
import sys
from itertools import product

# Run the model for each week
# define the weeks
week = 3
print("CFNP_{}_week_ahead_prediction".format(week))

# Run the model
# We provide training weight files for easy reproduction
r = subprocess.call(
    [
        sys.executable,
        "TestSymp.py",
        "--week",
        str(week),
    ]
    )
