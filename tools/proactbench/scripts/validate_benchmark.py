"""Validate the expanded benchmark without installing dependencies."""

import argparse
import json
from benchmark import DEFAULT_DATA, load, validate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATA)
    args = parser.parse_args()
    print(json.dumps(validate(load(args.data)), indent=2))
