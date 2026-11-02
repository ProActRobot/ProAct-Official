"""Project attributed speech from the expanded JSON, excluding stage directions."""

import argparse
from benchmark import DEFAULT_DATA, load, transcripts, write_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    write_json(args.output, transcripts(load(args.data)))
