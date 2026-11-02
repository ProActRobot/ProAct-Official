"""Single-file and batch conversion with deterministic output names."""

from .pipeline import convert
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Retarget BVH motion to Unitree G1 joint angles")
    parser.add_argument("input", type=Path, help="BVH file or directory of BVH files")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model", type=Path, default=ROOT / "assets/unitree_g1/scene_empty_coll.xml"
    )
    parser.add_argument("--hierarchy", type=Path, default=ROOT / "hierarchy_g1.bvh")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--visual", action="store_true")
    parser.add_argument("--smooth-sigma", type=float, default=0)
    parser.add_argument("--frequency", type=float, default=1000)
    args = parser.parse_args()
    if args.workers < 1 or args.frequency <= 0 or args.smooth_sigma < 0:
        parser.error("workers/frequency must be positive; sigma must be nonnegative")
    sources = sorted(args.input.glob("*.bvh")) if args.input.is_dir() else [args.input]
    if not sources or any((not path.is_file() for path in sources)):
        parser.error("No BVH input files found")
    if args.visual and (len(sources) != 1 or args.workers != 1):
        parser.error("--visual requires one input and one worker")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (
            path,
            args.output_dir,
            args.model,
            args.hierarchy,
            args.visual,
            args.smooth_sigma,
            args.frequency,
        )
        for path in sources
    ]
    if args.workers == 1:
        for job in jobs:
            print(convert(job))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(convert, jobs):
                print(result)


if __name__ == "__main__":
    main()
