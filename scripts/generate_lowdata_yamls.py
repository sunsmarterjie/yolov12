#!/usr/bin/env python3
import argparse
from pathlib import Path

COCO80 = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle",
    "wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch","potted plant","bed",
    "dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

def find_repo_root(start: Path) -> Path:
    """
    Try to locate repo root by walking upwards until a folder containing `datasets/` is found.
    Falls back to current working directory if not found.
    """
    for p in [start] + list(start.parents):
        if (p / "datasets").is_dir():
            return p
    return start

def write_yaml(out_path: Path, coco_path: str, train_list: str, val_list: str, test_list: str | None):
    lines = [
        f"path: {coco_path}",
        f"train: {train_list}",
        f"val: {val_list}",
    ]
    if test_list:
        lines.append(f"test: {test_list}")

    lines += ["", "# Classes", "names:"]
    for i, name in enumerate(COCO80):
        lines.append(f"  {i}: {name}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def strategy_of_list(filename: str) -> str | None:
    # Expected names: train2017_<percent>_p.txt / _random.txt / _topn.txt
    if filename.endswith("_p.txt"):
        return "p"
    if filename.endswith("_random.txt"):
        return "random"
    if filename.endswith("_topn.txt"):
        return "topn"
    return None

def main():
    ap = argparse.ArgumentParser(
        description="Generate YOLO dataset YAMLs for all train2017_<percent>_{p,random,topn}.txt subset lists.\n"
                    "Can be run with no arguments; outputs go to scripts/low_data/yamls/{p,random,topn}."
    )

    # Optional overrides (defaults are repo-relative)
    ap.add_argument("--repo-root", default=None, help="Path to yolov12_SmallData root (auto-detected if omitted).")
    ap.add_argument("--coco-path", default="./", help="Value for YAML field `path:` (default: ./).")
    ap.add_argument("--lists-dir", default=None, help="Override lists dir (default: <repo>/datasets).")
    ap.add_argument("--out-root", default=None, help="Override output root (default: <repo>/scripts/low_data/yamls).")
    ap.add_argument("--val", default="val2017.txt", help="Validation list filename (resolved under coco-path).")
    ap.add_argument("--test", default="test-dev2017.txt", help="Test list filename (optional; resolved under coco-path).")
    ap.add_argument("--prefix", default="train2017_", help="Only include list files starting with this prefix.")
    args = ap.parse_args()

    start = Path(__file__).resolve()
    repo_root = Path(args.repo_root).resolve() if args.repo_root else find_repo_root(start.parent)

    lists_dir = Path(args.lists_dir).resolve() if args.lists_dir else (repo_root / "datasets")
    out_root = Path(args.out_root).resolve() if args.out_root else (repo_root / "scripts" / "low_data" / "yamls")

    if not lists_dir.is_dir():
        raise SystemExit(f"Lists dir not found: {lists_dir}")

    out_root.mkdir(parents=True, exist_ok=True)
    for strat in ("p", "random", "topn"):
        (out_root / strat).mkdir(parents=True, exist_ok=True)

    subset_lists = sorted(p for p in lists_dir.glob(f"{args.prefix}*.txt") if p.is_file())
    if not subset_lists:
        raise SystemExit(f"No subset lists found in: {lists_dir}")

    count = 0
    for p in subset_lists:
        strat = strategy_of_list(p.name)
        if strat is None:
            # skip train2017.txt or anything not matching *_p/_random/_topn
            continue

        # train2017_10_p.txt -> coco_10_p.yaml
        stem = p.name.replace("train2017_", "").replace(".txt", "")
        yaml_name = f"coco_{stem}.yaml"
        out_path = out_root / strat / yaml_name

        write_yaml(
            out_path=out_path,
            coco_path=args.coco_path,
            train_list=p.name,    # YAML expects this file under coco_path directory
            val_list=args.val,
            test_list=args.test if args.test else None,
        )
        count += 1

    print(f"[OK] Generated {count} YAMLs under: {out_root}")
    print(f"     Lists source: {lists_dir}")
    print(f"     YAML path field set to: {args.coco_path!r} (override with --coco-path ...)")

if __name__ == "__main__":
    main()
