#!/usr/bin/env python3
import argparse
from pathlib import Path

TEMPLATE = """#!/bin/bash
#SBATCH --nodes={nodes}
#SBATCH --gres=gpu:{gpus_per_node}
#SBATCH --output={logs_root}/{exp_dir}/slurm_%j.out
#SBATCH --error={logs_root}/{exp_dir}/slurm_%j.err
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --time={time}
#SBATCH --partition={partition}
#SBATCH --exclusive

NODES=${{SLURM_NNODES:-1}}
GPUS_PER_NODE={gpus_per_node}

module purge
{module_lines}

echo "Using: NODES=$NODES  GPUS_PER_NODE=$GPUS_PER_NODE"
echo "Dataset YAML: {yaml_path}"

STORE_DIR=${{SLURM_TMPDIR:-/tmp}}/ddp_store_${{SLURM_JOB_ID}}
mkdir -p "$STORE_DIR"
export INIT_METHOD="file://$STORE_DIR/store"

srun --cpu-bind=cores --ntasks-per-node=${{GPUS_PER_NODE}} --gpus-per-node=${{GPUS_PER_NODE}} --kill-on-bad-exit=1 bash -lc '
  export NCCL_DEBUG=warn PYTHONUNBUFFERED=1;
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES=0,1,2,3

  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1

  export RANK=$SLURM_PROCID
  export WORLD_SIZE=$SLURM_NTASKS
  export LOCAL_RANK=$SLURM_LOCALID

  # The training code reads this environment variable
  export DATA_YAML="{yaml_path}"

  ulimit -n 65536

  {pre_cmd}

  {run_cmd}
'
"""

def find_repo_root(start: Path) -> Path:
    """Walk upwards until we find a folder that looks like the repo root (contains datasets/)."""
    for p in [start] + list(start.parents):
        if (p / "datasets").is_dir():
            return p
    return start

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Generate SLURM job scripts for all coco_*.yaml under:\n"
            "  scripts/low_data/yamls/{p,random,topn}\n"
            "Outputs jobs into:\n"
            "  scripts/low_data/slurm/{p,random,topn}\n\n"
            "Run with minimal args (partition + run-cmd). No cluster paths are hardcoded."
        )
    )

    # Required (cluster-specific) arguments — intentionally required to avoid hardcoding.
    ap.add_argument("--partition", required=True, help="SLURM partition name on your cluster.")
    ap.add_argument(
        "--run-cmd",
        required=True,
        help='Training command executed inside srun, e.g. "python -u /path/to/runmulti.py".'
    )

    # Optional overrides (defaults are repo-relative)
    ap.add_argument("--repo-root", default=None, help="Repo root (auto-detected if omitted).")
    ap.add_argument("--yamls-root", default=None, help="Override YAML root dir.")
    ap.add_argument("--out-root", default=None, help="Override SLURM output root dir.")

    # SLURM config knobs (defaults match your typical setup but are safe/generic)
    ap.add_argument("--logs-root", default="run_logs/LowData", help="Root folder for stdout/stderr logs.")
    ap.add_argument("--time", default="04:00:00")
    ap.add_argument("--nodes", type=int, default=4)
    ap.add_argument("--gpus-per-node", type=int, default=4)
    ap.add_argument("--cpus-per-task", type=int, default=8)

    ap.add_argument(
        "--module-lines",
        default="module load profile/deeplrn\nmodule load cineca-ai/4.3.0",
        help="Lines inserted after `module purge` (use literal newlines or escape with \\n).",
    )
    ap.add_argument(
        "--pre-cmd",
        default="",
        help="Optional shell lines before run_cmd inside the srun bash -lc block (escape newlines with \\n).",
    )

    args = ap.parse_args()

    this_file = Path(__file__).resolve()
    repo_root = Path(args.repo_root).resolve() if args.repo_root else find_repo_root(this_file.parent)

    yamls_root = Path(args.yamls_root).resolve() if args.yamls_root else (
        repo_root / "scripts" / "low_data" / "yamls"
    )
    out_root = Path(args.out_root).resolve() if args.out_root else (
        repo_root / "scripts" / "low_data" / "slurm"
    )

    module_lines = args.module_lines.replace("\\n", "\n")
    pre_cmd = args.pre_cmd.replace("\\n", "\n").strip()
    if pre_cmd:
        pre_cmd = pre_cmd + "\n"

    out_root.mkdir(parents=True, exist_ok=True)
    for strat in ("p", "random", "topn"):
        (out_root / strat).mkdir(parents=True, exist_ok=True)

    total = 0
    for strat in ("p", "random", "topn"):
        strat_yaml_dir = yamls_root / strat
        if not strat_yaml_dir.is_dir():
            print(f"[WARN] YAML dir missing, skipping: {strat_yaml_dir}")
            continue

        yamls = sorted(strat_yaml_dir.glob("coco_*.yaml"))
        if not yamls:
            print(f"[WARN] No YAMLs found in: {strat_yaml_dir}")
            continue

        for y in yamls:
            exp_name = y.stem  # e.g. coco_10_p
            exp_dir = f"{strat}/{exp_name}_%j"

            text = TEMPLATE.format(
                nodes=args.nodes,
                gpus_per_node=args.gpus_per_node,
                cpus_per_task=args.cpus_per_task,
                time=args.time,
                partition=args.partition,
                logs_root=args.logs_root.rstrip("/"),
                exp_dir=exp_dir,
                yaml_path=str(y),
                module_lines=module_lines,
                pre_cmd=pre_cmd,
                run_cmd=args.run_cmd,
            )

            out_path = out_root / strat / f"{exp_name}.slurm"
            out_path.write_text(text, encoding="utf-8")
            total += 1

    print(f"[OK] Generated {total} SLURM jobs under: {out_root}")
    print(f"     YAMLs source: {yamls_root}")
    print(f"     Logs root: {args.logs_root!r}")

if __name__ == "__main__":
    main()
