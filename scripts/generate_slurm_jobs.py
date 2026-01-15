#!/usr/bin/env python3
import argparse
import re
from pathlib import Path


TEMPLATE = """#!/bin/bash
#SBATCH --nodes={nodes}                 # default; override at submit: sbatch -N <N>
#SBATCH --gres=gpu:{gpus_per_node}              # default; override: sbatch --gres=gpu:<G>
#SBATCH --output={logs_root}/{exp_tag}_%j/slurm_%j.out
#SBATCH --error={logs_root}/{exp_tag}_%j/slurm_%j.err
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --time={time}
#SBATCH --partition={partition}
#SBATCH --exclusive

# Pick up actual allocation if you override at submit time
NODES=${{SLURM_NNODES:-1}}
GPUS_PER_NODE={gpus_per_node}
module purge
{module_lines}

echo "Using: NODES=$NODES  GPUS_PER_NODE=$GPUS_PER_NODE"
echo "Dataset: {dataset_name}"

STORE_DIR={store_dir_root}/$SLURM_JOB_ID
mkdir -p "$STORE_DIR"
export INIT_METHOD="file://$STORE_DIR/store"
# Do NOT set MASTER_ADDR/MASTER_PORT when using file://
srun --cpu-bind=cores --ntasks-per-node=${{GPUS_PER_NODE}} --gpus-per-node=${{GPUS_PER_NODE}} --kill-on-bad-exit=1 bash -lc '
  export NCCL_DEBUG=warn TORCH_NCCL_ASYNC_ERROR_HANDLING=1 PYTHONUNBUFFERED=1;
  export CUDA_DEVICE_ORDER=PCI_BUS_ID 
  export CUDA_VISIBLE_DEVICES=0,1,2,3
  unset NCCL_ASYNC_ERROR_HANDLING
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export NUMEXPR_NUM_THREADS=1
  export RANK=$SLURM_PROCID
  export WORLD_SIZE=$SLURM_NTASKS
  export LOCAL_RANK=$SLURM_LOCALID
  export DATA_YAML={data_yaml}
  ulimit -n 65536
  export TMPDIR={tmpdir}  
  mkdir -p "$TMPDIR"
  VENV={venv}
  source "$VENV/bin/activate"
  export PYTHONPATH="{pyproject_root}:$VENV/lib/python{python_version}/site-packages:$PYTHONPATH"
  export PYTHONNOUSERSITE=1
#  which python  
#  python -c "import sys, thop; print(sys.executable); print("thop ok")"
#  python -c "import huggingface_hub; print(f"huggingface-hub version: {huggingface_hub.__version__}")"
  {run_cmd}
'
"""


def extract_exp_tag(yaml_path: Path) -> str:
    """
    В шаблоне папка логов выглядит как "10_%j".
    Берём первое число после 'coco_' (например, coco_10_p.yaml -> '10').
    Если не совпало — используем stem.
    """
    m = re.search(r"coco_(\d+)", yaml_path.stem)
    return m.group(1) if m else yaml_path.stem


def iter_yaml_files(yamls_root: Path):
    """
    Если есть подпапки p/random/topn — берём из них.
    Иначе — берём coco_*.yaml прямо из yamls_root.
    """
    subdirs = ["p", "random", "topn"]
    has_strat = any((yamls_root / s).is_dir() for s in subdirs)

    if has_strat:
        for strat in subdirs:
            d = yamls_root / strat
            if not d.is_dir():
                continue
            for y in sorted(d.glob("coco_*.yaml")):
                yield strat, y
    else:
        for y in sorted(yamls_root.glob("coco_*.yaml")):
            yield None, y


def main():
    ap = argparse.ArgumentParser(description="Generate SLURM job scripts from coco_*.yaml using a fixed template.")

    ap.add_argument("--yamls-root", required=True, help="Root directory with coco_*.yaml (optionally with p/random/topn subdirs).")
    ap.add_argument("--out-root", required=True, help="Where to write generated *.slurm files.")

    ap.add_argument("--partition", default="boost_usr_prod")
    ap.add_argument("--nodes", type=int, default=4)
    ap.add_argument("--gpus-per-node", type=int, default=4)
    ap.add_argument("--cpus-per-task", type=int, default=8)
    ap.add_argument("--time", default="04:00:00")

    ap.add_argument("--logs-root", default="run_logs/LowData")

    ap.add_argument(
        "--module-lines",
        default="module load profile/deeplrn\nmodule load cineca-ai/4.3.0",
        help="Lines inserted after `module purge` (use literal newlines or escape with \\n).",
    )

    ap.add_argument("--store-dir-root", default="/leonardo_work/EUHPC_D18_074/abugajev/tmp/ddp")
    ap.add_argument("--tmpdir", default="/leonardo_work/EUHPC_D18_074/abugajev/tmp")
    ap.add_argument("--venv", default="/leonardo_work/EUHPC_D18_074/abugajev/yolomulti/env")
    ap.add_argument("--pyproject-root", default="/leonardo_work/EUHPC_D18_074/abugajev/yolomulti/yolov12-main")
    ap.add_argument("--python-version", default="3.11")

    ap.add_argument(
        "--run-cmd",
        default="python -u /leonardo_work/EUHPC_D18_074/abugajev/yolomulti/runmulti.py",
        help='Command placed as-is into the template (inside srun bash -lc block).',
    )

    args = ap.parse_args()

    yamls_root = Path(args.yamls_root).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    module_lines = args.module_lines.replace("\\n", "\n").rstrip("\n")

    total = 0
    for strat, y in iter_yaml_files(yamls_root):
        exp_tag = extract_exp_tag(y)

        text = TEMPLATE.format(
            nodes=args.nodes,
            gpus_per_node=args.gpus_per_node,
            logs_root=args.logs_root.rstrip("/"),
            exp_tag=exp_tag,
            cpus_per_task=args.cpus_per_task,
            time=args.time,
            partition=args.partition,
            module_lines=module_lines,
            dataset_name=y.name,
            store_dir_root=args.store_dir_root.rstrip("/"),
            data_yaml=str(y),
            tmpdir=args.tmpdir,
            venv=args.venv,
            pyproject_root=args.pyproject_root,
            python_version=args.python_version,
            run_cmd=args.run_cmd,
        )

        target_dir = (out_root / strat) if strat else out_root
        target_dir.mkdir(parents=True, exist_ok=True)

        out_path = target_dir / f"{y.stem}.slurm"
        out_path.write_text(text, encoding="utf-8")
        total += 1

    print(f"[OK] Generated {total} SLURM jobs under: {out_root}")
    print(f"     YAMLs source: {yamls_root}")
    print(f"     Logs root: {args.logs_root!r}")


if __name__ == "__main__":
    main()
