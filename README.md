# VST-MIA Artifact

Artifact for **Seeing Membership in Federated Learning: Visual-Statistical Trajectory-based Membership Inference Attack**.

VST-MIA performs membership inference from the loss trajectory of a target sample across federated-learning checkpoints. The artifact preserves the original evaluation pipeline:

```text
federated training
    -> loss-trajectory construction
    -> VLM scoring
    -> ECD/TF extraction
    -> reverse-eCDF calibration
    -> adaptive score fusion
    -> AUC and low-FPR evaluation
```

The artifact-evaluation configuration is **STL10 + ResNet-56 + Qwen3-VL-2B-Instruct**.

## 1. Artifact scope and claims

This package supports evaluation of the following claims:

1. Intermediate federated-learning checkpoints can be used to construct per-sample loss trajectories.
2. A visual language model can assign a visual membership score to each rendered trajectory.
3. Early Cumulative Descent (ECD) and Tail Fluctuation (TF) provide complementary trajectory statistics.
4. Reverse empirical-CDF calibration and confidence-adaptive fusion refine the visual score.
5. The resulting membership scores can be evaluated using ROC AUC and TPR at low FPR values.

The package is intentionally focused on the end-to-end VST-MIA workflow. Baseline attacks, defenses, and the full set of datasets and ablations from the paper are outside the scope of this artifact.

### Reference configuration

| Component | Configuration |
|---|---|
| Dataset | STL10 |
| Victim model | ResNet-56 |
| Federated learning | FedAvg, 5 clients, 200 rounds, 2 local epochs |
| Optimizer | SGD, learning rate 0.01, batch size 64 |
| Visual model | Qwen3-VL-2B-Instruct |


## 2. Method summary

For each target sample, the code constructs a loss trajectory from saved model checkpoints and renders it as an image. The attack then combines VLM-based visual assessment with ECD/TF statistical refinement and adaptive score fusion. The implementation is provided in `attack.py`; definitions and methodological details follow the accompanying paper and rebuttal.

## 3. Requirements

### Hardware

- NVIDIA GPU with CUDA support;
- at least 8 GB of GPU memory for the 2B VLM; more memory is recommended;
- at least 12 GB of free disk space for the dataset, processed arrays, checkpoints, plots, and VLM weights.

Runtime depends strongly on the GPU, storage device, number of federated rounds, and number of evaluated samples.

### Software

- Python 3.9 or later; Python 3.10 is recommended;
- Linux with Bash, or Windows 10/11 with PowerShell 5.1 or later;
- a PyTorch build compatible with the installed CUDA driver.

Create an isolated environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the default PyTorch wheel does not match the local CUDA installation, install the appropriate PyTorch and torchvision wheels first, then install the remaining requirements.

## 4. Quick artifact check

Use the quick preset for an end-to-end functional check. This mode verifies execution of the artifact pipeline and is not intended for performance reproduction.

### Linux

```bash
bash run_stl10.sh --install --quick
```

### Windows

From Command Prompt or PowerShell:

```powershell
.\run_stl10.bat -Install -Quick
```

The wrappers perform five stages:

1. validate Python and required packages;
2. download or locate the dataset;
3. download or locate Qwen3-VL-2B-Instruct;
4. run preprocessing, federated training, and trajectory rendering;
5. run VST-MIA and write the evaluation report.

A successful execution produces:

```text
reports_lira_lite/
└── audit_details.json
```

It also produces training logs, model checkpoints, trajectory images, and serialized loss histories in the directories described below.

## 5. Full reference evaluation

### Linux

```bash
bash run_stl10.sh
```

### Windows

```powershell
.\run_stl10.bat
```

The default reference run uses 200 federated rounds, five clients, five participants per round, two local epochs, learning rate 0.01, and batch size 64.

To select a GPU explicitly on Linux:

```bash
CUDA_VISIBLE_DEVICES=1 bash run_stl10.sh
```

Otherwise, `main.py` selects the visible GPU with the most free memory. Numerical results may vary slightly across CUDA, cuDNN, PyTorch, GPU, and driver versions.

## 6. Reusing existing artifacts

The pipeline stages can be skipped independently. Skipped stages require their expected files to already exist.

### Run only the attack on Linux

```bash
bash run_stl10.sh \
  --skip-data \
  --skip-vlm \
  --no-data-process \
  --no-train \
  --no-plots
```

### Run only the attack on Windows

```powershell
.\run_stl10.bat -SkipData -SkipVlm -NoDataProcess -NoTrain -NoPlots
```

### Invoke the attack entry point directly

If trajectory images and `metrics_history_selected.pkl` already exist:

```bash
python attack.py \
  --dataset STL10 \
  --model resnet \
  --max_samples 1000 \
  --vlm_type qwen3_2b \
  --vlm_path ./vlm_2b
```

PowerShell:

```powershell
python .\attack.py --dataset STL10 --model resnet --max_samples 1000 --vlm_type qwen3_2b --vlm_path .\vlm_2b
```

The stage-control options have distinct meanings:

| Option | Meaning |
|---|---|
| `--skip-data` / `-SkipData` | Do not download the raw dataset. |
| `--skip-vlm` / `-SkipVlm` | Do not download the VLM checkpoint. |
| `--no-data-process` / `-NoDataProcess` | Reuse `datas/STL10/full.npz`. |
| `--no-train` / `-NoTrain` | Reuse checkpoints under `models_main/`. |
| `--no-plots` / `-NoPlots` | Reuse trajectory images and loss histories under `plot/vlm_data/`. |

## 7. Runner options

The STL10 Linux and Windows wrappers expose equivalent options.

| Linux | Windows | Default | Description |
|---|---|---:|---|
| `--rounds N` | `-Rounds N` | `200` | Number of federated rounds. |
| `--clients N` | `-Clients N` | `5` | Total number of clients. |
| `--participant N` | `-Participant N` | `5` | Participating clients per round. |
| `--epochs N` | `-Epochs N` | `2` | Local epochs per client. |
| `--lr F` | `-Lr F` | `0.01` | Learning rate. |
| `--batch-size N` | `-BatchSize N` | `64` | Batch size. |
| `--quick` | `-Quick` | off | Run the smoke-test preset. |
| `--skip-data` | `-SkipData` | off | Skip raw-dataset download. |
| `--skip-vlm` | `-SkipVlm` | off | Skip VLM download. |
| `--no-data-process` | `-NoDataProcess` | off | Reuse processed data. |
| `--no-train` | `-NoTrain` | off | Reuse trained checkpoints. |
| `--no-plots` | `-NoPlots` | off | Reuse trajectory inputs. |
| `--vlm-source S` | `-VlmSource S` | `auto` | `auto`, `modelscope`, or `hf`. |
| `--vlm-dir DIR` | `-VlmDir DIR` | `./vlm_2b` | Local VLM checkpoint directory. |
| `--install` | `-Install` | off | Install `requirements.txt` first. |

## 8. Manual execution

The wrapper scripts are recommended because they check dependencies and prepare external assets automatically. The same full STL10 workflow can be invoked manually:

```bash
python main.py \
  --dataset STL10 \
  --model resnet \
  --client_num 5 \
  --participant 5 \
  --training_round 200 \
  --epochs 2 \
  --lr 0.01 \
  --batch_size 64 \
  --data_process_flag True \
  --train_model True \
  --regenerate_plots \
  --method ours \
  --vlm_type qwen3_2b \
  --vlm_path ./vlm_2b
```

`--data_process_flag` and `--train_model` use the existing boolean interface in `main.py`. The wrapper scripts are the preferred way to enable or disable these stages.

## 9. Output files

| Path | Contents |
|---|---|
| `datas/STL10/full.npz` | Preprocessed dataset. |
| `datas/STL10/resnet/{data_split}/` | Per-client retained and held-out splits. |
| `models_main/resnet/STL10/server_model/` | Per-round server checkpoints. |
| `models_main/resnet/STL10/client_model/` | Saved client checkpoints. |
| `plot/vlm_data/resnet/STL10/member/` | Member trajectory images. |
| `plot/vlm_data/resnet/STL10/nonmember/` | Non-member trajectory images. |
| `plot/vlm_data/resnet/STL10/metrics_history_selected.pkl` | Serialized loss sequences. |
| `reports_lira_lite/audit_details.json` | Per-sample attack results. |
| `log_file/resnet/STL10/` | Training and evaluation logs. |

Each record in `audit_details.json` contains the sample identifier, ground-truth membership label, final score, VLM score, and extracted features. Records that receive Phase-2 refinement additionally include `p_ecd`, `p_tf`, and the combined statistical score; all records include `ecd` and `tf`.

Existing results in these directories may be overwritten by a new run. Back up any results that must be retained.

## 10. Repository structure

| File | Purpose |
|---|---|
| `main.py` | End-to-end pipeline entry point. |
| `attack.py` | VLM scoring, ECD/TF calibration, adaptive fusion, and evaluation. |
| `trajectory.py` | Loss-trajectory construction and rendering. |
| `train.py` | Federated-learning training and checkpoint generation. |
| `data_processing.py` | STL10 preprocessing. |
| `Data.py` | Dataset wrappers and client partitioning. |
| `CSModels.py` | Dataset/model configuration and model factory. |
| `normalModel.py` | ResNet and neural-network implementations. |
| `fix.py` | Merges member and non-member loss histories. |
| `utils.py` | Random seeding, path handling, and NPZ loading. |
| `run.sh` | Linux pipeline runner used by `run_stl10.sh`. |
| `run_stl10.sh` | Linux reference-configuration wrapper. |
| `run.ps1` | Windows pipeline runner used by `run_stl10.bat`. |
| `run_stl10.bat` | Windows reference-configuration launcher. |

## 11. Reproducibility notes

- Data preparation and federated training use `--random_seed 123` by default.
- Attack evaluation uses `RANDOM_SEED = 42` in `attack.py`.
- `utils.set_seed()` seeds Python, NumPy, PyTorch, and CUDA RNGs.
- GPU kernels and floating-point reductions can still introduce small cross-machine differences.
- The VLM prompt is fixed in `attack.py` and is not modified by the runner scripts.
- For a controlled comparison, keep the software environment, GPU visibility, dataset files, VLM checkpoint, and command-line arguments unchanged.

## 12. Troubleshooting

### STL10 download returns an HTTP error

The STL10 host may be temporarily unavailable. Download the official archive manually, place and extract it under `datas/stl10/`, and rerun with `--skip-data` or `-SkipData`.

### CUDA out of memory

Close other GPU workloads and confirm that the 2B VLM is being loaded on the intended device. `device_map="auto"` is used for VLM loading. A GPU with more than 8 GB of memory is recommended for a less constrained run.

### VLM download fails

The runner tries ModelScope and Hugging Face when `--vlm-source auto` is used. You may select one explicitly:

```bash
bash run_stl10.sh --vlm-source modelscope
bash run_stl10.sh --vlm-source hf
```

Alternatively, download `Qwen/Qwen3-VL-2B-Instruct` manually into `./vlm_2b` and rerun with `--skip-vlm`.

### A skipped stage reports missing files

Each skip option assumes the corresponding outputs already exist. Rerun without the relevant skip option, or restore the expected files under `datas/`, `models_main/`, or `plot/vlm_data/`.

## 13. External assets

The dataset and VLM weights are not redistributed with this repository:

- STL10 is obtained through `torchvision.datasets.STL10`;
- Qwen3-VL-2B-Instruct is obtained from ModelScope or Hugging Face.

The licenses and terms of the corresponding upstream assets apply. For an **Artifacts Available** badge, publish the final artifact release in a persistent public repository and add its DOI or permanent archive identifier to the submission materials.

## 14. Citation

Citation metadata for this artifact is provided in [`CITATION.cff`](CITATION.cff). Please also cite the associated paper when using this implementation in academic work.

## 15. License

The source code in this repository is released under the [MIT License](LICENSE). Third-party datasets and model weights remain subject to their respective licenses and terms.
