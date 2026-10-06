# VST-MIA: ACSAC 2026 Artifact Evaluation

Artifact for **Seeing Membership in Federated Learning: Visual-Statistical Trajectory-based Membership Inference Attack**.

**Start here:** install the dependencies in Section 2, run `bash run_stl10.sh --quick`, and check the success criteria in Section 5. The quick reference check takes approximately **5 minutes** on a single **NVIDIA RTX 3090 (24 GB)** after the dataset and VLM weights are available locally. The full reference evaluation takes approximately **40 minutes** under the same conditions.

The reference configuration is **STL10 + ResNet-56 + Qwen3-VL-2B-Instruct**. Inference runs locally; no paid API, graphical desktop, or commercial software license is required.

Source repository: [Southwestedchen/VSTMIA](https://github.com/Southwestedchen/VSTMIA).

## 1. Scope and reference configuration

The artifact implements this workflow:

```text
federated training -> checkpoint-based loss trajectories -> trajectory images
    -> local VLM scoring -> ECD/TF extraction -> reverse-eCDF calibration
    -> adaptive score fusion -> ROC AUC and low-FPR TPR evaluation
```

| Component | Reference setting |
| --- | --- |
| Dataset / victim model | STL10 / ResNet-56 |
| Federated learning | FedAvg; 5 clients; all 5 participate per round |
| Full / quick training | 200 / 10 rounds; 2 local epochs per client |
| Optimizer | SGD; learning rate 0.01; batch size 64 |
| Partition / split ratio | Uniform client partition; split ratio 0.5 |
| Visual model | Qwen3-VL-2B-Instruct, loaded from `./vlm_2b` |
| Seeds | 123 for data/training/trajectories; 42 for attack evaluation |
| Observed evaluation set | 1,000 samples: 500 members and 500 non-members |

The full evaluation covers this reference workflow. The complete set of datasets, baselines, defenses, and ablations in the paper is outside this package's scope. The quick path checks end-to-end execution and is not intended to reproduce the paper's attack performance.

## 2. Requirements and installation

### Hardware and external services

| Resource | Requirement or reference information |
| --- | --- |
| CPU | Standard x86-64 host; the measured host reported 24 logical CPUs. No special CPU model is required. |
| System RAM | Plan for at least 8 GB, with additional headroom for the OS and other workloads. The sampled process-tree RAM peak was approximately 4.1 GiB in the full reference run; this is an observation, not a validated minimum. |
| GPU | An NVIDIA CUDA-capable GPU. Both measured runs used a single RTX 3090 with 24 GB VRAM. |
| GPU memory | At least 8 GB is recommended for the reference 2B VLM; the validated device had 24 GB. Lower-memory configurations were not measured. |
| Disk | Allow at least **15 GB for artifact files**, plus separate space for the Python environment, package caches, and temporary downloads. The observed full-run artifact directories occupied approximately 14.8 GB. |
| GUI | Not required. The workflow runs from a terminal and saves plots to files. |
| Network | Required for dependency installation and initial STL10/VLM downloads. Existing local assets can be reused. |
| Paid API / credentials | No paid API is used by the reference local-Qwen path. An OpenAI API key is not required. |
| Commercial software | Not required. |
| Licensing | Artifact source: MIT. Qwen weights: Apache-2.0 according to the [official model card](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct). STL10 remains subject to the terms provided by its [dataset authors](https://cs.stanford.edu/~acoates/stl10/). The source-code license does not replace third-party asset terms. |

GB means decimal gigabytes; GiB means binary gibibytes. Resource figures refer to the reference configuration and can change with software versions, workload, and reuse of existing files. Eight-GB RAM and VRAM figures are planning recommendations, not claims that this hardware configuration was independently validated.

### Operating systems and software

- **Validated:** Linux x86-64 with Bash, Python 3.10.19, and a CUDA-enabled PyTorch build.
- **Also provided:** Windows 10/11 launchers using PowerShell 5.1 or later. Windows runtime and resource measurements are not included here.
- **Python 3.10 or later** is required by the current source's type annotations. Python 3.10 is the measured reference version.

The following versions were recorded in both successful reference runs:

| Component | Recorded version |
| --- | --- |
| Linux kernel / glibc | 6.11.0-17-generic / 2.39 |
| NVIDIA driver | 580.173.02 |
| Python | 3.10.19 |
| PyTorch / torchvision | 2.5.1+cu121 / 0.20.1+cu121 |
| PyTorch CUDA build / cuDNN | 12.1 / 9.1.0 (`90100`) |
| Transformers / Accelerate | 5.8.0 / 1.12.0 |
| ModelScope / huggingface_hub | 1.34.0 / 1.14.0 |
| NumPy / SciPy / scikit-learn | 2.2.6 / 1.15.3 / 1.7.2 |
| pandas / Matplotlib / seaborn | 2.3.3 / 3.10.8 / 0.13.2 |
| tqdm / Pillow | 4.67.3 / 12.0.0 |

These are observed versions, not a promise of identical results across all versions satisfying `requirements.txt`. The requirements file specifies lower bounds rather than a complete environment lock.

### Linux installation

Run the commands from the repository root using Python 3.10 or later:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

The PyTorch command follows the [official instructions for the CUDA 12.1 build](https://pytorch.org/get-started/previous-versions/#v251). Use a compatible CUDA-enabled PyTorch build if your driver requires a different configuration. The recorded CUDA version is the PyTorch build version; the reference workflow does not compile custom CUDA extensions.

Check the environment before running:

```bash
python -c "import torch, transformers; from transformers import Qwen3VLForConditionalGeneration; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('transformers:', transformers.__version__)"
```

CUDA should be available for the GPU reference workflow, and the Qwen3-VL class should import successfully.

### Windows installation

From PowerShell in the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

## 3. Quick check and full evaluation

### Linux

With dependencies already installed:

```bash
# Minimal end-to-end check: 10 training rounds
bash run_stl10.sh --quick

# Full reference evaluation: 200 training rounds
bash run_stl10.sh
```

Alternatively, install dependencies through the runner before the quick check:

```bash
bash run_stl10.sh --install --quick
```

The runner locates or downloads the raw dataset and VLM weights, preprocesses STL10, trains federated models, renders trajectories, and evaluates the attack. `--quick` changes only the training round count to 10; client counts, local training settings, plotting settings, and attack sample limits remain unchanged. The measured quick and full runs both processed 1,000 attack samples.

Use `run_stl10.sh` for this reference configuration. The generic `run.sh` defaults to the separate `location + nn` configuration.

### Windows

```powershell
.\run_stl10.bat -Quick
.\run_stl10.bat
```

Use `-Install -Quick` if dependency installation is also needed. The measurements below were collected on Linux.

## 4. Measured runtime and resource use

One successful quick run and one successful full run were measured on **October 6, 2026**, using a **single RTX 3090 (24 GB)** and the software versions in Section 2. Raw STL10 data and Qwen weights were already local. Data preprocessing, training, trajectory generation, and attack evaluation were all enabled.

| Evaluation path | Rounds | Measured total | Planning estimate |
| --- | ---: | ---: | ---: |
| Quick check: `bash run_stl10.sh --quick` | 10 | 286.03 s (4 min 46 s) | Approximately 5 minutes |
| Full reference: `bash run_stl10.sh` | 200 | 2,316.78 s (38 min 37 s) | Approximately 40 minutes |

The totals include environment checks, Python startup, and computation. They exclude initial dependency installation and dataset/model downloads. Download time depends on the upstream service and network and is not assigned a fixed estimate. These are individual observations, not averages or upper bounds.

| Computation stage | Quick | Full |
| --- | ---: | ---: |
| Preprocessing | 6.25 s | 6.19 s |
| Federated training | 76.19 s | 1,524.23 s |
| Trajectory construction and rendering | 95.45 s | 678.02 s |
| Attack, including VLM loading, scoring, refinement and evaluation | 96.76 s | 96.40 s |

The attack stage includes approximately 95 seconds of visual scoring in each run. It is not divided by 20 in quick mode because the evaluated sample count is unchanged. The attack timing includes VLM loading, scoring, refinement and evaluation.

| Observed resource | Quick | Full |
| --- | ---: | ---: |
| Sampled process-tree RAM peak | 3.56 GiB | 4.05 GiB |
| PyTorch peak allocated GPU memory | 4.44 GiB | 5.07 GiB |
| PyTorch peak reserved GPU memory | 5.10 GiB | 6.28 GiB |
| Artifact directories after completion | 12.7 GB | 14.8 GB |

For these author measurements, RAM was sampled every five seconds across the process tree; shared pages may be counted more than once, and short-lived peaks may be missed. GPU allocated/reserved values were obtained from PyTorch process counters during the measurements. Disk figures sum the logical sizes of `datas`, `models_main`, `plot`, `log_file`, `reports_lira_lite`, and `vlm_2b`; they exclude the Python environment, external caches and measurement logs.

## 5. Expected outputs and success/failure criteria

### Pipeline outputs

| Path | Expected content |
| --- | --- |
| `datas/STL10/full.npz` | Preprocessed STL10 data. |
| `datas/STL10/resnet/uniform/` | Per-client retained and held-out data splits. |
| `models_main/resnet/STL10/server_model/server_<round>.pth` | 10 server checkpoints for quick, or 200 for full, in a clean run. |
| `models_main/resnet/STL10/client_model/` | Saved checkpoints for clients 0 and 1: 20 for quick or 400 for full, in a clean run. |
| `plot/vlm_data/resnet/STL10/member/` | Member trajectory PNGs; 500 in each measured run. |
| `plot/vlm_data/resnet/STL10/nonmember/` | Non-member trajectory PNGs; 500 in each measured run. |
| `plot/vlm_data/resnet/STL10/member_losses.pkl`, `nonmember_losses.pkl` | Per-sample loss sequences. |
| `plot/vlm_data/resnet/STL10/metrics_history_selected.pkl` | Merged member/non-member loss history used by the attack. |
| `reports_lira_lite/audit_details.json` | Per-sample membership scores and trajectory features. |
| `log_file/resnet/STL10/train_models` | Existing training log. |

Each audit record contains `id`, `gt`, `score`, `vlm_score`, and `features.ecd`/`features.tf`. Records receiving Phase-2 refinement also contain `features.p_ecd`, `features.p_tf`, and `features.score_phy`. Those additional fields are not expected on every record.

### Determine success

A reference run is successfully completed when:

1. the runner exits without an error;
2. `reports_lira_lite/audit_details.json` is created or updated by the current run and contains a nonempty list of sample records;
3. the report contains both membership classes (`gt = 0` and `gt = 1`), finite membership/VLM scores, and finite ECD/TF values;
4. required loss histories, member/non-member trajectory images, and requested checkpoints exist; and
5. the console prints a finite `ROC AUC` and the `TPR @ FPR` evaluation results.

Both measured runs met these criteria and evaluated 500 members plus 500 non-members. Execution success does not depend on exceeding an arbitrary AUC threshold and does not by itself certify reproduction of all paper results.

Aggregate AUC and TPR values are printed to the terminal. The JSON report stores per-sample results. To retain console output on Linux, use the following optional command:

```bash
set -o pipefail
bash run_stl10.sh --quick 2>&1 | tee quick_run.log
```

For full evaluation, omit `--quick` and use a separate log name. `quick_run.log` is created by this shell command; it is not an automatic pipeline output.

A basic report check can be run from the repository root without changing the source:

```bash
python - <<'PY'
import json
import math
from pathlib import Path

report = Path('reports_lira_lite/audit_details.json')
if not report.is_file():
    raise SystemExit('FAIL: audit report is missing.')
records = json.loads(report.read_text())
if not isinstance(records, list) or not records:
    raise SystemExit('FAIL: report is empty or invalid.')
if {row.get('gt') for row in records} != {0, 1}:
    raise SystemExit('FAIL: both membership classes are required.')
for row in records:
    features = row.get('features', {})
    values = [row.get('score'), row.get('vlm_score'), features.get('ecd'), features.get('tf')]
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
        raise SystemExit('FAIL: scores or ECD/TF are missing or non-finite.')
print('Report structure OK:', len(records), 'samples.')
print('Also confirm report freshness and check the console AUC/TPR.')
PY
```

A nonzero exit code, traceback, missing/empty report, invalid scores, or a message that both membership classes are unavailable indicates failure or incomplete evaluation. A normal exit or the existence of a JSON file alone is insufficient: the original attack can return without AUC if both classes are unavailable. Confirm that the outputs belong to the current run.

## 6. Claim-to-script/data mapping

| Claim / artifact component | Main scripts | Inputs | Observable outputs |
| --- | --- | --- | --- |
| C1: FL checkpoints support per-sample loss trajectories | `train.py`, `trajectory.py`, `fix.py` | `datas/STL10/full.npz`, client splits, server/client checkpoints | `models_main/resnet/STL10/`, trajectory PNGs, `metrics_history_selected.pkl` |
| C2: A VLM assigns visual membership scores | `attack.py:Phase1Screener`, `load_vlm` | Trajectory PNGs, local Qwen weights | `audit_details.json`: `vlm_score` |
| C3: ECD and TF characterize trajectory dynamics | `attack.py:DynamicsToolkit` | Per-sample loss histories | `audit_details.json`: `features.ecd`, `features.tf` |
| C4: Reverse-eCDF calibration and adaptive fusion refine scores | `attack.py:Phase2Investigator`, `LiraLiteAuditor.run` | VLM scores and calibration-pool trajectory features | `audit_details.json`: final `score`; `p_ecd`, `p_tf`, `score_phy` on refined records |
| C5: Membership inference can be evaluated using AUC and low-FPR TPR | `attack.py:LiraLiteAuditor._evaluate`, `main.py` | Ground-truth labels and final membership scores | Console `ROC AUC` and `TPR @ FPR` results |

For availability assessment, the table identifies the source, data dependencies and observable outputs of the reference workflow; comparative gains and ablation results require the corresponding experiments.

## 7. Reuse, reduced scope and external-data fallback

### Reuse existing outputs

For attack-only execution with all necessary assets already present:

```bash
bash run_stl10.sh --skip-data --skip-vlm --no-data-process --no-train --no-plots
```

Windows equivalent:

```powershell
.\run_stl10.bat -SkipData -SkipVlm -NoDataProcess -NoTrain -NoPlots
```

| Option | What must already be present |
| --- | --- |
| `--skip-data` / `-SkipData` | Raw STL10 files under `datas/stl10/stl10_binary/`. |
| `--skip-vlm` / `-SkipVlm` | Complete Qwen configuration, processor/tokenizer files and weights under `vlm_2b/`. |
| `--no-data-process` / `-NoDataProcess` | `datas/STL10/full.npz`. |
| `--no-train` / `-NoTrain` | Required checkpoints and per-client split files. |
| `--no-plots` / `-NoPlots` | Matching trajectory images and `metrics_history_selected.pkl`. |

An attack-only reuse run measures a reduced workflow, not the full reference runtime. The quick path reduces training cost but still performs local VLM inference and requires compatible GPU resources. No CPU-only timing or equivalence claim is made.

For direct attack evaluation with existing trajectories and loss histories:

```bash
python attack.py --dataset STL10 --model resnet --max_samples 1000 --vlm_type qwen3_2b --vlm_path ./vlm_2b
```

### Download failures and local assets

- **STL10:** obtain the binary archive from the [official dataset page](https://cs.stanford.edu/~acoates/stl10/), extract it so that `datas/stl10/stl10_binary/` contains the dataset files, and rerun with `--skip-data`. Public data availability is an upstream dependency; the dataset is not bundled with the source.
- **Qwen:** the default `--vlm-source auto` tries ModelScope, then Hugging Face. Use `--vlm-source modelscope` or `--vlm-source hf` to select a provider. If both are unavailable, obtain the complete [Qwen3-VL-2B-Instruct checkpoint](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct), place it under `./vlm_2b`, and rerun with `--skip-vlm`.
- Once dependencies and all assets are local, downloading can be disabled with the skip options. Offline execution still requires complete local files; missing assets are not replaced by synthetic data or API calls.

## 8. Reproducibility and known limitations

- `utils.set_seed()` seeds Python, NumPy, PyTorch, and CUDA RNGs. Data preparation, federated training and trajectory generation use seed 123; attack initialization resets the seed to 42.
- Evaluation adds the existing Gaussian noise with standard deviation `1e-9` to break tied scores before ROC calculation.
- Deterministic algorithms and cuDNN deterministic mode were disabled in the measured runs. Seeds therefore do not guarantee bitwise-identical GPU results. Keep software versions, asset contents, arguments and plotting settings fixed for comparisons.
- The VLM prompt and scoring logic are fixed in `attack.py`. Weights are loaded locally with `device_map="auto"`; numerical behavior and memory use can vary across library/GPU configurations.
- `main.py` automatically selects the GPU with the most free memory reported by `nvidia-smi` before importing PyTorch. Check the console log for the selected device and avoid competing GPU workloads during evaluation.
- Pipeline outputs use their existing paths and can be overwritten by later runs. Preserve quick/full results before rerunning. A clean output tree gives the clearest checkpoint counts; old checkpoints can remain after a shorter rerun.
- Input sample count is limited by both available images and loss histories. An unexpectedly smaller evaluation set should be investigated using the console's loaded member/non-member counts and the available input files.
- The reference measurements validate the stated Linux configuration. Alternative datasets/models, lower-memory hardware, Windows and different software stacks do not inherit the measured runtime or resource figures.

If a skipped stage reports missing files, rerun that stage or restore matching files. If CUDA memory is exhausted, close other workloads and verify device selection; then rerun the quick check. If the Transformers Qwen class cannot be imported, use a Qwen3-VL-capable Transformers version and check the recorded reference environment above.

## 9. Availability and licensing

**Permanent archive status:** a DOI or permanent archival identifier has not yet been supplied for the final artifact release. The final version must be deposited in a persistent public repository and its archival URL/DOI added to the submission materials. The source GitHub URL above is the current access point and is not presented as a completed permanent archive.

The source is released under the [MIT License](LICENSE). Third-party dataset/model files are downloaded separately and retain their upstream licenses and terms. Citation metadata is provided in [`CITATION.cff`](CITATION.cff); please also cite the associated paper when using this implementation.

Reviewer questions, corrected artifact locations and requested submission-field changes should be communicated through HotCRP as requested by the AE chairs.
