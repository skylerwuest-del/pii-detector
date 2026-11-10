# CS471 PII Detection Agent — Lesson 36

A Windows-first terminal application using **exactly one machine learning algorithm: a character-level 1D CNN in PyTorch**. It predicts risk from individual relative file paths, then checks a limited number of supported files using deterministic rules. Contents never enter the CNN. CPU only; no AI APIs, external models, uploads or cloud processing.

Use generated synthetic files for demonstrations. Other directories and computers require ownership or explicit audit authorization. Tests and the demo never scan real drives.

## Windows installation and complete demonstration

For the quickest start, install 64-bit Python 3.11 or newer, then double-click **`run.bat`** (or run `.\run.bat` from PowerShell). The launcher works from any working directory. Choose **1** for a controlled directory scan or **2** for a full computer scan. On first use it creates `.venv`, installs CPU dependencies (internet required), and generates synthetic training data and trains if no checkpoint exists. Later scans reuse the saved checkpoint. Full scans keep the existing `SCAN` and `INSPECT` approval prompts.

The launcher also offers **3** to regenerate synthetic data and retrain (replaces the saved checkpoint), **4** for the complete synthetic demo (also regenerates data and retrains), **5** to open `config.py` in Notepad, **6** to install or repair dependencies, and **0** to exit. Edit variables directly in `config.py`; use option 3 after changing model, encoder, or training settings. Returning with `back` from mode 1 returns to the launcher menu; `exit` also ends that mode and returns to the launcher. A failed operation shows its error and returns to the menu.

Manual installation is also available:

Install 64-bit Python 3.11 or newer with a compatible PyTorch CPU wheel. In PowerShell, from the checkout:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python demo.py
.\.venv\Scripts\python -m unittest discover -s tests -v
```

Activation is optional; these commands avoid activation-policy changes. To activate: `.\.venv\Scripts\Activate.ps1`. PyTorch is the only direct third-party dependency. Tests use standard-library `unittest`. Dependency installation requires network access once; the application runs offline.

The demo generates fictional data, trains with per-epoch losses, evaluates held-out data, saves the best validation checkpoint, ranks the synthetic tree, and inspects supported files. It then exercises the full-mode pipeline on the **same synthetic directory with the same predictor instance**, explicitly approving synthetic inspection. It never discovers or scans real drives.

Individual steps:

```powershell
.\.venv\Scripts\python generate_data.py
.\.venv\Scripts\python train.py
.\.venv\Scripts\python inference.py "C:\Code\FinalProjectAI\pii-detector\synthetic_demo"
.\.venv\Scripts\python main.py
```

Replace the example path with your checkout location. The direct inference command authorizes inspection of that explicitly supplied root. Train before opening the scan menu. No scan retrains the model. Generated files and weights are ignored by Git and regenerated locally.

## Directory structure and module responsibilities

```text
config.py           All experiment, inference and inspection settings
preprocessing.py    Character IDs, normalization, unknowns, suffix truncation
model.py            Single trainable CNN, with tensor-shape comments
generate_data.py    Synthetic contents, known labels, grouped split and demo tree
train.py            CPU backpropagation, checkpoint selection, test metrics
scanner.py          Navigator, root validation, safe enumeration, tree, drive discovery
inference.py        Shared Predictor, batches, disk-backed ranking, reports
inspection.py       Limited text/CSV readers and deterministic verification
main.py             Two-mode menu and explicit approval prompts
demo.py             Complete synthetic-only demonstration
tests/              Encoder, CNN, boundary, reporting and inspection tests
data/dataset.json   Path, label, template-family ID and split
data/corpus/        Synthetic contents corresponding to the manifest
synthetic_demo/     Small authorized example tree
outputs/model.pt    Weights, architecture and required encoder configuration
outputs/training_metrics.json   Loss history, weight change, held-out metrics
outputs/tree.txt    Controlled-mode directory hierarchy
outputs/file_paths.txt          Controlled-mode original absolute paths
outputs/results.csv            Controlled-mode descending scores and ranks
outputs/full_scan_file_paths.txt  Incremental full-mode absolute paths
outputs/full_scan_results.csv     Full-mode descending scores and ranks
outputs/*ranking.sqlite3          Disk-backed ranking databases
```

The small demo includes employees CSV, schedule text, meeting notes, config text, `.gitignore`, an unsupported file, misleading `personal_notes.txt` without PII, and generic `item.txt` with fictional PII. All examples use fictional names, reserved example email domains, example phone numbers and SSN-like strings. Never replace them with real personal data for class exercises.

## Mode 1: controlled directory navigator

The menu starts `Navigator` in your home directory. Commands: `ls`, `cd <directory>`, `cd ..`, `cd ~`, `pwd`, `select`, `scan`, `help`, `back`, `exit`. Spaces are accepted directly or with double quotes. These are safe Python operations, never arbitrary shell commands.

Example (replace path):

```text
cd C:\Code\FinalProjectAI\pii-detector\synthetic_demo
select
ls
cd Documents
cd ..
scan
back
```

Before selection navigation is unrestricted. After `select`, `Navigator.cd` uses `safe_path` to reject resolved paths outside the root, absolute outside paths, parent escapes and links/junctions. `cd ..` at the root fails; `cd ~` fails if home is outside. `scan` starts at the locked root even when your current directory is a subfolder. Return to the main menu to select a new root.

`enumerate_files` uses `os.scandir` recursively, validates containment with `pathlib`, skips reparse points and permission errors, and reads metadata only. Original absolute paths remain separate from normalized relative CNN inputs.

`save_tree` runs Windows `tree.com <root> /F /A` using an argument list and saves its output. File paths come from Python enumeration, never ASCII-tree parsing. Since Windows tree has no reliable no-follow flag, a preliminary safety walk detects links/unreadable entries; those trees use a safe Python hierarchy instead. Non-Windows systems also use that fallback. Keep the test tree stable during scans.

## Mode 2: explicitly authorized full computer audit

`fixed_drives` calls Windows `GetLogicalDrives` and `GetDriveTypeW == DRIVE_FIXED`. It excludes network/removable drives by default, displays discovered roots and requires the exact word `SCAN`. Common Windows/system/application directories and configured exclusions are skipped; no filesystem permissions are bypassed.

Stage 1 runs the same `Predictor` checkpoint and encoder over metadata batches. Stage 2 separately requests `INSPECT` before `inspect_candidates` reads only the top K accessible supported files. Declining opens no candidate contents. Disabling `FULL_SCAN_ALLOW_CONTENT_INSPECTION` removes this stage entirely. Paths themselves may reveal sensitive metadata, so handle local reports appropriately.

Progress shows evaluated count and elapsed time. `MAX_FILES_TO_SCAN` bounds a run; Ctrl+C cancels enumeration/inference while retaining completed batches. Paths and CSV rows are written incrementally. SQLite stores scores on disk and streams a final descending CSV; memory does not grow with the complete file list. The console displays the top K; the CSV contains every rank. Inspection traverses sorted candidates lazily and skips failed files until K successful supported inspections.

## CNN operations and mathematics

Each relative path, e.g. `documents/employees.csv`, is one example. No entire tree, contents, drive letter, absolute root, size or timestamp enters this checkpoint's CNN.

1. `PathEncoder.encode` lowercases, converts separators to `/`, maps characters to integer IDs (0 padding, 1 unknown), keeps the last 160 characters, and right-pads. Keeping the suffix preserves filenames. Neural layers perform arithmetic, so text needs numerical encoding.
2. `PathCNN.embedding` learns 16 numbers per character: `[B,L] -> [B,L,16]`. IDs are lookup indexes, not character similarity scores.
3. `PathCNN.conv` receives the transposed `[B,16,L]` tensor. Its 32 filters slide across five-character windows. Each output is a learned weighted sum plus bias: `h[b,f,t] = bias[f] + sum(W[f,c,j] * E[b,c,t*stride+j-padding])`. Default output: `[B,32,L]`.
4. `torch.relu` computes `max(0,h)`, adding nonlinearity.
5. `features.amax(dim=2)` takes each filter's strongest match across all positions: `[B,32]`. A fragment can matter anywhere in a path; exact location information is lost.
6. Training-only dropout disables 20% of pooled features. `nn.Linear` computes a weighted sum plus bias: `[B,32] -> [B,1]`, squeezed to `[B]`, one raw logit per path.
7. During inference, `sigmoid(z) = 1/(1+exp(-z))` produces [0,1] scores. These are model predictions, **not necessarily calibrated probabilities**. The threshold creates binary decisions; ranking uses continuous scores and includes lower-scored paths.

The actual learning mathematics lives in `model.py` and `train.py`: `BCEWithLogitsLoss`, `loss.backward()` (chain-rule gradients), and `optimizer.step()` (Adam weight updates). Binary cross-entropy fits two labels: `-[y log(sigmoid(z)) + (1-y) log(1-sigmoid(z))]`. The logits loss combines these operations stably; do not sigmoid logits before this loss. Adam is an optimizer for the same CNN, not another prediction algorithm. Regex inspection is deterministic, not ML.

See official [PyTorch Conv1D documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html) and [BCEWithLogitsLoss documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html).

## Hyperparameters: exactly where to change them

Edit **`config.py`**, beneath the highly visible CHANGE ME comments. Every configurable variable has an effect comment. Retrain after architecture/encoder changes. Inference loads saved settings, so editing current config cannot silently alter an existing checkpoint's architecture.

| Variable | Default and reason | Increase / decrease |
|---|---|---|
| `EMBEDDING_DIM` | 16: modest character vectors | Richer/slower / simpler representation |
| `NUM_FILTERS` | 32: several fragment detectors | More capacity and overfitting risk / fewer learned patterns |
| `KERNEL_SIZE` | 5: short fragments such as `staff` | Longer local context / shorter fragments |
| `STRIDE` | 1: every neighboring window | Faster but skips positions / minimum is 1 |
| `PADDING` | 2: preserves length with kernel 5 | More edge windows / less edge coverage |
| `DROPOUT_RATE` | 0.2: mild regularization | More regularization/underfit risk / more fitting freedom |
| `MAX_PATH_LENGTH` | 160: filename plus modest directory context | More context/memory / more discarded prefix |
| `CHARACTERS`, `VOCABULARY_SIZE` | Fixed basic alphabet + two special IDs | More characters reduce unknowns but enlarge embedding; size is derived |
| `LEARNING_RATE` | 0.002: moderate Adam updates | Faster/possibly unstable / slower learning |
| `BATCH_SIZE` | 32: small CPU batches | Smoother updates/more memory / noisier, more updates |
| `EPOCHS` | 25: inexpensive demonstration | More fitting/overfit risk / underfit risk |
| `RANDOM_SEED` | 471: reproducibility | Changing changes randomness; no monotonic quality effect |
| `TRAIN_SPLIT`, `VALIDATION_SPLIT`, `TEST_SPLIT` | 0.70/0.15/0.15 | Trade training data against reliability of held-out evaluation; sum must be 1 |
| `EXAMPLES_PER_FAMILY` | 12: manageable corpus | More data/time / fewer variants |
| `PREDICTION_THRESHOLD` | 0.5: conventional binary cutoff | Usually more precision, less recall / converse |
| `TOP_K_FILES` | 10: limited content checks | More content read / fewer checks |
| `INFERENCE_BATCH_SIZE` | 64: bounded CPU batches | More memory/throughput / more overhead |
| `MAX_INSPECTION_BYTES` | 262144: bounded reading | More coverage / faster partial checks |

Other centralized controls: `DEFAULT_SCAN_MODE`, `MAX_FILES_TO_SCAN`, `INCLUDE_HIDDEN_FILES`, `SHOW_SCAN_PROGRESS`, `SUPPORTED_EXTENSIONS`, `TEST_DIRECTORY`, `OUTPUT_DIRECTORY`, `DATA_DIRECTORY`, `FULL_SCAN_REQUIRE_CONFIRMATION`, `FULL_SCAN_ALLOW_CONTENT_INSPECTION`, `EXCLUDED_DIRECTORIES`, `FOLLOW_SYMLINKS`. More exclusions reduce coverage. Hidden entries are included by default. None removes the explicit file limit. Links are always disabled; setting True is rejected. Changing extensions alone does not add a parser: update `inspection.py` too. Disabling metadata confirmation is only for deliberate authorized automation; contents still require separate approval.

One convolution is deliberate. More layers (an edit to `model.py`, with matching saved settings) could combine fragments into longer features, but increase runtime, explanation complexity and overfitting risk. Ensure kernel/stride/padding yield a positive output length when experimenting.

## Ground truth, training and evidence of learning

The generator chooses fictional PII-bearing or benign contents first and records their known truth, never deriving a label from filename keywords or inspection rules. Filenames often correlate with contents but include misleading generic positives and personal/payroll-named negatives. Employee records, personnel docs, contacts, payroll, personal schedules, meeting notes, general documentation, configs and unrelated texts are represented.

It stratifies labels, shuffles template families with the seed, splits whole families, then generates random suffixes *inside* each split. No numbered variant of the same path template crosses splits. Default rounding gives 336 training, 72 validation and 72 test examples. Shared semantic words support transfer; synthetic examples are still correlated and do not establish real-world accuracy.

Training shuffles batches, computes loss, backpropagates, and updates embedding/convolution/output weights. Validation disables gradients and dropout. Lowest validation loss selects the saved checkpoint; held-out test evaluation happens afterward. The JSON records epoch losses, test precision/recall/F1, seed and maximum convolution-weight change. Precision measures correctness among predicted positives; recall measures positives found; F1 combines them. Zero-denominator cases return zero.

To check learning rather than memorization:

- Confirm nonzero convolution-weight change and declining training loss. Updates alone do not prove generalization.
- Compare validation loss against training loss; divergence suggests overfitting. Evaluate unseen template families and misleading names.
- Repeat with several seeds, report variability, and reserve new families rather than repeatedly tuning against test scores.
- Add independently designed synthetic path/content families. Compare against a constant-label sanity baseline without training another ML algorithm.
- A future experimental run of the **same CNN architecture** with shuffled training labels should lose held-out performance. Good results on this small synthetic corpus alone are not proof of real PII detection.

## Verification behavior and limitations

`inspection.inspect_file` revalidates containment before reading at most the byte cap. Text checks email, phone, SSN-like and clearly labeled name patterns. CSV uses `csv.DictReader`, pattern checks on text values and supported nonempty name columns. `.gitignore` reports sensitive filename/directory **metadata clues**, not proof of PII. Unsupported files are ranked but never opened for inspection. The console prints categories/yes-no results, never matched values.

Statuses distinguish detected indicators, no supported indicators, unsupported inspection and failed inspection. Truncation is reported. No indicators **does not prove absence of PII**. Rules have false positives/negatives, SSN-like strings are not verified identities, and UTF-8 replacement can miss other encodings. Physical addresses, detailed schedules, PDF/Office/binary parsing are future extensions.

This educational implementation rechecks paths but is not a race-free boundary against an adversary changing links during traversal/opening/tree execution. Use a stable authorized synthetic tree. Generated output, dataset, environment and Git folders of this checkout are excluded to avoid feedback into scans. Reports contain metadata only. Disk usage still grows with file count. Ctrl+C during final sorting/report writing may leave a partial CSV; completed batches remain in SQLite. Application code never elevates filesystem permissions.

## Lesson 36 and Lesson 41

Lesson 36 delivers the real trainable CPU CNN, reproducible content-ground-truth data, family-disjoint evaluation, model persistence, shared two-mode inference, root-locked navigation, drive discovery, incremental ranked reports, explicit broad-scan approvals, limited deterministic verification, tests and synthetic demo.

Suggested Lesson 41 extensions (confirm against your course rubric): larger independently designed datasets, repeated-seed experiments, error analysis, calibration evaluation, CNN hyperparameter experiments without new ML algorithms, resumable scans, stronger cancellation/report recovery, additional approved parsers, multilingual paths, address/schedule verification and stronger filesystem-race protection. This checkpoint prioritizes likely PII; it does not certify file safety.

## Verification in the development environment

The complete synthetic demo ran on Windows, Python 3.14 and PyTorch 2.15.0+cpu. The default seed produced test precision 1.0000, recall 0.9722 and F1 0.9859. Convolution weights changed by up to 0.322566. Validation selected epoch 2 (loss about 0.6989); subsequent training loss fell while validation loss rose. This is **overfitting evidence**, despite the favorable small test score, and motivates a larger independent validation corpus for Lesson 41. The training script explicitly reports this and retains the early checkpoint.

The synthetic demo verified Windows `tree /F /A`, ranking and limited inspection in both pipelines. Automated tests include native Windows junction restrictions, fixed-drive API filtering with mocked drive types, approval gates, unsupported-file handling and original-path reporting. No broad real-drive traversal or real personal-file inspection was performed. PyTorch may emit a missing-NumPy interoperability warning on a minimal installation; this project does not use NumPy and training/tests still work.
