# CS471 PII Detection Agent — Lesson 36

A Windows-first terminal application using **exactly one machine learning algorithm: a character-level 1D CNN in PyTorch**. It predicts risk from individual relative file paths, then checks a limited number of supported files using deterministic rules. Contents never enter the CNN. CPU only; no AI APIs, external models, uploads or cloud processing.

Use generated synthetic files for demonstrations. Other directories and computers require ownership or explicit audit authorization. Tests and the demo never scan real drives.

## Windows installation and complete demonstration

For the quickest start, install 64-bit Python 3.11 or newer, then double-click **`run.bat`** (or run `.\run.bat` from PowerShell). The launcher works from any working directory. Choose **1** for a controlled directory scan or **2** for a full computer scan. On first use it creates `.venv`, installs CPU dependencies (internet required), and generates synthetic training data and trains if no checkpoint exists. Later scans reuse the saved checkpoint. Full scans keep the existing `SCAN` and `INSPECT` approval prompts.

The launcher also offers **3** to regenerate synthetic data and retrain (replaces the saved checkpoint), **4** for the complete synthetic demo (also regenerates data and retrains), **5** to open `config.py` in Notepad, **6** to install or repair dependencies (including optional PDF support), **7** to review flagged files, **8** to open the last HTML report, **9** for the full Python menu, and **0** to exit. Edit variables directly in `config.py`; use option 3 after changing model, encoder, or training settings. Inside Python (`python main.py`) the menu offers the same scans plus **3** train/retrain, **4** model details (held-out precision/recall/F1/AUC, threshold, calibration, feedback labels used), **5** review flagged files, **6** restore from quarantine, **7** settings and **0** exit; choosing a scan without a trained model offers to train first. `python main.py <command>` runs the same features without menus (see *Command line*). Returning with `back` from mode 1 returns to the launcher menu; `exit` also ends that mode and returns to the launcher. A failed operation shows its error and returns to the menu.

Manual installation is also available:

Install 64-bit Python 3.11 or newer with a compatible PyTorch CPU wheel. In PowerShell, from the checkout:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install "torch>=2.9,<3" --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pip install -r requirements-optional.txt
.\.venv\Scripts\python demo.py
.\.venv\Scripts\python -m unittest discover -s tests -v
```

Activation is optional; these commands avoid activation-policy changes. To activate: `.\.venv\Scripts\Activate.ps1`. PyTorch and NumPy are the direct third-party dependencies; `requirements-optional.txt` adds `pypdf` (PDF text) and `pytesseract` + `Pillow` (OCR for images and scanned PDFs). OCR also needs the free Tesseract program: on Windows run `winget install UB-Mannheim.TesseractOCR` (it is found automatically in `C:\Program Files\Tesseract-OCR`, or set `TESSERACT_COMMAND` in `config.py`). Without these, PDFs and images are still ranked, just not opened. NumPy enables PyTorch interoperability without startup warnings. Tests use standard-library `unittest`. Dependency installation requires network access once; the application runs offline.

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
main.py             Menu, navigator shell, settings, command line and explicit approval prompts
review.py           Review flagged files: open, keep, dismiss, quarantine, Recycle Bin; feedback labels
report.py           Self-contained HTML report (counts and paths, never matched values)
ui.py               Stdlib-only terminal helpers: color (auto-off when piped), tables, progress, loss chart
demo.py             Complete synthetic-only demonstration
tests/              Encoder, CNN, boundary, readers, scan engine, review, training, UI and CLI tests
.github/workflows/  CI: tests, synthetic demo and CLI on Windows and Linux for every push
requirements-optional.txt   pypdf (PDF), pytesseract + Pillow (OCR)
user_settings.json  Scan options saved from the Settings menu (git-ignored; delete to reset)
data/feedback.json  Your review decisions, used as extra training labels (stays local)
data/dataset.json   Path, label, template-family ID and split
data/corpus/        Synthetic contents corresponding to the manifest
synthetic_demo/     Small authorized example tree
outputs/model.pt    Weights, architecture and required encoder configuration
outputs/training_metrics.json   Loss history, weight change, held-out metrics
outputs/tree.txt    Controlled-mode directory hierarchy
outputs/file_paths.txt          Controlled-mode original absolute paths
outputs/results.csv            Controlled-mode descending scores and ranks
outputs/inspection_report.csv   Per-file verdicts, severity and indicator COUNTS (never values)
outputs/seed_study.json         Written by `train.py --seeds N`
outputs/full_scan_file_paths.txt  Incremental full-mode absolute paths
outputs/full_scan_results.csv     Full-mode descending scores and ranks
outputs/*ranking.sqlite3          Disk-backed ranking databases (results + scan metadata)
outputs/scan_history.sqlite3      Size, modified time and score per file across scans (reuse, change tracking)
outputs/*report.html              HTML report of the last scan
outputs/quarantine/               Files moved by review, with manifest.json for restore
outputs/sweep.json                Written by `train.py --sweep`
```

The small demo includes employees CSV, schedule text, meeting notes, config text, `.gitignore`, an unsupported file, misleading `personal_notes.txt` without PII, and generic `item.txt` with fictional PII. All examples use fictional names, reserved example email domains, example phone numbers and SSN-like strings. Never replace them with real personal data for class exercises.

## Mode 1: controlled directory navigator

The menu starts `Navigator` in your home directory. Commands: `ls` (folders first, sizes), `cd <directory>`, `cd ..`, `cd ~`, `cd -` (previous folder), `pwd`, `tree [depth]`, `places` (Documents, Downloads, Desktop ...), `select`, `scan`, `results [--high] [--ext csv] [--folder Documents] [--changed]`, `next` / `prev` (pages), `folders [depth]` (riskiest folders), `inspect` (verify the top files again), `review [--all]`, `restore`, `report` (HTML, opens in the browser), `help`, `back`, `exit`. Tab completes commands and sub-folders where Python's `readline` exists (not on stock Windows). Links are listed but never entered. The prompt turns green with `[root locked]` after `select`. Spaces are accepted directly or with double quotes. These are safe Python operations, never arbitrary shell commands.

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

`enumerate_files` uses `os.scandir` recursively, validates each folder with `safe_path` once, checks every entry as it is read, skips reparse points and permission errors, and reads metadata only. It optionally fills a `stats` dict (links, hidden, excluded, unreadable) that the scan summary prints so you can see what was not covered. Exclusions are cheap string-prefix tests rather than a path resolution per entry. Original absolute paths remain separate from normalized relative CNN inputs.

`save_tree` runs Windows `tree.com <root> /F /A` using an argument list and saves its output. File paths come from Python enumeration, never ASCII-tree parsing. Since Windows tree has no reliable no-follow flag, a preliminary safety walk detects links/unreadable entries; those trees use a safe Python hierarchy instead. Non-Windows systems also use that fallback. Keep the test tree stable during scans.

## Mode 2: explicitly authorized full computer audit

`fixed_drives` calls Windows `GetLogicalDrives` and `GetDriveTypeW == DRIVE_FIXED`. It excludes network/removable drives by default, displays discovered roots and requires the exact word `SCAN`. Common Windows/system/application directories and configured exclusions are skipped; no filesystem permissions are bypassed.

Stage 1 runs the same `Predictor` checkpoint and encoder over metadata batches. Stage 2 separately requests `INSPECT` before `inspect_candidates` reads only the top K accessible supported files. Declining opens no candidate contents. Disabling `FULL_SCAN_ALLOW_CONTENT_INSPECTION` removes this stage entirely. Paths themselves may reveal sensitive metadata, so handle local reports appropriately.

Rescans are fast and interrupted scans resume: `outputs/scan_history.sqlite3` remembers every file's size, modified time and score, and a file whose size and time are unchanged keeps its saved score instead of going through the CNN again (only for the same `model.pt`; `REUSE_UNCHANGED_SCORES = False` disables this). So after Ctrl+C, scanning again re-lists folders but skips re-scoring finished files, and the menu says when a previous full scan stopped early. Each result is marked new, changed or unchanged since the last scan; `results --changed` lists only what changed.

Progress is one self-updating line (count, files/s, elapsed) on a terminal and an occasional plain line when piped; the summary then reports files evaluated, how many scored above the decision threshold, and what was skipped. Scores use the validation-tuned threshold saved in the checkpoint (see below). `MAX_FILES_TO_SCAN` bounds a run; Ctrl+C cancels enumeration/inference while retaining completed batches. Paths and CSV rows are written incrementally. SQLite stores scores on disk and streams a final descending CSV; memory does not grow with the complete file list. The console displays the top K; the CSV contains every rank. Inspection traverses sorted candidates lazily and skips failed files until K successful supported inspections.

## CNN operations and mathematics

Each relative path, e.g. `documents/employees.csv`, is one example. No entire tree, contents, drive letter, absolute root or timestamp enters the CNN. File size enters only when `USE_METADATA_FEATURES = True` (see below).

1. `PathEncoder.encode` lowercases, converts separators to `/`, folds accents when the checkpoint says so (`nómina` -> `nomina`; `STRIP_ACCENTS`), maps characters to integer IDs (0 padding, 1 unknown), keeps the last 160 characters, and right-pads. Keeping the suffix preserves filenames. Neural layers perform arithmetic, so text needs numerical encoding.
2. `PathCNN.embedding` learns 16 numbers per character: `[B,L] -> [B,L,16]`. IDs are lookup indexes, not character similarity scores.
3. `PathCNN.conv` receives the transposed `[B,16,L]` tensor. Its 32 filters slide across five-character windows. Each output is a learned weighted sum plus bias: `h[b,f,t] = bias[f] + sum(W[f,c,j] * E[b,c,t*stride+j-padding])`. Default output: `[B,32,L]`.
4. `torch.relu` computes `max(0,h)`, adding nonlinearity.
5. `features.amax(dim=2)` takes each filter's strongest match across all positions: `[B,32]`. A fragment can matter anywhere in a path; exact location information is lost.
6. Training-only dropout disables 20% of pooled features. With `USE_METADATA_FEATURES`, 8 extra numbers (log file size, folder depth, has-extension, file-type group) are concatenated: `[B,32] -> [B,40]`. `nn.Linear` computes a weighted sum plus bias: `[B,32 or 40] -> [B,1]`, squeezed to `[B]`, one raw logit per path. This is still the same single CNN; the extra inputs only widen its final layer.
7. During inference, `sigmoid(z / T) = 1/(1+exp(-z/T))` produces [0,1] scores, where the temperature `T` (saved with the checkpoint, 1.0 when not calibrated) only softens over-confident logits and never changes the ranking. These are model predictions, **not guaranteed probabilities**; the training report shows how well they are calibrated. The threshold creates binary decisions; ranking uses continuous scores and includes lower-scored paths.

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
| `PREDICTION_THRESHOLD` | 0.5: fallback cutoff, used when `AUTO_TUNE_THRESHOLD` is False or the checkpoint has none | Usually more precision, less recall / converse |
| `AUTO_TUNE_THRESHOLD`, `THRESHOLD_F_BETA` | True, 2.0: cutoff tuned on validation data, favoring recall | Beta higher = fewer missed PII files, more false alarms |
| `WEIGHT_DECAY` | 0.01: AdamW penalty on large weights | More regularization / none at 0 |
| `GRADIENT_CLIP` | 1.0: largest gradient norm per update | Lower = steadier / None disables |
| `BALANCE_CLASSES` | True: rarer label weighted in the loss | False treats all examples equally |
| `LR_PATIENCE`, `LR_FACTOR` | 3, 0.5: halve the learning rate after 3 flat epochs | Lower factor cuts harder; 1.0 disables |
| `EARLY_STOPPING_PATIENCE` | 8: stop after 8 epochs without improvement | None trains all epochs |
| `CHECKPOINT_BY` | "auc": keep the epoch with the best validation ranking | "loss" keeps the lowest validation loss |
| `AUGMENT_TRAINING_PATHS`, `AUGMENT_PROBABILITY` | True, 0.5: vary paths each epoch | Higher = more variety; False trains on the raw paths |
| `NUM_THREADS` | 2: CPU threads for training and inference | Faster / heavier on the computer |
| `USE_METADATA_FEATURES` | False: paths only | True adds size, depth and type group as CNN inputs; synthetic sizes carry no label signal, so this helps once you have feedback labels |
| `STRIP_ACCENTS` | True: fold accents before encoding | False treats accented letters as unknown characters |
| `CALIBRATE_SCORES` | True: fit a softening temperature on validation | False keeps raw sigmoid scores |
| `USE_FEEDBACK`, `FEEDBACK_REPEAT` | True, 3: your review decisions join training, each repeated 3 times per epoch | Higher trusts your labels more |
| `SWEEP_GRID` | filters 16/32/64, kernel 3/5/7, dropout 0.2/0.4 | Settings tried by `train.py --sweep` |
| `REUSE_UNCHANGED_SCORES` | True: unchanged files keep saved scores | False re-scores everything |
| `OCR_ENABLED`, `OCR_TIMEOUT_SECONDS`, `MAX_OCR_IMAGES`, `TESSERACT_COMMAND` | True, 30 s, 10, auto | OCR images and scanned PDF pages when installed; time limit per image; images per PDF; Tesseract location |
| `QUARANTINE_DIRECTORY`, `RESULTS_PAGE_SIZE` | outputs/quarantine, 20 | Where quarantined files go; rows per results page |
| `TOP_K_FILES` | 10: limited content checks | More content read / fewer checks |
| `INFERENCE_BATCH_SIZE` | 64: bounded CPU batches | More memory/throughput / more overhead |
| `MAX_INSPECTION_BYTES` | 262144: bounded reading | More coverage / faster partial checks |

Other centralized controls: `DEFAULT_SCAN_MODE`, `MAX_FILES_TO_SCAN`, `INCLUDE_HIDDEN_FILES`, `SHOW_SCAN_PROGRESS`, `SUPPORTED_EXTENSIONS`, `TEST_DIRECTORY`, `OUTPUT_DIRECTORY`, `DATA_DIRECTORY`, `FULL_SCAN_REQUIRE_CONFIRMATION`, `FULL_SCAN_ALLOW_CONTENT_INSPECTION`, `EXCLUDED_DIRECTORIES`, `FOLLOW_SYMLINKS`. More exclusions reduce coverage. Hidden entries are included by default. None removes the explicit file limit. Links are always disabled; setting True is rejected. Inspection can read text formats (`.txt .csv .tsv .log .md .json .xml .ini .cfg .yaml .yml .gitignore .ics .vcf`), email (`.eml`), Office (`.docx .xlsx .pptx`), legacy Office and Outlook (`.doc .xls .ppt .msg`), zip archives, `.pdf` with pypdf, and images (`.png .jpg .jpeg .tif .tiff .bmp .gif`) with OCR; `SUPPORTED_EXTENSIONS` selects among them, and anything else needs a new reader in `inspection.py`. Disabling metadata confirmation is only for deliberate authorized automation; contents still require separate approval.

One convolution is deliberate. More layers (an edit to `model.py`, with matching saved settings) could combine fragments into longer features, but increase runtime, explanation complexity and overfitting risk. Ensure kernel/stride/padding yield a positive output length when experimenting.

## Ground truth, training and evidence of learning

The generator chooses fictional PII-bearing or benign contents first and records their known truth, never deriving a label from filename keywords or inspection rules. Filenames often correlate with contents but include misleading generic positives and personal/payroll-named negatives. Employee records, personnel docs, contacts, payroll, personal schedules, meeting notes, general documentation, configs and unrelated texts are represented.

It stratifies labels, shuffles template families with the seed, splits whole families, then generates random suffixes *inside* each split. No numbered variant of the same path template crosses splits. The corpus has 52 hand-written families plus 180 reproducible programmatic ones built from word pools (sensitive, neutral and benign folder and file words in English, Spanish, French, German, Italian and Portuguese, with deliberate decoys such as benign `payroll_blank` files and PII-bearing `item`/`data` files). Programmatic families take realistic shapes: dated (`2024-03 payroll {id}`), versioned (`salary_final ({id})`), deeply nested and whole-folder families, with `.xlsx`, `.docx` and `.pdf` names shared by both labels. Each record also gets a file size drawn independently of its label. Default rounding gives 1944 training, 408 validation and 432 test examples; the original 40-family corpus had only six validation families, which made checkpoint selection noisy. Shared semantic words support transfer; synthetic examples are still correlated and do not establish real-world accuracy, and decoy families cap the accuracy any path-only model can reach.

Each epoch trains on freshly augmented paths (random neutral parent folders, `_`/`-`/space swaps, copy suffixes such as ` (1)` or `_final`, and an occasional typo; labels never change), shuffles batches, computes a class-weighted loss, backpropagates, clips gradients and updates embedding/convolution/output weights with AdamW. The learning rate halves after three flat validation epochs and training stops after eight epochs without improvement (Ctrl+C also stops cleanly and keeps the best epoch). Validation disables gradients and dropout. The epoch with the best validation **ROC AUC** (ranking quality, which is what scans use; loss breaks ties) is saved; `CHECKPOINT_BY = "loss"` restores loss-based selection. The decision threshold is then tuned on validation data only (F-beta with beta 2, the median of tied cutoffs) and saved inside `model.pt`; the held-out test set is evaluated once at the end. The JSON records per-epoch loss, AUC, F1 and learning rate, test precision/recall/F1/AUC/accuracy/confusion counts, the threshold, an all-positive baseline, seed and maximum convolution-weight change. Precision measures correctness among predicted positives; recall measures positives found; F1 combines them. Zero-denominator cases return zero.

After selecting the best epoch, training fits a temperature on validation logits (only values of 1 or more are allowed: a nearly separable validation set otherwise asks for sharper scores, which over-state confidence on harder paths), then tunes the threshold on the calibrated validation scores. The report prints expected calibration error before and after with a reliability table, and an **error analysis**: which test path families and words produce false alarms and missed PII, saved in `training_metrics.json` under `errors`. Use it to decide which training families to add.

**Learning from your decisions.** Every review decision (keep, not PII, quarantine, recycle) is saved to `data/feedback.json` as the file's path relative to the scan root plus a label. When `USE_FEEDBACK` is on, the next training run adds these to the training split (repeated `FEEDBACK_REPEAT` times); validation and test stay synthetic so scores remain comparable. The file stays on your computer and is git-ignored.

`python train.py --sweep` tries every combination in `SWEEP_GRID` on the same data, ranks them by **validation** AUC (loss breaks ties), prints the best settings and saves `outputs/sweep.json`; it changes nothing until you copy the values into `config.py` and retrain. The test score of the chosen setting is shown once for reference only.

Two built-in checks cover the points below. `python train.py --control` retrains the same CNN on shuffled labels (saving nothing) and compares test AUC with the real run; `python train.py --seeds 5` repeats training over consecutive seeds on the same data and reports mean, standard deviation and range. Other options: `--epochs N`, `--seed S`.

To check learning rather than memorization:

- Confirm nonzero convolution-weight change and declining training loss. Updates alone do not prove generalization.
- Compare validation loss against training loss; divergence suggests overfitting. Evaluate unseen template families and misleading names.
- Repeat with several seeds, report variability, and reserve new families rather than repeatedly tuning against test scores.
- Add independently designed synthetic path/content families. Compare against a constant-label sanity baseline without training another ML algorithm.
- A future experimental run of the **same CNN architecture** with shuffled training labels should lose held-out performance. Good results on this small synthetic corpus alone are not proof of real PII detection.

## Verification behavior and limitations

`inspection.inspect_file` opens every file through `scanner.safe_open`, which closes the gap between checking a path and opening it: it validates containment and every path component, records which file that is (device and inode) and that it is a regular file, opens it without following links (`O_NOFOLLOW` where the OS has it), confirms the open descriptor is the same file, and on Windows confirms the open handle's real path (`GetFinalPathNameByHandleW`) is still inside the root. A file swapped for a link or another file in between is refused and reported as a failed inspection. Every reader then works from that one open handle and reads at most the byte cap. Readers turn each format into bounded text: plain text with BOM detection, a UTF-16 heuristic, strict UTF-8 and a Windows-1252 fallback (the detected encoding is recorded); Office documents from their XML parts (streamed with the byte cap, never trusting declared sizes); PDFs page by page (at most 50 pages) when pypdf is installed, OCRing the images of pages that have almost no text (scans) when OCR is available; images by OCR (scaled to at most 3000 px, 30 s each); legacy `.doc/.xls/.ppt/.msg` OLE files by recovering their 8-bit and UTF-16 text runs (a `strings`-style reader, not a full parser); zip archives member by member (at most 50 readable members, nested archives and encrypted members skipped); email headers and text bodies; vCard and calendar names. Checks then count email; US and international phone numbers; SSN-like numbers (an impossible structure such as area 000, 666 or 9xx, group 00 or serial 0000 counts only when the line labels it as an SSN); labeled names, dates of birth, passport, driver's-license and bank-account numbers; ABA routing numbers that pass their checksum; IBANs that pass mod-97; street addresses; credit-card numbers that pass the Luhn checksum; and personal **schedules**: lines with a date, a time and an appointment word (doctor, dentist, pick up, flight, interview ...) and calendar events that have a start time plus a location or attendee. CSV/TSV uses `csv.DictReader`: non-empty cells under recognizable column headers (`email`, `phone`, `ssn`, `dob`, `address`, `card_number`, name columns ...) count too, without double counting a value the patterns already found. Each file gets a plain-language severity: High (any government, card or bank identifier, or three or more categories), Medium (two), Low (one). `.gitignore` reports sensitive filename/directory **metadata clues**, not proof of PII. Unsupported files are ranked but never opened for inspection. The console prints categories with occurrence counts, never matched values, and `outputs/inspection_report.csv` saves the same counts per file as a review list. Nothing is ever permanently deleted by this program; see *Reviewing and cleaning up*.

Statuses distinguish detected indicators, no supported indicators, unsupported inspection and failed inspection. Truncation is reported. No indicators **does not prove absence of PII**. Rules have false positives/negatives, identifier checks are structural (a valid checksum does not prove a real account), OCR misreads small or stylized text and is slow, the legacy-file reader can miss text stored in unusual ways, schedule detection is English-only and keyword-based, and encoding detection is heuristic.

File contents are opened race-safely (see above). Directory traversal and Windows `tree.com` still check links before use rather than through open handles, so an adversary changing links during traversal could affect which names are listed, though not which file contents are read. Use a stable authorized synthetic tree. Generated output, dataset, environment and Git folders of this checkout are excluded to avoid feedback into scans. Reports contain metadata only. Disk usage still grows with file count. Ctrl+C during final sorting/report writing may leave a partial CSV; completed batches remain in SQLite. Application code never elevates filesystem permissions.

## Reviewing and cleaning up flagged files

After a scan, `review` (navigator), menu option 5, `run.bat` option 7 or `python main.py review` steps through every inspected file that has PII indicators (`--all` includes the rest). For each file you see its severity, indicator counts, CNN score and size, then choose:

- **o** open it in its default application (to look before deciding);
- **k** keep: it is PII, leave it where it is;
- **n** not PII: a false alarm;
- **q** quarantine: after a y/N confirmation, move it to `outputs/quarantine/<id>/` and record where it came from; `restore` (or menu option 6) puts it back and refuses to overwrite anything;
- **r** recycle: after a y/N confirmation, send it to the Windows Recycle Bin (or the macOS/Linux Trash), where it can be restored by the operating system;
- **s** skip, **x** stop.

Nothing is permanently deleted. Every decision except skip is also saved as a training label (see *Learning from your decisions*). Files that no longer exist are left out, and actions refuse links and missing files.

## Reports, results and folders

`results` pages through the ranked list of the last scan with filters that combine: `--high` (above the threshold), `--ext csv`, `--folder Documents` (any folder of that name below the root), `--changed` (new or changed since the previous scan). `next` and `prev` move between pages. `folders [depth]` ranks folders by how many files score above the threshold, then by their highest and mean score, which shows where PII clusters. `report` writes a self-contained HTML page (summary cards, verified files with indicator counts, riskiest folders, top 50 paths; light and dark mode) and opens it. The report contains paths from your computer, so keep it private.

## Settings menu

Menu option 7 lists the scan options people change most (`TOP_K_FILES`, `MAX_FILES_TO_SCAN`, `INCLUDE_HIDDEN_FILES`, `SHOW_SCAN_PROGRESS`, `PREDICTION_THRESHOLD`, `AUTO_TUNE_THRESHOLD`, `MAX_INSPECTION_BYTES`, `REUSE_UNCHANGED_SCORES`). Changes are checked (for example a threshold must lie between 0 and 1) and apply to the session; **s** saves only values that differ from `config.py` into `user_settings.json`, which overrides `config.py` on the next start, and **r** resets everything to `config.py`. Model and training settings stay in `config.py` because they require retraining.

## Command line

`python main.py` with no arguments opens the menu. Commands for scripts and quick use:

```powershell
python main.py scan "C:\Users\me\Documents" --top 20 --html   # rank, verify the top 20, write and open the report
python main.py scan "C:\Users\me\Documents" --no-inspect        # rank names only; opens no file contents
python main.py scan "D:\Shared" --fail-on-pii                     # exit code 3 when any verified file has indicators
python main.py results --high --ext xlsx --limit 100
python main.py report --no-open
python main.py review
python main.py restore
python main.py full                                               # still asks for SCAN and INSPECT
python main.py train --epochs 30
```

Passing a folder to `scan` is the authorization for that folder (like selecting it in the navigator); links and junctions are refused as roots. `full` keeps both approvals unless `--yes` (metadata scan) and `--inspect` (content check) are given explicitly for authorized automation. Exit codes: 0 success, 1 failure or cancelled, 2 bad arguments, 3 PII found with `--fail-on-pii`.

## Continuous integration

`.github/workflows/tests.yml` runs on every push and pull request on **Windows and Linux** with Python 3.11 and 3.13: it installs CPU PyTorch and the optional PDF reader, runs all unit tests (on Windows this includes the junction-boundary and fixed-drive tests that are skipped elsewhere), runs the synthetic demo, and exercises the command line. Check the Actions tab on GitHub for results.

## Lesson 36 and Lesson 41

Lesson 36 delivers the real trainable CPU CNN, reproducible content-ground-truth data, family-disjoint evaluation, model persistence, shared two-mode inference, root-locked navigation, drive discovery, incremental ranked reports, explicit broad-scan approvals, limited deterministic verification, tests and synthetic demo.

Lesson 41 extensions now implemented (confirm against your course rubric): a larger dataset with realistic and multilingual families, repeated-seed experiments, a shuffled-label control, error analysis, calibration evaluation, a validation-only hyperparameter sweep of the same CNN, resumable scans with change tracking, additional parsers (Office, PDF, zip, email, contacts, calendars), address and identifier verification, and learning from your own review decisions. Also implemented since: race-safe file opening, schedule verification, legacy Office/Outlook files, and OCR for images and scanned PDFs. Remaining ideas: a full OLE parser, non-English schedule words, and directory-level race protection during traversal (files are protected at open time). This checkpoint prioritizes likely PII; it does not certify file safety.

## Verification in the development environment

The original synthetic demo was run on Windows, Python 3.14 and PyTorch 2.15.0+cpu. The later changes were developed on Linux (Python 3.13, PyTorch 2.15.0 from PyPI, pypdf 7.1.1): all 62 `unittest` tests pass with four Windows-only tests skipped (junction boundary, fixed-drive discovery, Recycle Bin and the open-handle path check), OCR was tested with Tesseract 5.3, and the suite also passes with no optional packages installed, the menu, navigator, review, settings and command line were driven with scripted input and in a real terminal, and the HTML report was rendered in a browser at desktop and phone widths. **They have not been run on Windows here**: the CI workflow runs the whole suite, including the Windows-only tests, on Windows for every push, and `run.bat`, `tree.com`, the Recycle Bin and console colors still deserve a quick manual check.

Default-seed results on the current synthetic corpus (held-out test, 432 paths): precision 0.797, recall 0.857, F1 0.826, ROC AUC 0.945; the all-positive baseline F1 is 0.667. Five consecutive seeds gave F1 0.851 +/- 0.016 (range 0.826-0.873) and AUC 0.931 +/- 0.017. The shuffled-label control scored AUC 0.45 against 0.95 for real labels, so the CNN learns from the labels rather than memorizing. The sweep confirmed the default architecture (32 filters, kernel 5, dropout 0.2) as best on validation. Calibration: expected calibration error is 0.12; validation is nearly separable, so the fitted temperature stays at 1.0; the remaining over-confidence most likely comes from decoy families that happen to be rarer in validation than in test. The error analysis attributes most mistakes to those decoys (for example benign `reports/weather` files and PII-bearing files named only `records` or in other languages), which no path-only model can separate; this is why verification reads the contents of the top files. Scores here are on synthetic data and do not establish real-world accuracy. No broad real-drive traversal or real personal-file inspection was performed. NumPy is included in the dependencies to prevent PyTorch missing-NumPy startup warnings.
