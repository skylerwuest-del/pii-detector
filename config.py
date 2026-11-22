"""All experiment settings. Paths are relative to this project, not the shell."""
from pathlib import Path
import string

# ==================================================
# CHANGE ME: MODEL AND EXPERIMENT CONFIGURATION
# These variables control the behavior of the CNN.
# Change these values to experiment with the model.
# ==================================================
EMBEDDING_DIM = 16  # Numbers per character; higher is richer/slower, lower is simpler.
NUM_FILTERS = 32  # Learned pattern detectors; higher adds capacity/overfitting, lower reduces it.
KERNEL_SIZE = 5  # Characters per window; higher sees longer fragments, lower sees shorter ones.
STRIDE = 1  # Window step; higher is faster but skips positions, lower preserves detail (minimum 1).
PADDING = 2  # Zeros around convolution input; higher retains more edge windows, lower crops edges.
DROPOUT_RATE = 0.2  # Fraction disabled during training; higher regularizes more, lower fits more freely.
MAX_PATH_LENGTH = 160  # Characters retained; higher uses more memory/context, lower drops more prefixes.
CHARACTERS = string.ascii_lowercase + string.digits + "/._- ()[]@!#$%&+=';,"  # More characters reduce unknowns but enlarge embedding.
VOCABULARY_SIZE = len(CHARACTERS) + 2  # Derived: padding + unknown + characters; do not edit independently.
LEARNING_RATE = 0.002  # Update size; higher learns faster but may oscillate, lower learns slower.
BATCH_SIZE = 32  # Examples per update; higher uses more memory/smoother updates, lower adds noise.
EPOCHS = 25  # Passes through training data; higher may overfit, lower may underfit.
RANDOM_SEED = 471  # Repeatable random choices; changing it changes samples and initialization, not quality monotonically.
TRAIN_SPLIT = 0.70  # Share of template families for training; higher leaves fewer held-out families.
VALIDATION_SPLIT = 0.15  # Share for checkpoint selection; higher leaves fewer training/test families.
TEST_SPLIT = 0.15  # Share for final evaluation; higher leaves fewer training families; shares must sum to 1.
PREDICTION_THRESHOLD = 0.5  # Positive decision cutoff; higher lowers recall, lower usually lowers precision.
WEIGHT_DECAY = 0.01  # AdamW penalty on large weights; higher regularizes more, 0 disables it.
GRADIENT_CLIP = 1.0  # Largest gradient norm per update; lower stabilizes, None disables clipping.
BALANCE_CLASSES = True  # True weights the rarer label in the loss; False treats every example equally.
CHECKPOINT_BY = "auc"  # Epoch kept: "auc" = best validation ranking (what scans use), "loss" = lowest validation loss.
LR_PATIENCE = 3  # Epochs without validation improvement before the learning rate is reduced.
LR_FACTOR = 0.5  # Learning-rate multiplier on a plateau; lower cuts harder, 1.0 disables reduction.
EARLY_STOPPING_PATIENCE = 8  # Epochs without improvement before training stops; None trains all EPOCHS.
AUGMENT_TRAINING_PATHS = True  # True varies training paths every epoch (folders, separators, copy suffixes, typos).
AUGMENT_PROBABILITY = 0.5  # Chance of each path variation; higher = more variety, lower stays closer to the data.
AUTO_TUNE_THRESHOLD = True  # True saves a validation-tuned cutoff with the checkpoint; False always uses PREDICTION_THRESHOLD.
THRESHOLD_F_BETA = 2.0  # Tuning weight: 2 favors recall (missing PII costs more), 1 balances, 0.5 favors precision.
USE_METADATA_FEATURES = False  # True adds file size, depth and type group as extra CNN inputs; most useful with feedback labels.
STRIP_ACCENTS = True  # True folds accents (nomina = nómina) before encoding; saved with the checkpoint.
CALIBRATE_SCORES = True  # True fits a temperature on validation so scores behave more like probabilities.
USE_FEEDBACK = True  # True adds your review decisions (data/feedback.json) to the training split.
FEEDBACK_REPEAT = 3  # Copies of each feedback example per epoch; higher trusts your labels more.
SWEEP_GRID = {"NUM_FILTERS": [16, 32, 64], "KERNEL_SIZE": [3, 5, 7], "DROPOUT_RATE": [0.2, 0.4]}  # train.py --sweep
NUM_THREADS = 2  # CPU threads for training and inference; higher is faster but uses more of the computer.
EXAMPLES_PER_FAMILY = 12  # Synthetic variants per family; higher grows data/time, lower reduces diversity.
PROJECT_DIRECTORY = Path(__file__).resolve().parent  # Project anchor; changing moves default artifacts.
TEST_DIRECTORY = PROJECT_DIRECTORY / "synthetic_demo"  # Authorized synthetic demo root; changing selects another demo location.
OUTPUT_DIRECTORY = PROJECT_DIRECTORY / "outputs"  # Reports and checkpoint folder; changing relocates outputs.
DATA_DIRECTORY = PROJECT_DIRECTORY / "data"  # Synthetic training corpus; changing relocates generated data.
MAX_INSPECTION_BYTES = 262144  # Per-file read limit; higher checks more text, lower bounds reading more tightly.

# ================================================
# CHANGE ME: SCANNING MODE CONFIGURATION
# ================================================
DEFAULT_SCAN_MODE = 1  # Empty menu choice; 1 opens navigator, 2 opens confirmation for broad scan.
TOP_K_FILES = 10  # Maximum accessible supported files inspected; higher reads more, lower reads fewer.
MAX_FILES_TO_SCAN = None  # Metadata limit; higher scans more; None removes the explicit limit.
INCLUDE_HIDDEN_FILES = True  # True includes hidden entries; False reduces coverage.
SHOW_SCAN_PROGRESS = True  # True prints counters/time; False produces quieter output.
INFERENCE_BATCH_SIZE = 64  # Paths per inference; higher uses more memory, lower has more overhead.
SUPPORTED_EXTENSIONS = [".txt", ".csv", ".tsv", ".log", ".md", ".json", ".xml", ".ini", ".cfg", ".yaml", ".yml", ".gitignore",
                        ".ics", ".vcf", ".eml", ".docx", ".xlsx", ".pptx", ".doc", ".xls", ".ppt", ".msg", ".pdf", ".zip",
                        ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".gif"]  # .pdf needs pypdf; images need OCR (below).
OCR_ENABLED = True  # Read text from images and scanned PDFs when pytesseract, Pillow and Tesseract are installed.
OCR_TIMEOUT_SECONDS = 30  # Longest OCR time per image; slow images are reported as failed inspections.
TESSERACT_COMMAND = None  # Full path to tesseract(.exe) if it is not on PATH or in C:\Program Files\Tesseract-OCR.
MAX_OCR_IMAGES = 10  # Images OCRed per PDF; higher finds more in long scans but takes longer.
FULL_SCAN_REQUIRE_CONFIRMATION = True  # True requests broad-scan approval; False skips that prompt for authorized automation.
FULL_SCAN_ALLOW_CONTENT_INSPECTION = True  # True allows a separate approval prompt; False disables broad content inspection.
EXCLUDED_DIRECTORIES = []  # Additional absolute directory exclusions; more exclusions reduce coverage.
REUSE_UNCHANGED_SCORES = True  # True reuses saved scores for files whose size and time are unchanged (fast rescans and resume).
QUARANTINE_DIRECTORY = OUTPUT_DIRECTORY / "quarantine"  # Review "quarantine" moves files here; restore puts them back.
RESULTS_PAGE_SIZE = 20  # Rows per page for the results / next commands.
FOLLOW_SYMLINKS = False  # Safety invariant: links/junctions are always skipped; True is rejected.


# ================================================
# SAVED SETTINGS FROM THE MENU (main.py option 5)
# ================================================
# The Settings menu writes a few scan options to user_settings.json beside this file.
# They override the values above. Delete that file to return to these defaults.
USER_SETTINGS_FILE = PROJECT_DIRECTORY / "user_settings.json"
USER_SETTABLE = ("TOP_K_FILES", "MAX_FILES_TO_SCAN", "INCLUDE_HIDDEN_FILES", "SHOW_SCAN_PROGRESS",
                 "PREDICTION_THRESHOLD", "AUTO_TUNE_THRESHOLD", "MAX_INSPECTION_BYTES", "REUSE_UNCHANGED_SCORES")


def load_user_settings():
    import json
    try:
        saved = json.loads(USER_SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    applied = {k: v for k, v in saved.items() if k in USER_SETTABLE}
    globals().update(applied)
    return applied


DEFAULTS = {name: globals()[name] for name in USER_SETTABLE}  # Values before saved settings apply.
load_user_settings()
