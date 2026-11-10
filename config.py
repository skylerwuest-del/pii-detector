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
SUPPORTED_EXTENSIONS = [".txt", ".csv", ".gitignore"]  # More types expand inspection only if a reader supports them.
FULL_SCAN_REQUIRE_CONFIRMATION = True  # True requests broad-scan approval; False skips that prompt for authorized automation.
FULL_SCAN_ALLOW_CONTENT_INSPECTION = True  # True allows a separate approval prompt; False disables broad content inspection.
EXCLUDED_DIRECTORIES = []  # Additional absolute directory exclusions; more exclusions reduce coverage.
FOLLOW_SYMLINKS = False  # Safety invariant: links/junctions are always skipped; True is rejected.
