"""Character encoding shared by training and BOTH scanning modes."""
import unicodedata
import config


def fold_accents(text):
    """empleados/nómina -> empleados/nomina: accented Latin letters map onto the base alphabet."""
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def normalize_path(path, strip_accents=False):
    text = str(path).replace("\\", "/").lower()
    return fold_accents(text) if strip_accents else text


class PathEncoder:
    def __init__(self, characters=config.CHARACTERS, max_length=config.MAX_PATH_LENGTH, strip_accents=False):
        if max_length < 1:
            raise ValueError("max_length must be positive")
        self.characters = characters
        self.max_length = max_length
        # Saved with the checkpoint; checkpoints from before this option keep the old behavior.
        self.strip_accents = strip_accents
        self.vocabulary = {char: i + 2 for i, char in enumerate(characters)}

    def encode(self, path):
        # CNNs do arithmetic, so strings must become integer IDs and learned vectors.
        # Keep the suffix so long directories do not erase the filename.
        text = normalize_path(path, self.strip_accents)[-self.max_length:]
        ids = [self.vocabulary.get(char, 1) for char in text]  # 1 = unseen character.
        return ids + [0] * (self.max_length - len(ids))  # 0 = padding.

    def settings(self):
        return {"characters": self.characters, "max_length": self.max_length, "strip_accents": self.strip_accents}


# Optional metadata inputs for the SAME CNN (joined after pooling, see model.py).
EXTENSION_GROUPS = {
    "text": {".txt", ".log", ".md", ".json", ".xml", ".ini", ".cfg", ".yaml", ".yml", ".ics", ".vcf", ".eml"},
    "table": {".csv", ".tsv", ".xlsx", ".xls", ".ods"},
    "document": {".docx", ".doc", ".pdf", ".odt", ".rtf", ".pptx"},
    "archive": {".zip", ".7z", ".rar", ".gz", ".tar"},
    "media": {".jpg", ".jpeg", ".png", ".gif", ".mp3", ".mp4", ".mov", ".wav"},
}
NUM_METADATA_FEATURES = 3 + len(EXTENSION_GROUPS)


def metadata_features(relative_path, size):
    """[log size, folder depth, has-extension, one-hot extension group], each roughly in [0, 1]."""
    import math
    text = normalize_path(relative_path)
    name = text.rsplit("/", 1)[-1]
    extension = "." + name.rsplit(".", 1)[-1] if "." in name.strip(".") else ""
    features = [min(1.0, math.log1p(max(0, size or 0)) / 20), min(1.0, text.count("/") / 10), float(bool(extension))]
    features += [float(extension in group) for group in EXTENSION_GROUPS.values()]
    return features
