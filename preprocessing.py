"""Character encoding shared by training and BOTH scanning modes."""
import config


def normalize_path(path):
    return str(path).replace("\\", "/").lower()


class PathEncoder:
    def __init__(self, characters=config.CHARACTERS, max_length=config.MAX_PATH_LENGTH):
        if max_length < 1:
            raise ValueError("max_length must be positive")
        self.characters = characters
        self.max_length = max_length
        self.vocabulary = {char: i + 2 for i, char in enumerate(characters)}

    def encode(self, path):
        # CNNs do arithmetic, so strings must become integer IDs and learned vectors.
        # Keep the suffix so long directories do not erase the filename.
        text = normalize_path(path)[-self.max_length:]
        ids = [self.vocabulary.get(char, 1) for char in text]  # 1 = unseen character.
        return ids + [0] * (self.max_length - len(ids))  # 0 = padding.

    def settings(self):
        return {"characters": self.characters, "max_length": self.max_length}
