# Progress Summary: PII Detection Agent (CS471 Final Project Checkpoint)

Repository: https://github.com/skylerwuest-del/pii-detector (checkpoint tag: `checkpoint`)

The PII Detection Agent finds files that likely contain personal information so they can be reviewed and cleaned up. Its core algorithm is a character-level 1D convolutional neural network (PyTorch, CPU) that scores each file path. Deterministic rules then check the contents of the highest-scoring files without ever displaying the matched values.

**Completed functionality.** The CNN trains on 232 synthetic path families with family-disjoint train/validation/test splits. On the held-out test set it reaches F1 0.826 and ROC AUC 0.945, while a shuffled-label control falls to AUC 0.453, which shows it learns from the labels. The application offers a root-locked folder scan and an approval-gated full-computer scan. Verification covers text, Office, PDF, email, archive and image files (OCR). The program also provides saved scores for fast rescans, an HTML report, and a review screen to quarantine or recycle files. 62 automated tests pass on Windows and Linux in continuous integration.

**Remaining work.** Measure accuracy on real, consented folders. Retrain using review feedback. Improve score calibration. Add a full parser for legacy Office files and non-English schedule detection.

**Contributions.** Skyler built the checkpoint baseline, with OpenAI Codex: the CNN, training, synthetic data, both scan modes, verification, `run.bat`, the demo and the original tests. Zachary built everything after that, with Claude Code: the training improvements, additional file readers and PII checks, scan history, reports, the review and cleanup workflow, the interface, 48 more tests, CI and documentation.
