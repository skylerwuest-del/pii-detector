"""Generate fictional contents first; label by the selected content, never by filename."""
import json
import random
from pathlib import Path
import config

# Families, not individual numbered files, are split. No template crosses splits.
# Shared words are intentional: the CNN needs transferable character fragments.
FAMILIES = [
    ("hr/employees_{id}.csv", 1), ("staff/personnel_{id}.txt", 1),
    ("finance/payroll_{id}.csv", 1), ("contacts/customer_{id}.csv", 1),
    ("calendar/personal_{id}.txt", 1), ("private/employee_{id}.txt", 1),
    ("office/contact_list_{id}.txt", 1), ("people/salary_{id}.csv", 1),
    ("archive/roster_{id}.csv", 1), ("records/personnel_form_{id}.txt", 1),
    ("inbox/item_{id}.txt", 1), ("misc/data_{id}.csv", 1),
    ("docs/manual_{id}.txt", 0), ("settings/config_{id}.txt", 0),
    ("meetings/notes_{id}.txt", 0), ("public/readme_{id}.txt", 0),
    ("guides/setup_{id}.txt", 0), ("examples/tutorial_{id}.txt", 0),
    ("projects/design_{id}.txt", 0), ("reports/weather_{id}.csv", 0),
    ("assets/colors_{id}.csv", 0), ("reference/api_{id}.txt", 0),
    ("sandbox/personal_notes_{id}.txt", 0), ("templates/payroll_blank_{id}.txt", 0),
    ("human_resources/employee_contacts/{id}.txt", 1),
    ("accounts/payroll_export/{id}.csv", 1),
    ("team/personnel_details/{id}.txt", 1), ("clients/contact_cards/{id}.csv", 1),
    ("handbook/general_docs/{id}.txt", 0), ("build/configuration/{id}.txt", 0),
    ("minutes/public_meeting/{id}.txt", 0), ("learning/text_documents/{id}.txt", 0),
    ("schedules/private_contacts/{id}.txt", 1), ("directory/employee_names/{id}.csv", 1),
    ("reference/general_notes/{id}.txt", 0), ("software/config_defaults/{id}.txt", 0),
    ("wages/personnel_payroll/{id}.csv", 1), ("addressbook/contact_emails/{id}.txt", 1),
    ("help/documentation/{id}.txt", 0), ("release/meeting_summary/{id}.txt", 0),
]


def content_for(label, csv_file=False):
    if label:
        # Reserved example domain and fictional indicators; these are NOT real people.
        return ("name,email,phone,ssn\nExample Person,example@example.invalid,202-555-0100,000-12-3456\n"
                if csv_file else "Name: Example Person\nEmail: example@example.invalid\nPhone: 202-555-0100\n")
    return "topic,value\nproject,synthetic documentation\n" if csv_file else "General project notes. Configure the blue theme. No personal records here.\n"


def generate():
    proportions = [config.TRAIN_SPLIT, config.VALIDATION_SPLIT, config.TEST_SPLIT]
    if min(proportions) <= 0 or abs(sum(proportions) - 1) > 1e-8:
        raise ValueError("Positive split proportions must sum to 1")
    rng = random.Random(config.RANDOM_SEED)
    groups = {key: [] for key in ("train", "validation", "test")}
    # Stratify by content label so every split has both classes.
    for label in (0, 1):
        families = [(i, p) for i, (p, y) in enumerate(FAMILIES) if y == label]
        rng.shuffle(families)
        a = int(len(families) * config.TRAIN_SPLIT)
        b = a + int(len(families) * config.VALIDATION_SPLIT)
        for split, subset in zip(groups, (families[:a], families[a:b], families[b:])):
            for family, template in subset:
                for _ in range(config.EXAMPLES_PER_FAMILY):
                    relative = template.format(id=f"{rng.getrandbits(48):012x}")
                    text = content_for(label, relative.endswith(".csv"))
                    target = config.DATA_DIRECTORY / "corpus" / split / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(text, encoding="utf-8")
                    groups[split].append(dict(path=relative, label=label, family=family))
    config.DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
    (config.DATA_DIRECTORY / "dataset.json").write_text(json.dumps(groups, indent=2), encoding="utf-8")
    demo = {"Documents/employees.csv": content_for(1, True),
            "Documents/personal_notes.txt": content_for(0), "Calendar/schedule.txt": content_for(1),
            "Meeting Notes/summary.txt": content_for(0), "Misc/item.txt": content_for(1),
            "settings/config.txt": content_for(0), "unsupported.bin": "synthetic only",
            ".gitignore": "private/payroll.csv\ncontacts/\n"}
    for relative, text in demo.items():
        target = config.TEST_DIRECTORY / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    print("Generated synthetic dataset:", {k: len(v) for k, v in groups.items()})
    return groups


if __name__ == "__main__":
    generate()
