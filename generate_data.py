"""Generate fictional contents first; label by the selected content, never by filename."""
import json
import random
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
    # Added for variety: everyday names people actually give files.
    ("downloads/bank_statement_{id}.csv", 1), ("documents/tax_return_{id}.txt", 1),
    ("backup/address_book_{id}.csv", 1), ("hr/applicants/resume_{id}.txt", 1),
    ("medical/patient_records_{id}.csv", 1), ("family/emergency_contacts_{id}.txt", 1),
    ("src/main_{id}.txt", 0), ("logs/server_{id}.txt", 0), ("recipes/dinner_{id}.txt", 0),
    ("photos/vacation_plan_{id}.txt", 0), ("data/measurements_{id}.csv", 0),
    ("templates/blank_form_{id}.txt", 0),
]


# Word pools for programmatic families. Labels are chosen FIRST (the contents are PII-bearing or
# benign); words are then drawn with realistic overlap, including misleading decoys, so a
# filename keyword alone never settles the label.
SENSITIVE_FOLDERS = ["hr", "finance", "people", "staff", "accounts", "records", "admin", "private", "crm",
                     "clinic", "school", "payroll", "legal", "insurance", "recruiting", "benefits", "tax"]
SENSITIVE_NAMES = ["employees", "personnel", "payroll", "customers", "contacts", "salary", "roster", "applicants",
                   "patients", "members", "clients", "tenants", "students", "donors", "subscribers", "residents",
                   "beneficiaries", "guests", "drivers", "claimants", "borrowers", "volunteers", "candidates"]
NEUTRAL_FOLDERS = ["documents", "downloads", "desktop", "backup", "misc", "old", "shared", "archive", "projects",
                   "data", "exports", "temp", "stuff", "new_folder"]
NEUTRAL_NAMES = ["item", "data", "export", "list", "records", "dump", "sheet", "file", "report", "copy", "notes", "final"]
BENIGN_FOLDERS = ["docs", "help", "src", "build", "assets", "examples", "public", "release", "tools", "tests",
                  "themes", "fonts", "icons", "vendor", "samples", "tutorials", "recipes", "music", "games"]
BENIGN_NAMES = ["manual", "config", "readme", "guide", "tutorial", "design", "weather", "colors", "api", "changelog",
                "license", "menu", "inventory_parts", "benchmarks", "lyrics", "translations", "settings", "schedule_public",
                "release_notes", "shopping_list", "playlist", "budget_template", "meeting_agenda", "style_guide"]


# Other languages (accents are folded by the encoder: nómina -> nomina).
SENSITIVE_FOLDERS += ["rrhh", "finanzas", "personalabteilung", "ressources_humaines", "risorse_umane", "recursos_humanos"]
SENSITIVE_NAMES += ["empleados", "nómina", "clientes", "mitarbeiter", "gehälter", "kunden", "salaires", "employés",
                    "paie", "dipendenti", "stipendi", "funcionários", "salários", "contatos", "pacientes"]
BENIGN_FOLDERS += ["documentacion", "dokumentation", "aide", "ajuda", "esempi", "plantillas"]
BENIGN_NAMES += ["manual_usuario", "configuración", "recetas", "handbuch", "einstellungen", "recettes",
                 "notizie", "documentação", "receitas", "fahrplan_öffentlich"]
MONTHS = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
# Extensions are shared by both labels except for a mild tilt, so the type alone settles little.
POSITIVE_EXTENSIONS = [".csv", ".txt", ".xlsx", ".docx", ".csv", ".txt", ".tsv", ".pdf", ".log"]
NEGATIVE_EXTENSIONS = [".txt", ".md", ".xlsx", ".docx", ".csv", ".txt", ".json", ".pdf", ".log"]


def template_for(folder, name, rng):
    """Realistic shapes: dated, versioned, deeply nested and whole-folder families."""
    style = rng.random()
    if style < 0.15:  # Whole-folder families such as people/staff/{id}.csv.
        return f"{folder}/{name}/{{id}}"
    if style < 0.30:  # Dated: "2024-03 payroll {id}".
        return f"{folder}/{rng.choice(['2022', '2023', '2024', '2025'])}-{rng.choice(MONTHS)} {name.replace('_', ' ')} {{id}}"
    if style < 0.40:  # Versioned copies: "payroll_final (2)".
        return f"{folder}/{name}_{rng.choice(['final', 'v2', 'old', 'copy', 'backup'])} ({{id}})"
    if style < 0.55:  # Deeper trees, the way real drives look.
        return f"{rng.choice(NEUTRAL_FOLDERS)}/{folder}/{rng.choice(NEUTRAL_FOLDERS)}/{name}_{{id}}"
    return f"{folder}/{name}_{{id}}"


def build_families(per_label=90):
    """Reproducible extra families, independent of RANDOM_SEED so the family list is stable."""
    rng = random.Random(1936)
    families = list(FAMILIES)
    known = {template for template, _ in families}
    for label in (1, 0):
        added = 0
        while added < per_label:
            if label:
                folder = rng.choice(SENSITIVE_FOLDERS if rng.random() < 0.65 else NEUTRAL_FOLDERS)
                name = rng.choice(SENSITIVE_NAMES if rng.random() < 0.75 else NEUTRAL_NAMES)
            else:
                folder = rng.choice(BENIGN_FOLDERS if rng.random() < 0.7 else NEUTRAL_FOLDERS if rng.random() < 0.7 else SENSITIVE_FOLDERS)
                name = rng.choice(BENIGN_NAMES if rng.random() < 0.75 else NEUTRAL_NAMES if rng.random() < 0.6 else SENSITIVE_NAMES)
            template = template_for(folder, name, rng)
            template += rng.choice(POSITIVE_EXTENSIONS if label else NEGATIVE_EXTENSIONS)
            if template not in known:
                known.add(template)
                families.append((template, label))
                added += 1
    return families


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
    # File sizes are drawn independently of the label (only feedback from real files can make
    # size informative); they feed the optional metadata features.
    size_rng = random.Random(config.RANDOM_SEED + 1)
    groups = {key: [] for key in ("train", "validation", "test")}
    # Stratify by content label so every split has both classes.
    all_families = build_families()
    for label in (0, 1):
        families = [(i, p) for i, (p, y) in enumerate(all_families) if y == label]
        rng.shuffle(families)
        a = int(len(families) * config.TRAIN_SPLIT)
        b = a + int(len(families) * config.VALIDATION_SPLIT)
        for split, subset in zip(groups, (families[:a], families[a:b], families[b:])):
            for family, template in subset:
                for _ in range(config.EXAMPLES_PER_FAMILY):
                    relative = template.format(id=f"{rng.getrandbits(48):012x}")
                    text = content_for(label, relative.endswith((".csv", ".tsv")))
                    if relative.endswith(".tsv"):
                        text = text.replace(",", "\t")
                    target = config.DATA_DIRECTORY / "corpus" / split / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(text, encoding="utf-8")
                    size = int(size_rng.lognormvariate(8.5, 2.0))
                    groups[split].append(dict(path=relative, label=label, family=family, size=size))
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
