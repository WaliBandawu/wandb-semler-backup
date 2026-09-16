import json

import sftp_backup_lib as sftp_lib

PROJECT_FOLDER = sftp_lib.resolve_backup_base() / "theta-tech-ai_semler-qfhd"
MANIFEST_FILE = PROJECT_FOLDER / "backup_manifest.json"

with open(MANIFEST_FILE) as f:
    manifest = json.load(f)

# Map old folder basename -> run_id, from whatever the manifest recorded.
old_name_to_id = {}
for run_id, record in manifest["runs"].items():
    folder = record.get("folder")
    if folder:
        old_name_to_id[Path(folder).name] = run_id

renamed = 0
skipped_already_id = 0
unmatched = []

for child in sorted(PROJECT_FOLDER.iterdir()):
    if not child.is_dir():
        continue

    name = child.name

    # Already renamed (or somehow already just an id)?
    if name in manifest["runs"]:
        skipped_already_id += 1
        continue

    run_id = old_name_to_id.get(name)

    if run_id is None:
        # Fallback: the id is always the last underscore-separated
        # token appended by the old naming scheme.
        candidate = name.rsplit("_", 1)[-1]
        if candidate in manifest["runs"]:
            run_id = candidate

    if run_id is None:
        unmatched.append(name)
        continue

    target = PROJECT_FOLDER / run_id

    if target.exists():
        print(f"SKIP (target already exists): {name} -> {run_id}")
        continue

    child.rename(target)
    renamed += 1

    record = manifest["runs"].get(run_id)
    if record is not None:
        record["folder"] = str(target)

with open(MANIFEST_FILE, "w") as f:
    json.dump(manifest, f, indent=2, default=str)

print(f"Renamed: {renamed}")
print(f"Already correct: {skipped_already_id}")
print(f"Unmatched (left as-is): {len(unmatched)}")
for name in unmatched:
    print(f"  ? {name}")
