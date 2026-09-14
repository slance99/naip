# =============================================================================
# launch_gee_watermask.py
#
# Interactive launcher for gee_watermask_only.py
# Scans available rivers and GPKG folders and prompts you to choose,
# then generates and optionally submits the sbatch command.
#
# Usage:
#   python launch_gee_watermask.py
# =============================================================================

import subprocess
from pathlib import Path

# =============================================================================
# CONFIG — match these to your actual paths
# =============================================================================

DATA_ROOT    = Path("/home/geomorph/california_rivers/naip")
GPKG_DIR     = DATA_ROOT / "gpkgs" / "all"
GEE_NAIP_DIR = DATA_ROOT / "gee_naip"
BATCH_SCRIPT = Path("/home/geomorph/california_rivers/naip/batch_files/gee/gee_watermask_only.sh")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def list_options(directory, description, suffix=None):
    """
    List subdirectories (or folders matching a suffix) and prompt user to pick.
    Returns the chosen folder name as a string.
    """
    if not directory.exists():
        print(f"ERROR: {description} directory not found: {directory}")
        raise SystemExit(1)

    options = sorted([
        d.name for d in directory.iterdir()
        if d.is_dir()
        and not d.name.startswith(".")
        and (suffix is None or d.name.endswith(suffix))
    ])

    if not options:
        print(f"ERROR: No options found in {directory}"
              + (f" ending in '{suffix}'" if suffix else ""))
        raise SystemExit(1)

    print(f"\n{'='*50}")
    print(f"{description}:")
    print(f"{'='*50}")
    for i, opt in enumerate(options, 1):
        print(f"  {i:3}. {opt}")

    print()
    while True:
        try:
            choice = input(f"Enter number (1-{len(options)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx]
            else:
                print(f"  Please enter a number between 1 and {len(options)}")
        except ValueError:
            print("  Please enter a valid number")
        except KeyboardInterrupt:
            print("\nCancelled.")
            raise SystemExit(0)


def list_gpkg_folders(river):
    """
    List GPKG folders that match the chosen river name.
    Filters to folders starting with the river name ending in _gpkgs.
    Falls back to showing all _gpkgs folders if none match.
    """
    if not GPKG_DIR.exists():
        print(f"ERROR: GPKG directory not found: {GPKG_DIR}")
        raise SystemExit(1)

    # First try folders matching the river name
    matching = sorted([
        d.name for d in GPKG_DIR.iterdir()
        if d.is_dir()
        and d.name.endswith("_gpkgs")
        and river in d.name
    ])

    # Fall back to all _gpkgs folders if nothing matches
    if not matching:
        print(f"\n  No GPKG folders found containing '{river}' — showing all:")
        matching = sorted([
            d.name for d in GPKG_DIR.iterdir()
            if d.is_dir() and d.name.endswith("_gpkgs")
        ])

    if not matching:
        print(f"ERROR: No _gpkgs folders found in {GPKG_DIR}")
        raise SystemExit(1)

    print(f"\n{'='*50}")
    print(f"GPKG folders for '{river}':")
    print(f"{'='*50}")
    for i, folder in enumerate(matching, 1):
        # Count GPKGs inside to give user more info
        gpkg_count = len(list((GPKG_DIR / folder).glob("*.gpkg")))
        print(f"  {i:3}. {folder}  ({gpkg_count} gpkg files)")

    print()
    while True:
        try:
            choice = input(f"Enter number (1-{len(matching)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(matching):
                return matching[idx]
            else:
                print(f"  Please enter a number between 1 and {len(matching)}")
        except ValueError:
            print("  Please enter a valid number")
        except KeyboardInterrupt:
            print("\nCancelled.")
            raise SystemExit(0)


def confirm(prompt):
    """Simple yes/no prompt."""
    while True:
        try:
            answer = input(f"{prompt} [y/n]: ").strip().lower()
            if answer in ("y", "yes"):
                return True
            if answer in ("n", "no"):
                return False
            print("  Please enter y or n")
        except KeyboardInterrupt:
            print("\nCancelled.")
            raise SystemExit(0)


# =============================================================================
# MAIN
# =============================================================================

print("\n" + "="*50)
print("  GEE Watermask Processor Launcher")
print("="*50)

# Step 1 — choose river from gee_naip folder
print(f"\nScanning rivers in: {GEE_NAIP_DIR}")
river = list_options(GEE_NAIP_DIR, "Available rivers")
print(f"\n  Selected river: {river}")

# Step 2 — choose GPKG folder
print(f"\nScanning GPKG folders in: {GPKG_DIR}")
gpkgs = list_options(GPKG_DIR, "Available GPKG folders", suffix="_gpkgs")
print(f"\n  Selected GPKGs: {gpkgs}")

# Derive outputs folder name automatically (same logic as the script)
outputs = gpkgs.replace("_gpkgs", "_outputs")
print(f"  Derived outputs: {outputs}")

# Step 3 — show key config values from the script for confirmation
print(f"\n{'='*50}")
print("Job configuration:")
print(f"{'='*50}")
print(f"  River:          {river}")
print(f"  GPKGs folder:   {gpkgs}")
print(f"  Outputs folder: {outputs}")
print(f"  NAIP tiles:     {DATA_ROOT / 'gee_naip' / river}")
print(f"  Masks output:   {DATA_ROOT / 'outputs' / 'gee' / outputs}")
print(f"  Batch script:   {BATCH_SCRIPT}")

# Step 4 — build sbatch command
sbatch_cmd = f"sbatch {BATCH_SCRIPT} {river} {gpkgs}"

print(f"\n  Command to run:")
print(f"  {sbatch_cmd}")

# Step 5 — confirm and submit
print()
if confirm("Submit this job?"):
    result = subprocess.run(
        sbatch_cmd, shell=True, capture_output=True, text=True
    )
    if result.returncode == 0:
        print(f"\n  {result.stdout.strip()}")
        print(f"  Monitor with: tail -f logs/*.out")
    else:
        print(f"\nERROR submitting job:")
        print(f"  {result.stderr.strip()}")
else:
    print(f"\nNot submitted. To run manually:")
    print(f"  {sbatch_cmd}")

print()
