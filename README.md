# Beach Drone Detection

Software for a drone that flies over a beach and autonomously detects, from
top-down aerial video:

- Rip currents
- Sharks & marine life
- Swimmer distress
- Isolated swimmers
- Vessel encroachment
- Water quality issues / algal blooms

## Project layout

```
beach_drone_detection/   # the actual Python package (importable code)
  detectors/              # one file per detection task (rip_currents.py, etc.)
  pipeline/                # video input handling (reading a file or a live feed)
  utils/                   # shared helper code
configs/                  # settings files (thresholds, paths, camera params)
data/
  raw/                     # untouched source video/images (not committed to git)
  processed/               # cleaned/labeled data ready for training (not committed)
models/                    # trained model weight files (not committed to git)
notebooks/                 # scratch space for experiments/exploration
scripts/                   # small command-line entry points, e.g. "run on this video"
tests/                     # automated tests
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate      # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Status

Scaffolding only — detector modules are stubs (`NotImplementedError`) that
define the shape each detection task will take. No libraries installed yet.
