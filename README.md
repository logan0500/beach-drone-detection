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

| Detector | Status |
|---|---|
| Sharks & marine life | Trained model wired in (`shark-best.pt`) |
| Rip currents | Trained model wired in (`rip-best.pt`) |
| Isolated swimmers | Trained model wired in (`swimmer-multiclass-best.pt`), distance-based isolation logic on top |
| Swimmer distress | Trained model wired in (`swimmer-distress-best.pt`) — **not yet validated**, treat "drowning" alerts as needing human confirmation |
| Vessel encroachment | Reuses `swimmer-multiclass-best.pt`'s boat/jetski classes; geofencing needs a real swim-zone polygon (`configs/default.yaml`) before it flags anything |
| Water quality / algal blooms | Not started — no model, planned as classical HSV color-thresholding rather than a trained model |

Model weight files go in `models/` (gitignored — not committed; copy them in
locally, they aren't tracked here).

Run on a single image:

```bash
python scripts/run_on_image.py path/to/frame.jpg -o output.jpg
```

Run on a video, frame by frame:

```bash
python scripts/run_on_video.py path/to/flight.mp4 -o output.mp4
```

Both read detector settings (which model, confidence threshold, isolation
distance, swim-zone polygon) from `configs/default.yaml`.
