# Perception — perceived urban safety from street-level images

Thesis code (UTEC, Data Science). Research question: how to represent perceived urban safety
with interpretable visual information and spatial context.

**Perceived safety ≠ observed crime.** Place Pulse 2.0 measures human perception; the Lima 2016
records measure reported crime. The two are never mixed without saying so.

## Repository layout

| Folder | Content |
|---|---|
| `benchmark_perception/placepulse2/src/` | Main experimental code: Place Pulse 2.0 benchmark (pairs, splits, feature extraction, ranking heads, evaluation) and every experiment script `expNNN_*.py` |
| `benchmark_perception/placepulse2/tests/` | Unit tests (synthetic data, no network) |
| `benchmark_perception/placepulse2/survey_web/` | Pairwise perception survey for Lima (Google Apps Script + static page) |
| `benchmark_perception/placepulse2/data/` | Small summaries of the dataset, pairs and splits (the full manifests are regenerated) |
| `image-extraction/` | Street View / Mapillary collection for Lima districts (dry run, resume, budget ledger, QA) |
| `graph/` | Inherited Lima pipeline: street graph, crime labels, neighbourhood tensors, classifier |
| `distritos/` | District-level crime data for Lima (read by `graph/` and the Lima experiments) |

The folders refer to each other with relative paths (`REPO_ROOT = parents[3]`, `../distritos`,
`../image-extraction`); do not move or rename them.

## Data (not versioned)

Images, features, checkpoints and results are not in Git. Scripts expect them under
`external_assets/`:

| Asset | Source |
|---|---|
| `placepulse2/raw/place-pulse-2.0.zip` | <https://www.dropbox.com/s/grzoiwsaeqrmc1l/place-pulse-2.0.zip> |
| `placepulse2/raw/votes_clean.csv` | Figshare DOI `10.6084/m9.figshare.11859993` |
| `checkpoints/*.pt` | Project checkpoints (SimCLR / ViT) |

Street View images are not redistributed (Google Maps Platform license); only the collection code
and the district grids are versioned. Outputs go to `benchmark_perception/placepulse2/outputs/`.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch torchvision timm scikit-learn pandas numpy scipy \
    matplotlib networkx osmnx geopandas shapely pyyaml requests pillow pytest
# only for some experiments: open_clip_torch transformers huggingface_hub ultralytics xgboost gdown
```

A GPU is recommended for feature extraction (about 13 h on CPU for three encoders).

## Pipeline (Place Pulse 2.0)

Run from `benchmark_perception/placepulse2/src/`:

```bash
python build_manifest.py            # hashes of the raw assets
python smoke_test_checkpoints.py    # checkpoints load and give the expected dimension
python inspect_dataset.py           # dataset audit -> data/manifests/
python prepare_pairs.py             # decisive "safe" pairs and ties
python split_dataset.py --seed 42   # paper_protocol, image_disjoint, city_holdout
python extract_features.py          # frozen encoder features (see encoders.py)
python train_probe.py               # linear Bradley-Terry ranking head
python evaluate.py                  # pairwise accuracy with bootstrap CI
```

Each experiment is one script `expNNN_<topic>.py`; its docstring states the question it answers,
its inputs and its outputs. Shared code: `utils.py` (paths, seeds, run logging), `probe_utils.py`,
`ranking_head.py`, `v3_common.py`, `segment_graph.py`, `visual_repr.py`, `network_distance.py`,
`lima_views.py`, `figstyle.py`.

## Lima collection

```bash
python image-extraction/streetview_collector.py collect --district lince --spot-source street_sampling --dry-run
```

Configuration in `image-extraction/configs/streetview_lima.yaml`. Requires a Google Maps API key in
`image-extraction/.env` (ignored by Git).

## Tests

```bash
python -m pytest image-extraction/tests benchmark_perception/placepulse2/tests -q
```

Tests that need the full Place Pulse splits are skipped when those files are not present.
