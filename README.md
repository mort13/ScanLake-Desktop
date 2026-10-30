# ScanLake Desktop

Desktop app that reads the Star Citizen mining-scan HUD and uploads the scans to ScanLake.
It writes exactly the data the web app ([ScanLake Capture](https://github.com/mort13/ScanLake-Capture))
writes, in the format published by [ScanLake Format](https://github.com/mort13/ScanLake-Format),
but runs the CNNs natively (ONNX Runtime with DirectML or CPU, or PyTorch), batches all ROIs of a
frame, and votes over several frames before you stage a scan.

- **Standard mode** (players): capture and upload.
- **Dev mode** (`--dev` or Settings → Advanced): profile/ROI editor, labelers, training, export.

## Layout

| Path | What |
|---|---|
| `packages/scanlake-core` | Format, profiles, anchors, ROIs, preprocessing, CTC decode, voting, validation (numpy + OpenCV only) |
| `packages/scanlake-inference` | Batched inference backends: ONNX Runtime (DirectML/CPU), PyTorch |
| `packages/scanlake-models` | PyTorch model definitions (DigitCRNN, WordCNN) and checkpoints |
| `packages/scanlake-io` | Format-driven Parquet writer, Store/Gateway clients, `scanlake-data` CLI |
| `packages/scanlake-training` | Datasets, synthetic data, training, ONNX export, `scanlake-train` CLI |
| `apps/desktop` | The PySide6 app |
| `assets/assets.lock.json` | Pinned profiles, anchors, models and format (binaries are fetched, never committed) |
| `packaging/` | Asset fetcher, PyInstaller specs, installer |

## Development

Needs [uv](https://docs.astral.sh/uv/). Python 3.13 is installed by uv if missing.

```bash
python packaging/fetch_assets.py   # download + verify pinned assets into assets/
uv sync                            # dev env: everything incl. PyTorch (CPU) and training
uv run pytest
uv run ruff check . && uv run pyright
```

`python packaging/fetch_assets.py --update` re-downloads every asset URL and re-pins the lock,
e.g. after a new model or profile was published.

The player build installs only the ORT runtime:

```bash
uv sync --package scanlake-desktop --no-dev
```

Run the app from source with `uv run scanlake-desktop` (add `--dev` for dev mode, `--replay
FOLDER` to read saved frames instead of the screen).

### Endpoints

| Service | Default | Override |
|---|---|---|
| Store (uploads) | `https://scanlake-store.mo-55d.workers.dev` | `SCANLAKE_STORE_URL` |
| Gateway (format, assets) | `https://gateway.scanlake.rocks` | `SCANLAKE_GATEWAY_URL` |
| Auth (Discord login) | `https://auth.scanlake.rocks` | `SCANLAKE_AUTH_URL` |
| Format | `{gateway}/api/format` | `SCANLAKE_FORMAT_URL` |

## The upload format

What is uploaded is defined in [ScanLake Format](https://github.com/mort13/ScanLake-Format)
(`format.json`), fetched at start-up. Every column is filled from the app (session, user,
location), from the HUD (a ROI of the ship profile, bound by the leaf's `column` in the
profile's `output_schema`), or computed. `scanlake-data coverage <profile.json>` shows which.

To capture something new: add the column to `format.json` (as `"required": false` first), then
in dev mode draw a ROI in the profile editor and let it fill that column. No code change.

## Dev mode

| Dev menu | What |
|---|---|
| Pipeline debug window | The frame, anchors and ROIs as the pipeline sees them; **Save snapshot** writes a golden-test fixture (frame + what was read) |
| Profile editor | Move/draw ROIs and cut anchors on a frozen frame, bind ROIs to format columns, see format coverage, test-read; saved profiles override the bundled ones |
| Labeler | Review the ROI crops kept from saved scans (Settings → Advanced → keep ROI crops) into `training/real`, the honest validation set |
| Training | Generate synthetic data, train, watch the curves, export to ONNX, compare with the current model, install it (needs PyTorch) |
| Prepare asset release | Package profiles, anchors and models (with your edits) plus the manifest players update from |

Dev data lives in `%LOCALAPPDATA%\ScanLake\Desktop` (`assets/` overrides, `training/`, `publish/`).

### Training from the command line

```bash
uv run scanlake-train synth strips data/strips --per-format 1000
uv run scanlake-train train crnn --data data/strips --val-data <real crops> --epochs 40 --patience 8
uv run scanlake-train export runs/crnn-…/digit_crnn out/digit_crnn     # ONNX + verification
uv run scanlake-train evaluate out/digit_crnn --data <labelled folder>
uv run scanlake-train checkpoint convert "<Rock Capture CNN>/data/models/crnn_model.pth" assets/models/digit_crnn
```

Without `--val-data`, a fixed 15 % of the training folder (by file name hash) validates. The
exported ONNX files keep the web app's names and metadata, so they work in ScanLake Capture too.

## Asset updates

Players get new profiles, anchors and models without a new app: the app reads
`{gateway}/public/desktop/manifest.json` at start-up, downloads files it does not have from
`/public/desktop/files/<sha256>`, verifies them, and uses the new set from the next start
(Settings → Advanced resets to the installed set). Build a release in dev mode or with
`scanlake-data manifest <folder> --version <v>`, then run the printed `wrangler r2 object put`
commands (files first, manifest last). The Gateway must serve `*manifest.json` with
`Cache-Control: no-cache`.

## Packaging

```bash
python packaging/build.py ort    # player build: ONNX Runtime only (~460 MB), self-tested
python packaging/build.py full   # + PyTorch engine and dev-mode training (several GB)
```

Output in `dist/`. With [Inno Setup](https://jrsoftware.org/isinfo.php) installed the script
also builds a per-user installer (`packaging/installer/scanlake-desktop.iss`, no admin rights).
Settings and unsent scans survive an uninstall.
