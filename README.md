# CellScope ML: Health Detection Platform

CellScope ML is a web platform for hosting medical-image detection models. Signed-in users pick a model from the dashboard, upload an image, and see the predicted class, how confident the model is, the probability of every class, and a Grad-CAM heatmap.

> **Research and screening aid only. It does not give a diagnosis.** A qualified clinician must review every result.

## Models

| Model | ID | Specialty | Input | Classes |
|---|---|---|---|---|
| Cervical Cytology | `cervical-cytology` | Oncology | Pap smear image | NILM, ASCUS, LSIL, HSIL, SCC |

Cervical cytology is the first model. More will follow; see [Adding a new model](#adding-a-new-model).

## Stack
- **ML:** PyTorch (Apple GPU via MPS, or CUDA, or CPU), timm, scikit-learn, TensorBoard
- **Models:** `scratch_cnn` is our own residual CNN with Squeeze-Excitation, about 4.4M params, trained from random weights. `efficientnet_b0` is an ImageNet-pretrained baseline. The model with the higher test macro-F1 is the one that gets deployed.
- **Serving:** ONNX Runtime for predictions, plus PyTorch for the Grad-CAM heatmaps
- **Backend:** FastAPI (`backend/`)
- **Frontend:** React, Vite, TypeScript, Tailwind, Recharts (`frontend/`)
- **Deploy:** Docker Compose (nginx serves the UI and proxies `/api`)

## 1. Setup
```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements-ml.txt -r backend/requirements.txt -r requirements-dev.txt
cd frontend && npm install && cd ..
```

The steps below train the cervical cytology model. Each class is shown with a risk level: NILM is Low; ASCUS and LSIL are Moderate; HSIL and SCC are High.

## 2. Put the dataset in place
Download the dataset from Kaggle. Then put each class folder under `data/raw/`. Sub-folders inside a class folder are fine.
```
data/raw/ASCUS/...   data/raw/HSIL/...   data/raw/LSIL/...   data/raw/NILM/...   data/raw/SCC/...
```

## 3. Train and evaluate
```bash
python -m ml.prepare_data                  # checks images, removes duplicates, makes the 70/15/15 split, computes normalisation stats
python -m ml.train --model pretrained --overfit-batch   # optional sanity check: should print "OVERFIT CHECK PASSED"
python -m ml.train --model pretrained      # EfficientNet-B0 baseline (~30 epochs)
python -m ml.train --model scratch         # our CNN from scratch (~100 epochs, early stopping)
python -m ml.evaluate                      # test-set metrics, confusion matrices, Grad-CAM samples, report
python -m ml.export                        # winning model -> artifacts/models/cervical-cytology/
tensorboard --logdir artifacts/runs        # optional: training curves
```
**Watching training:** sign in as an admin and open **Training** in the sidebar. Every run started with `python -m ml.train` appears there while it trains, with:
- a progress bar showing the current epoch and batch, and time elapsed and remaining
- loss, accuracy and macro-F1 curves, plus a table of every epoch
- the run's final status: completed, early stopped, stopped (Ctrl+C), failed (with the error), or interrupted (the process died)

The page refreshes every 3 seconds while a run is active. Each run's progress is saved to `artifacts/training/<model_id>/<run_id>.json`.

Hyperparameters live in `ml/config.yaml`. Evaluation writes `artifacts/report.md`, which compares the two models on these metrics:
- accuracy, macro-F1 and ROC-AUC
- **abnormal sensitivity**: how many non-NILM images were flagged as abnormal at all
- **HSIL/SCC recall**

## 4. Create the first admin
Only an administrator can add users, so create one admin from the command line first. It prompts for a password: at least 10 characters, mixing letters with numbers or symbols.
```bash
python -m backend.app.create_admin --username admin
```
Run the same command again to reset that admin's password. Accounts are stored in `data/cellscope.db`. Set `CELLSCOPE_DB` to use a different path.

## 5. Run the portal (development)
```bash
uvicorn backend.app.main:app --port 8000
```
```bash
npm --prefix frontend run dev
```
Open http://localhost:5173 and sign in. The login page has two tabs:
- **User login**: upload images and see results.
- **Admin login**: everything a user can do, plus the **Users** page, where admins add users, reset passwords, and disable or delete accounts.

API docs are at http://localhost:8000/docs. Click **Authorize** there and paste a token to try the protected endpoints.

Calling the API from a script:
```bash
TOKEN=$(curl -s localhost:8000/api/auth/login -H 'Content-Type: application/json' \
  -d '{"username":"admin","password":"<your password>"}' | python -c 'import sys,json;print(json.load(sys.stdin)["access_token"])')
curl -H "Authorization: Bearer $TOKEN" -F file=@slide.jpg localhost:8000/api/models/cervical-cytology/predict
```

How authentication works:
- Passwords are hashed with scrypt.
- Signing in returns a random **bearer token**, valid for 12 hours. Every API call must send it in the `Authorization: Bearer <token>` header. The server stores only a SHA-256 hash of each token, so a leaked copy of the database exposes no usable tokens. The portal keeps the token in the browser tab's `sessionStorage`: it survives a page refresh and is cleared when the tab closes.
- Signing out revokes the token on the server immediately.
- Five failed sign-ins lock that username for 15 minutes.
- Disabling a user or resetting their password signs them out at once.
- There is always at least one active admin: the last one can't be disabled or deleted.

API (everything except health requires sign-in):
- `POST /api/auth/login`, `POST /api/auth/logout`, `GET /api/auth/me`, `POST /api/auth/change-password`
- `GET/POST /api/users`, `PATCH/DELETE /api/users/{id}` (admin only)
- `GET /api/training/runs` and `GET /api/training/runs/{model_id}/{run_id}` (admin only): training progress and per-epoch history
- `GET /api/models` lists every registered model with its status (`ready` or `not_trained`), classes and test metrics. `GET /api/models/{id}` returns one model.
- `POST /api/models/{id}/predict` takes a multipart field `file` (JPG/PNG/BMP/TIFF, up to 10 MB) and an optional `?heatmap=false`.
- `GET /api/health` is a health check.

## 6. Run with Docker
Run this after `python -m ml.export` has filled `artifacts/models/`:
```bash
docker compose up --build
```
Then create the first admin inside the container:
```bash
docker compose exec api python -m backend.app.create_admin --username admin
```
Open http://localhost:8080. Accounts live in the `cellscope-data` volume. Serve the site over HTTPS in production: anyone who sees a bearer token on the network can use it until it expires.

## Tests
```bash
python -m pytest backend
```
The prediction tests need an exported model in `artifacts/models/cervical-cytology/`, or `ARTIFACTS_DIR` pointing at an artifacts folder that has one. Without one they are skipped.

## Adding a new model
1. **Register it** in `models.yaml`. Give it an `id`, `name`, `category`, `modality`, `summary`, `input_hint`, and each class with a `name`, `description` and `risk` (Low, Moderate or High).
2. **Train it** with any pipeline that produces an ONNX image classifier.
3. **Export it** to `artifacts/models/<id>/`:
   - `model.onnx`: input `input` with shape (N, 3, H, W); output `logits`
   - `model_info.json`: `classes` in output order, `image_size`, normalisation `mean` and `std`, `arch`, `model_kind`, and optionally `test` metrics
   - `model.pt` (optional): enables Grad-CAM heatmaps. It must be loadable by `ml.models.load_checkpoint`.

   The included `ml/` pipeline does all of this: set `model_id` in `ml/config.yaml`.
4. Open the dashboard. The model appears in the sidebar and on the Dashboard, marked "Not trained" until its files exist, and needs no restart.

## Layout
```
ml/            data prep, dataset/augmentations, models, train (+ live progress), evaluate, export, gradcam
models.yaml    registry of detection models (names, classes, risk levels)
backend/app/   FastAPI app (main.py), model registry (registry.py), inference, auth + user management (auth.py, security.py, db.py), create_admin CLI
frontend/src/  React dashboard: sidebar layout, Dashboard, one page per model, Training, Users
artifacts/     checkpoints, metrics, plots; served models in artifacts/models/<id>/ (git-ignored)
data/          raw images and generated splits (git-ignored)
```

## License
Released under the [MIT License](LICENSE). Copyright (c) 2026 Priyaranjan Paul.

The dataset is not included in this repository. It keeps its own license, set by its publisher on Kaggle.
