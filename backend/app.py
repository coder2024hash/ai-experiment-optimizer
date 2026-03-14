from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    accuracy_score,
    f1_score,
)
import numpy as np
from datetime import datetime, UTC
import json
import wikipediaapi
from typing import Optional
import time


# Directory setup
BASE = Path(__file__).resolve().parent
UPLOAD_DIR = BASE / "data"
UPLOAD_DIR.mkdir(exist_ok=True)

EXPERIMENTS_DIR = BASE / "experiments"
EXPERIMENTS_DIR.mkdir(exist_ok=True)
EXPERIMENTS_LOG = EXPERIMENTS_DIR / "experiments.jsonl"


app = FastAPI(
    title="AI Experiment Optimizer API",
    description="Automated ML platform for scientific experimentation",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "AI Experiment Optimizer API - Running!"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        dest = UPLOAD_DIR / file.filename
        with open(dest, "wb") as buffer:
            buffer.write(await file.read())

        df = pd.read_csv(dest)
        preview = df.head(5).to_dict(orient="records")
        columns = df.columns.tolist()
        # Numeric columns = possible targets
        candidates = df.select_dtypes(include="number").columns.tolist()
        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

        return {
            "filename": file.filename,
            "rows": len(df),
            "columns": columns,
            "dtypes": dtypes,
            "preview": preview,
            "suggested_targets": candidates,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Upload failed: {str(e)}")


def infer_problem_type_from_target(y: pd.Series) -> str:
    """
    Infer 'regression' vs 'classification' from target column.
    - Non-numeric -> classification
    - Numeric with very few unique values relative to samples -> classification
    - Otherwise -> regression
    """
    n_samples = len(y)
    n_unique = y.nunique(dropna=True)

    # Non-numeric target => classification
    if not pd.api.types.is_numeric_dtype(y):
        return "classification"

    # Pure 0/1 or 0/1/2 style small integer codes AND small unique count => classification
    if pd.api.types.is_integer_dtype(y) and n_unique <= min(10, int(0.05 * n_samples)):
        return "classification"

    # Otherwise treat as regression
    return "regression"

# Initialize Wikipedia client (English)
wiki_client = wikipediaapi.Wikipedia(
    language="en",
    user_agent="AIExperimentOptimizer/1.0 (contact: skillset2023@gmail.com)",
)
def get_wikipedia_summary(topic: str, max_chars: int = 600) -> dict:
    """
    Fetch a short summary + URL from Wikipedia for a given topic.
    Returns empty dict if page not found.
    """
    if not topic or not topic.strip():
        return {}

    page = wiki_client.page(topic.strip())
    if not page.exists():
        return {}

    summary = page.summary.strip()
    if len(summary) > max_chars:
        summary = summary[:max_chars].rsplit(" ", 1)[0] + "..."

    return {
        "topic": topic,
        "summary": summary,
        "url": page.fullurl,
    }


@app.post("/inspect")
async def inspect_file(filename: str = Form(...)):
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Uploaded file not found")

    try:
        df = pd.read_csv(file_path)

        # Basic info
        n_rows, n_cols = df.shape
        dtypes = {col: str(dt) for col, dt in df.dtypes.items()}

        # Simple stats for numeric
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_summary = {}
        for col in numeric_cols:
            col_series = df[col]
            numeric_summary[col] = {
                "min": float(col_series.min()),
                "max": float(col_series.max()),
                "mean": float(col_series.mean()),
                "std": float(col_series.std()),
                "n_missing": int(col_series.isna().sum()),
            }

        # Unique counts for all columns
        unique_counts = {col: int(df[col].nunique(dropna=True)) for col in df.columns}

        # Heuristic candidate targets:
        #  - last column
        #  - any column with relatively few unique values or that looks like an outcome
        candidate_targets = []
        if len(df.columns) > 0:
            candidate_targets.append(df.columns[-1])

        for col in df.columns:
            if col.lower() in ["target", "label", "output", "result", "yield"]:
                if col not in candidate_targets:
                    candidate_targets.append(col)

        # Add numeric columns with not-too-many unique values
        for col in numeric_cols:
            if unique_counts[col] <= max(20, int(0.2 * n_rows)):
                if col not in candidate_targets:
                    candidate_targets.append(col)

        # Remove obvious ID-like columns from candidates
        id_like = [
            col for col in df.columns
            if unique_counts[col] == n_rows  # all unique
        ]
        candidate_targets = [c for c in candidate_targets if c not in id_like]

        preview = df.head(5).to_dict(orient="records")

        return {
            "filename": filename,
            "rows": n_rows,
            "cols": n_cols,
            "dtypes": dtypes,
            "unique_counts": unique_counts,
            "numeric_summary": numeric_summary,
            "candidate_targets": candidate_targets,
            "id_like_columns": id_like,
            "preview": preview,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inspect failed: {str(e)}")

def get_dataset_meta(filename: str) -> dict:
    """Compute simple meta-features from a dataset."""
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    df = pd.read_csv(file_path)
    
    # Basic counts
    n_rows = len(df)
    n_cols = len(df.columns)
    
    # Numeric vs categorical
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    n_numeric = len(numeric_cols)
    n_categorical = len(cat_cols)
    
    # Missing values
    total_missing = df.isnull().sum().sum()
    missing_ratio = total_missing / (n_rows * n_cols)
    
    # Target stats (last column, or first numeric)
    target_col = df.columns[-1] if len(df.columns) > 0 else ""
    target_stats = {}
    if target_col in df.columns:
        target_series = df[target_col]
        target_stats = {
            "target_mean": float(target_series.mean()),
            "target_std": float(target_series.std()),
            "target_n_unique": int(target_series.nunique()),
            "target_is_numeric": pd.api.types.is_numeric_dtype(target_series),
        }
    
    return {
        "filename": filename,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_numeric": n_numeric,
        "n_categorical": n_categorical,
        "missing_ratio": float(missing_ratio),
        "target_stats": target_stats,
    }
@app.post("/dataset_meta")
async def get_dataset_meta_endpoint(filename: str = Form(...)):
    return get_dataset_meta(filename)

@app.post("/train")
async def train(
    filename: str = Form(...),
    target_column: str = Form(...),
    problem_type: str = Form("auto"),  # "auto" | "regression" | "classification"ls 0
    test_size: float = Form(0.2),
    random_state: int = Form(42),
    n_estimators: int = Form(200),
    max_depth: Optional[int] = Form(None),
    context_topic: Optional[str] = Form(None),
):
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Uploaded file not found")

    try:
        # 1. Load data
        df = pd.read_csv(file_path)

        if target_column not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"Target column '{target_column}' not found in file",
            )

        # 2. Split X, y
        y = df[target_column]
        X = df.drop(columns=[target_column])

        # 2a. Simple cleanup: drop completely empty columns
        X = X.dropna(axis=1, how="all")

        # 2b. Separate numeric & categorical features
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

        # Basic heuristic: drop ID-like columns (all values unique) from features
        drop_cols = []
        for col in X.columns:
            if X[col].nunique(dropna=True) == len(X):
                drop_cols.append(col)
        if drop_cols:
            X = X.drop(columns=drop_cols)
            num_cols = [c for c in num_cols if c not in drop_cols]
            cat_cols = [c for c in cat_cols if c not in drop_cols]

        # 3. Handle missing values
        # Numeric: fill with median
        for col in num_cols:
            X[col] = X[col].fillna(X[col].median())
        # Categorical: fill with mode, then one-hot encode
        for col in cat_cols:
            mode_value = X[col].mode(dropna=True)
            if not mode_value.empty:
                X[col] = X[col].fillna(mode_value.iloc[0])
            else:
                X[col] = X[col].fillna("UNK")

        # One-hot encode categoricals
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

        # 4. Infer problem type if needed
        if problem_type == "auto":
            detected_problem_type = infer_problem_type_from_target(y)
        else:
            detected_problem_type = problem_type

        if detected_problem_type not in {"regression", "classification"}:
            raise HTTPException(
                status_code=400,
                detail="problem_type must be 'auto', 'regression', or 'classification'",
            )

        # 5. Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # 6. Choose and train model
        if detected_problem_type == "regression":
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1,
            )
            model_name = "RandomForestRegressor"
        else:
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1,
            )
            model_name = "RandomForestClassifier"

        # ---- START TIMING ----
        start_time = time.time()

        model.fit(X_train, y_train)

        # 7. Evaluate
        y_pred = model.predict(X_test)

        if detected_problem_type == "regression":
            r2 = r2_score(y_test, y_pred)
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            metrics = {
                "r2_score": float(r2),
                "rmse": rmse,
            }
        else:
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro")
            metrics = {
                "accuracy": float(acc),
                "f1_macro": float(f1),
            }

        # Choose primary metric
        if detected_problem_type == "regression":
            primary_metric_name = "r2_score"
            primary_metric_value = float(metrics["r2_score"])
        else:
            primary_metric_name = "accuracy"
            primary_metric_value = float(metrics["accuracy"])

        # ---- END TIMING ----
        end_time = time.time()
        wall_time = end_time - start_time

        lambda_time = 0.001  # can be tuned later
        score = primary_metric_value - lambda_time * wall_time

        # 8. Feature importance
        importances = getattr(model, "feature_importances_", None)
        if importances is not None:
            importances = importances / importances.sum()
            feature_importance = {
                col: float(imp) for col, imp in zip(X.columns, importances)
            }
        else:
            feature_importance = {}

        # 9. Example predictions table (first 5 from test set)
        preview_count = min(5, len(X_test))
        results_table = []
        for i in range(preview_count):
            results_table.append(
                {
                    "index": int(X_test.index[i]),
                    "features": {
                        col: (
                            float(X_test.iloc[i][col])
                            if np.issubdtype(X_test.dtypes[col], np.number)
                            else str(X_test.iloc[i][col])
                        )
                        for col in X.columns
                    },
                    "actual": (
                        float(y_test.iloc[i])
                        if pd.api.types.is_numeric_dtype(y_test)
                        else str(y_test.iloc[i])
                    ),
                    "predicted": (
                        float(y_pred[i])
                        if isinstance(y_pred[i], (int, float, np.number))
                        else str(y_pred[i])
                    ),
                }
            )

        # 10. Decide Wikipedia topic
        if context_topic and context_topic.strip():
            wikipedia_topic = context_topic.strip()
        else:
            keyword_candidates = []
            for col in df.columns:
                cname = col.lower()
                if any(
                    k in cname
                    for k in [
                        "crop",
                        "disease",
                        "material",
                        "chemical",
                        "battery",
                        "virus",
                        "cell",
                        "city",
                        "country",
                        "species",
                    ]
                ):
                    keyword_candidates.append(col)

            if keyword_candidates:
                wikipedia_topic = keyword_candidates[0]
            else:
                wikipedia_topic = target_column.replace("_", " ")

        knowledge_card = get_wikipedia_summary(wikipedia_topic)

        # 11. Build output
        output = {
            "filename": filename,
            "target": target_column,
            "problem_type": detected_problem_type,
            "model": model_name,
            "n_samples": int(len(df)),
            "n_features": int(X.shape[1]),
            "metrics": metrics,
            "feature_importance": feature_importance,
            "results_table": results_table,
            "summary": f"{model_name} trained on {len(df)} rows with {X.shape[1]} features (test_size={test_size})",
            "knowledge_card": knowledge_card,
        }
         # Add dataset meta
        dataset_meta = get_dataset_meta(filename)

        # 12. Log experiment metadata to JSONL
        now_utc = datetime.now(UTC)
        
        experiment_record = {
            "experiment_id": now_utc.isoformat(),
            "timestamp": now_utc.isoformat(),
            "filename": filename,
            "target": target_column,
            "problem_type": detected_problem_type,
            "model": model_name,
            "n_samples": output["n_samples"],
            "n_features": output["n_features"],
            "metrics": metrics,
            "primary_metric_name": primary_metric_name,
            "primary_metric_value": primary_metric_value,
            "wall_time_seconds": wall_time,
            "compute_seconds": wall_time,
            "score": score,
            "dataset_meta": dataset_meta 
        }

        with open(EXPERIMENTS_LOG, "a") as f:
            f.write(json.dumps(experiment_record) + "\n")

        print("\n--- TRAINING OUTPUT ---")
        print(output)
        print("--- END OUTPUT ---\n")

        return JSONResponse(content=output)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.get("/experiments")
async def list_experiments():
    if not EXPERIMENTS_LOG.exists():
        return []

    records = []
    with open(EXPERIMENTS_LOG, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records
@app.get("/best_experiments")
async def best_experiments(top_k: int = 5):
    if not EXPERIMENTS_LOG.exists():
        return []

    records = []
    with open(EXPERIMENTS_LOG, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Only consider those with a score
            if "score" in rec and rec.get("score") is not None:
                records.append(rec)

    # Sort by score descending
    records.sort(key=lambda r: r.get("score", 0.0), reverse=True)

    return records[:top_k]
@app.post("/compute_experiment_labels")
async def compute_experiment_labels():
    if not EXPERIMENTS_LOG.exists():
        return {"message": "No experiments found"}

    records = []
    with open(EXPERIMENTS_LOG, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if "score" in rec and rec["score"] is not None:
                    records.append(rec)
            except json.JSONDecodeError:
                continue

    if not records:
        return {"message": "No scored experiments"}

    scores = [r["score"] for r in records]
    q30 = float(np.percentile(scores, 30))
    q70 = float(np.percentile(scores, 70))

    for rec in records:
        s = rec["score"]
        if s >= q70:
            rec["quality_label"] = "good"
        elif s >= q30:
            rec["quality_label"] = "medium"
        else:
            rec["quality_label"] = "bad"

    return {
        "n_experiments": len(records),
        "score_quantiles": {"q30": q30, "q70": q70},
        "labeled_examples": records[:5]
    }
@app.post("/suggest_config")
async def suggest_config(filename: str = Form(...)):
    meta = get_dataset_meta(filename)

    candidate_configs = [
        {"n_estimators": 50, "max_depth": None},
        {"n_estimators": 100, "max_depth": 5},
        {"n_estimators": 200, "max_depth": 10},
        {"n_estimators": 400, "max_depth": None},
    ]

    # Placeholder policy: pick the largest n_estimators
    best_cfg = max(candidate_configs, key=lambda c: c["n_estimators"])

    return {
        "filename": filename,
        "dataset_meta_used": meta,
        "suggested_config": best_cfg,
    }
@app.post("/run_experiment_batch")
async def run_experiment_batch(
    filename: str = Form(...),
    target_column: str = Form(...),
    problem_type: str = Form("auto"),
    test_size: float = Form(0.2),
    random_state: int = Form(42),
    n_runs: int = Form(5),
    context_topic: Optional[str] = Form(None),
):
    results = []

    for _ in range(n_runs):
        suggestion = await suggest_config(filename=filename)
        suggested = suggestion["suggested_config"]
        n_estimators = suggested["n_estimators"]
        max_depth = suggested["max_depth"]

        resp = await train(
            filename=filename,
            target_column=target_column,
            problem_type=problem_type,
            test_size=test_size,
            random_state=random_state,
            n_estimators=n_estimators,
            max_depth=max_depth,
            context_topic=context_topic,
        )
        results.append(json.loads(resp.body.decode("utf-8")))

    return results
