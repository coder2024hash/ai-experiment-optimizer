import json
from pathlib import Path
from backend.database import SessionLocal, ModelMeta, init_db

def save_model_meta(meta):
    """
    Save model metadata to database
    
    Args:
        meta: dictionary containing model metadata
    """
    init_db()
    session = SessionLocal()
    try:
        obj = ModelMeta(
            model_id=meta.get("id"),
            file_name=meta.get("file_name"),
            model_path=meta.get("model_path"),
            target_column=meta.get("target_column"),
            problem_type=meta.get("problem_type"),
            metrics=json.dumps(meta.get("metrics", {})),
            feature_columns=json.dumps(meta.get("feature_columns", [])),
            feature_importance=json.dumps(meta.get("feature_importance", {})),
            n_estimators=meta.get("n_estimators"),
            max_depth=meta.get("max_depth"),
        )
        session.add(obj)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()

def list_models_from_db():
    """
    List all trained models
    
    Returns:
        list of model metadata dictionaries
    """
    init_db()
    session = SessionLocal()
    try:
        rows = session.query(ModelMeta).order_by(ModelMeta.created_at.desc()).all()
        out = []
        for r in rows:
            out.append({
                "model_id": r.model_id,
                "file_name": r.file_name,
                "model_path": r.model_path,
                "target_column": r.target_column,
                "problem_type": r.problem_type,
                "metrics": json.loads(r.metrics) if r.metrics else {},
                "feature_columns": json.loads(r.feature_columns) if r.feature_columns else [],
                "feature_importance": json.loads(r.feature_importance) if r.feature_importance else {},
                "n_estimators": r.n_estimators,
                "max_depth": r.max_depth,
                "created_at": r.created_at.isoformat()
            })
        return out
    finally:
        session.close()

def get_model_by_id(model_id):
    """
    Get model metadata by ID
    
    Args:
        model_id: unique model identifier
    
    Returns:
        model metadata dictionary or None
    """
    init_db()
    session = SessionLocal()
    try:
        r = session.query(ModelMeta).filter(ModelMeta.model_id == model_id).first()
        if not r:
            return None
        return {
            "model_id": r.model_id,
            "file_name": r.file_name,
            "model_path": r.model_path,
            "target_column": r.target_column,
            "problem_type": r.problem_type,
            "metrics": json.loads(r.metrics) if r.metrics else {},
            "feature_columns": json.loads(r.feature_columns) if r.feature_columns else [],
            "feature_importance": json.loads(r.feature_importance) if r.feature_importance else {},
            "n_estimators": r.n_estimators,
            "max_depth": r.max_depth,
            "created_at": r.created_at.isoformat()
        }
    finally:
        session.close()

def delete_model(model_id):
    """
    Delete model from database and filesystem
    
    Args:
        model_id: unique model identifier
    
    Returns:
        True if deleted, False if not found
    """
    init_db()
    session = SessionLocal()
    try:
        r = session.query(ModelMeta).filter(ModelMeta.model_id == model_id).first()
        if not r:
            return False
        
        # Delete model file
        model_path = Path(r.model_path)
        if model_path.exists():
            model_path.unlink()
        
        # Delete from database
        session.delete(r)
        session.commit()
        return True
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
