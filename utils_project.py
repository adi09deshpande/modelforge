# utils_project.py
import os, json
import pandas as pd
import streamlit as st  # type: ignore

def require_auth():
    if not st.session_state.get("user_id"):
        st.warning("Please login from Home to continue.")
        st.stop()  # [10]

def require_project():
    require_auth()
    if not st.session_state.get("project_path"):
        st.warning("Please select or create a project in 'Master' page.")
        st.stop()  # [15]

def get_active_paths():
    require_project()
    base = st.session_state["project_path"]
    return {
        "base": base,
        "working_csv": os.path.join(base, "working_data.csv"),
        "raw_csv": os.path.join(base, "raw_data.csv"),
        "models_dir": os.path.join(base, "models"),
        "history_dir": os.path.join(base, "history"),
        "meta": os.path.join(base, "metadata.json"),
    }  # [15]

def ensure_dirs():
    p = get_active_paths()
    os.makedirs(p["base"], exist_ok=True)
    os.makedirs(p["models_dir"], exist_ok=True)
    os.makedirs(p["history_dir"], exist_ok=True)
    if not os.path.exists(p["meta"]):
        with open(p["meta"], "w") as f:
            f.write('{"history_index": 0}')  # [15]

def load_latest():
    ensure_dirs()
    p = get_active_paths()
    if os.path.exists(p["working_csv"]):
        st.session_state["data"] = pd.read_csv(p["working_csv"])
    else:
        st.session_state["data"] = None  # [15]

def save_working(df: pd.DataFrame):
    ensure_dirs()
    p = get_active_paths()
    df.to_csv(p["working_csv"], index=False)
    st.session_state["data"] = df  # [15]

def snapshot(df: pd.DataFrame, label: str = ""):
    ensure_dirs()
    p = get_active_paths()
    meta_path = p["meta"]
    try:
        with open(meta_path, "r") as f:
            import json; meta = json.load(f)
    except Exception:
        meta = {"history_index": 0}
    idx = int(meta.get("history_index", 0)) + 1
    meta["history_index"] = idx
    with open(meta_path, "w") as f:
        import json; json.dump(meta, f)
    snap = os.path.join(p["history_dir"], f"{idx:04d}_working_data.csv")
    df.to_csv(snap, index=False)
    st.session_state.setdefault("undo_stack", []).append(snap)
    st.session_state["redo_stack"] = []
    return snap  # [16]

def undo():
    stack = st.session_state.get("undo_stack", [])
    if len(stack) < 2:
        return False
    current = stack.pop()
    st.session_state.setdefault("redo_stack", []).append(current)
    prev = stack[-1]
    df = pd.read_csv(prev)
    save_working(df)
    return True  # [16]

def redo():
    rstack = st.session_state.get("redo_stack", [])
    if not rstack:
        return False
    snap = rstack.pop()
    st.session_state.setdefault("undo_stack", []).append(snap)
    df = pd.read_csv(snap)
    save_working(df)
    return True  # [16]
