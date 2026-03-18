import io
import streamlit as st
import pandas as pd
from services.services_dataset import (
    get_current_data,
    add_version,
    list_versions,
    rollback_version,
)

def load_current_df(db, dataset_id):
    raw = get_current_data(db, dataset_id)
    return pd.read_csv(io.BytesIO(raw))


def autosave_df(db, dataset_id, df, note: str):
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    add_version(db, dataset_id, buf.getvalue(), note=note)


def dataset_version_sidebar(db, dataset_id):
    st.sidebar.markdown("### 🕘 Dataset Versions")
    versions = list_versions(db, dataset_id)

    labels = {
        f"v{v.version_number} — {v.note or ''}": v.version_number
        for v in versions
    }

    choice = st.sidebar.selectbox(
        "Rollback to version",
        list(labels.keys()),
    )

    if st.sidebar.button("↩️ Rollback"):
        rollback_version(db, dataset_id, labels[choice])
        st.success("Rolled back dataset")
        st.rerun()
