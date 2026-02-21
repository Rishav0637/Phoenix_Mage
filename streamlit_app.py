import os
import io
import time
import streamlit as st
from analyst_in_a_box import run_pipeline

st.set_page_config(page_title="Analyst in a Box", page_icon="📊", layout="wide")

st.title("📊 Analyst in a Box")
st.write("Upload a CSV and get auto-cleaning, EDA, charts, insights, and a PDF report.")

with st.sidebar:
    st.header("Settings")
    sample_rows = st.number_input("Sample rows (optional, speeds up big files)", min_value=0, value=0, step=50000)
    time_col = st.text_input("Time column (optional)")
    agg = st.selectbox("Time aggregation", ["mean", "sum", "median"], index=0)
    outdir = st.text_input("Artifacts directory", value="artifacts")
    reportdir = st.text_input("Reports directory", value="reports")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

run = st.button("▶️ Run Analysis", type="primary", disabled=(uploaded is None))

if run and uploaded:
    # Save uploaded file to a temp path
    os.makedirs("uploads", exist_ok=True)
    temp_path = os.path.join("uploads", uploaded.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded.getbuffer())

    st.info("Running pipeline... this usually takes a few seconds.")
    start = time.time()

    try:
        res = run_pipeline(
            csv_path=temp_path,
            out_dir=outdir,
            report_dir=reportdir,
            sample_rows=sample_rows or None,
            prefer_time_col=time_col or None,
            agg=agg
        )
        elapsed = time.time() - start
        st.success(f"Done in {elapsed:.1f}s")

        # Show summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Charts generated", len(res["charts"]))
        with col2:
            st.metric("Insights", res["insights_count"])
        with col3:
            st.metric("Time column", res["time_col"] or "Auto/None")

        # Download PDF
        report_path = res["report_path"]
        if os.path.exists(report_path):
            with open(report_path, "rb") as f:
                st.download_button(
                    "⬇️ Download PDF report",
                    data=f.read(),
                    file_name=os.path.basename(report_path),
                    mime="application/pdf"
                )

        # Preview key charts inline
        st.subheader("Preview Charts")
        ncols = 2
        for i in range(0, len(res["charts"]), ncols):
            cols = st.columns(ncols)
            for j in range(ncols):
                if i + j < len(res["charts"]):
                    img_path, caption = res["charts"][i + j]
                    cols[j].image(img_path, caption=caption, use_column_width=True)

        # Outlier note
        if res["outliers_cols"]:
            st.info(f"Columns with many outliers: {', '.join(res['outliers_cols'])}")

    except Exception as e:
        st.error(f"Error: {e}")
