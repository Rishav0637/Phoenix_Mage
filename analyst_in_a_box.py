# analyst_in_a_box.py
# An "Analyst in a Box": Auto-EDA + Insights + PDF report for any CSV.
# Author: Rishav's Copilot

import os
import io
import sys
import math
import argparse
import textwrap
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle

warnings.filterwarnings("ignore")

# -----------------------------
# Utilities
# -----------------------------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def fmt_pct(x):
    return f"{x*100:.1f}%"

def safe_top_values(s, topn=10):
    vc = s.astype(str).value_counts(dropna=False).head(topn)
    return vc

def human_mem(nbytes: int) -> str:
    for unit in ['B','KB','MB','GB','TB']:
        if nbytes < 1024.0:
            return f"{nbytes:3.1f} {unit}"
        nbytes /= 1024.0
    return f"{nbytes:.1f} PB"

# -----------------------------
# Data Loading & Typing
# -----------------------------

def load_csv(path: str, sample_rows: int = None) -> pd.DataFrame:
    # Basic CSV read; if huge, allow sampling to throttle for speed
    if sample_rows:
        return pd.read_csv(path, nrows=sample_rows, low_memory=False)
    return pd.read_csv(path, low_memory=False)

def detect_types(df: pd.DataFrame):
    types = {}
    # Attempt datetime inference for object columns
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            types[col] = "datetime"
        elif pd.api.types.is_numeric_dtype(series):
            types[col] = "numeric"
        elif pd.api.types.is_categorical_dtype(series):
            types[col] = "categorical"
        else:
            # Try parse to datetime (lightweight check)
            if series.dropna().shape[0] > 0:
                try:
                    pd.to_datetime(series.dropna().head(50), errors="raise", infer_datetime_format=True)
                    types[col] = "datetime"
                    df[col] = pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
                except Exception:
                    # If few unique values -> categorical
                    nunique = series.nunique(dropna=True)
                    if nunique <= max(20, int(0.02 * len(series))):
                        types[col] = "categorical"
                    else:
                        types[col] = "text"
            else:
                types[col] = "text"
    return types

# -----------------------------
# Cleaning
# -----------------------------

def impute_dataframe(df: pd.DataFrame, types: dict) -> pd.DataFrame:
    dfc = df.copy()
    for col, t in types.items():
        if t == "numeric":
            if dfc[col].isna().any():
                dfc[col] = dfc[col].fillna(dfc[col].median())
        elif t == "categorical":
            if dfc[col].isna().any():
                mode_val = dfc[col].mode(dropna=True)
                dfc[col] = dfc[col].fillna(mode_val.iloc[0] if not mode_val.empty else "Unknown")
        elif t == "datetime":
            if dfc[col].isna().any():
                dfc[col] = dfc[col].fillna(method="ffill").fillna(method="bfill")
        else:
            # text: leave missing as is or fill with empty string for presentation
            pass
    return dfc

# -----------------------------
# Analysis & Insights
# -----------------------------

def missingness_summary(df: pd.DataFrame):
    miss = df.isna().mean().sort_values(ascending=False)
    return miss

def numeric_summary(df: pd.DataFrame, types: dict):
    num_cols = [c for c,t in types.items() if t=="numeric"]
    if not num_cols:
        return None
    desc = df[num_cols].describe().T
    desc["missing_pct"] = df[num_cols].isna().mean().values
    return desc

def categorical_summary(df: pd.DataFrame, types: dict, topn=10):
    cat_cols = [c for c,t in types.items() if t=="categorical"]
    out = {}
    for c in cat_cols:
        out[c] = safe_top_values(df[c], topn=topn)
    return out

def pick_time_column(df: pd.DataFrame, types: dict):
    dt_cols = [c for c,t in types.items() if t=="datetime"]
    if not dt_cols:
        return None
    # Heuristic: prefer columns named like date/time or with higher uniqueness
    priority = sorted(dt_cols, key=lambda c: (int("date" not in c.lower() and "time" not in c.lower()), -df[c].nunique()))
    return priority[0]

def detect_outliers_iqr(df: pd.DataFrame, types: dict):
    results = {}
    idx_union = set()
    for c, t in types.items():
        if t != "numeric":
            continue
        s = df[c].dropna()
        if s.empty:
            continue
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (df[c] < lower) | (df[c] > upper)
        count = int(mask.sum())
        if count > 0:
            results[c] = {"count": count, "lower": lower, "upper": upper}
            idx_union.update(df.index[mask].tolist())
    return results, idx_union

def correlation_matrix(df: pd.DataFrame, types: dict, max_cols=25):
    num_cols = [c for c,t in types.items() if t=="numeric"]
    if len(num_cols) == 0:
        return None
    if len(num_cols) > max_cols:
        # pick top by variance
        var = df[num_cols].var().sort_values(ascending=False)
        num_cols = var.head(max_cols).index.tolist()
    corr = df[num_cols].corr(method="pearson")
    return corr

def time_series_aggregate(df: pd.DataFrame, time_col: str, types: dict, how="mean", max_numeric=5):
    num_cols = [c for c,t in types.items() if t=="numeric"]
    if not num_cols or time_col is None:
        return None
    # Resampling frequency: auto
    timespan = df[time_col].max() - df[time_col].min()
    if timespan.days > 540:
        rule = "M"
    elif timespan.days > 90:
        rule = "W"
    else:
        rule = "D"
    # pick top numeric by variance
    var = df[num_cols].var().sort_values(ascending=False)
    top = var.head(max_numeric).index.tolist()
    g = df.set_index(time_col)[top].resample(rule).agg(how)
    g = g.dropna(how="all")
    return g

# -----------------------------
# Charting
# -----------------------------

def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close()

def plot_histograms(df, types, out_dir, max_cols=6):
    paths = []
    num_cols = [c for c,t in types.items() if t=="numeric"]
    if not num_cols:
        return paths
    # pick by variance to keep signal high
    var = df[num_cols].var().sort_values(ascending=False)
    for c in var.head(max_cols).index:
        plt.figure(figsize=(6,4))
        plt.hist(df[c].dropna(), bins=30, color="#4472C4", edgecolor="white")
        plt.title(f"Histogram: {c}")
        plt.xlabel(c)
        plt.ylabel("Frequency")
        p = os.path.join(out_dir, f"hist_{c}.png")
        save_fig(p)
        paths.append((p, f"Distribution of {c}"))
    return paths

def plot_boxplots(df, types, out_dir, max_cols=6):
    paths = []
    num_cols = [c for c,t in types.items() if t=="numeric"]
    if not num_cols:
        return paths
    var = df[num_cols].var().sort_values(ascending=False)
    for c in var.head(max_cols).index:
        plt.figure(figsize=(4,4))
        plt.boxplot(df[c].dropna(), vert=True, patch_artist=True,
                    boxprops=dict(facecolor="#70AD47"))
        plt.title(f"Boxplot: {c}")
        plt.ylabel(c)
        p = os.path.join(out_dir, f"box_{c}.png")
        save_fig(p)
        paths.append((p, f"Outlier view for {c} (IQR-driven)"))
    return paths

def plot_categorical_bars(df, types, out_dir, max_cols=5):
    paths = []
    cat_cols = [c for c,t in types.items() if t=="categorical"]
    for c in cat_cols[:max_cols]:
        vc = df[c].astype(str).value_counts().head(15)
        plt.figure(figsize=(7,4))
        vc.sort_values(ascending=True).plot(kind="barh", color="#ED7D31")
        plt.title(f"Top categories: {c}")
        plt.xlabel("Count")
        plt.ylabel(c)
        p = os.path.join(out_dir, f"bar_{c}.png")
        save_fig(p)
        paths.append((p, f"Top categories for {c}"))
    return paths

def plot_correlation(corr, out_dir):
    if corr is None or corr.empty:
        return []
    plt.figure(figsize=(7,6))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Pearson r")
    ticks = np.arange(len(corr.columns))
    plt.xticks(ticks, corr.columns, rotation=90)
    plt.yticks(ticks, corr.index)
    plt.title("Correlation heatmap (numeric)")
    p = os.path.join(out_dir, f"corr_heatmap.png")
    save_fig(p)
    return [(p, "Correlation heatmap among numeric features")]

def plot_time_series(ts_df, out_dir):
    if ts_df is None or ts_df.empty:
        return []
    paths = []
    plt.figure(figsize=(8,4))
    for c in ts_df.columns:
        plt.plot(ts_df.index, ts_df[c], label=c)
    plt.title("Time series trends (aggregated)")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend(ncol=2, fontsize=8)
    p = os.path.join(out_dir, f"time_trends.png")
    save_fig(p)
    paths.append((p, "Time series trends for top numeric features"))
    return paths

# -----------------------------
# Insights (LLM optional)
# -----------------------------

def rule_based_insights(df, df_clean, types, miss, num_desc, cat_summary, outliers, ts_df, corr):
    insights = []

    # Dataset shape and memory
    mem = df.memory_usage(deep=True).sum()
    insights.append(f"Dataset has {len(df):,} rows × {df.shape[1]} columns ({human_mem(mem)} in memory).")

    # Missingness hotspots
    if miss is not None and miss.max() > 0:
        top_miss = miss.head(5)
        parts = [f"{c}: {fmt_pct(p)}" for c,p in top_miss.items() if p > 0]
        if parts:
            insights.append("Highest missingness by column → " + "; ".join(parts) + ".")

    # Numeric distribution flags
    if num_desc is not None:
        skew_candidates = []
        for c in num_desc.index:
            s = df[c].dropna()
            if s.shape[0] >= 100:
                skew = s.skew()
                if abs(skew) > 1:
                    skew_candidates.append((c, skew))
        if skew_candidates:
            skew_candidates = sorted(skew_candidates, key=lambda x: -abs(x[1]))[:5]
            insights.append("Highly skewed numeric features → " +
                            "; ".join([f"{c} (skew={sk:.2f})" for c, sk in skew_candidates]) + ".")

    # Outlier concentration
    if outliers:
        top_out = sorted(outliers.items(), key=lambda kv: kv[1]["count"], reverse=True)[:5]
        insights.append("Outlier-heavy columns → " +
                        "; ".join([f"{c} ({meta['count']} outliers)" for c, meta in top_out]) + ".")

    # Categorical concentration
    if cat_summary:
        dominant = []
        for c, vc in cat_summary.items():
            if vc.sum() > 0:
                top_name = vc.index[0]
                pct = vc.iloc[0] / vc.sum()
                if pct > 0.6 and vc.sum() >= 50:
                    dominant.append((c, top_name, pct))
        if dominant:
            insights.append("Category dominance detected → " +
                            "; ".join([f"{c}: '{val}' ({fmt_pct(p)})" for c,val,p in dominant]) + ".")

    # Time trends
    if ts_df is not None and not ts_df.empty:
        diffs = []
        for c in ts_df.columns:
            series = ts_df[c].dropna()
            if len(series) >= 2:
                change = (series.iloc[-1] - series.iloc[0])
                pct = (change / (abs(series.iloc[0]) + 1e-9))
                diffs.append((c, change, pct))
        if diffs:
            diffs = sorted(diffs, key=lambda x: -abs(x[2]))[:5]
            insights.append("Largest relative changes over time → " +
                            "; ".join([f"{c}: Δ={chg:.2f} ({fmt_pct(abs(pct))})" for c, chg, pct in diffs]) + ".")

    # Correlations
    if corr is not None and not corr.empty:
        pairs = []
        cols = corr.columns
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                r = corr.iloc[i, j]
                if abs(r) >= 0.7:
                    pairs.append((cols[i], cols[j], r))
        if pairs:
            pairs = sorted(pairs, key=lambda x: -abs(x[2]))[:5]
            insights.append("Strong correlations → " +
                            "; ".join([f"{a} ↔ {b} (r={r:.2f})" for a,b,r in pairs]) + ".")

    if not insights:
        insights.append("No major anomalies detected. Distributions appear broadly stable with manageable missingness.")
    return insights

def llm_insights_if_available(context_text: str):
    """
    Optional Azure OpenAI call. Uses env vars:
    AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_API_VERSION
    Gracefully falls back if not configured or openai not installed.
    """
    try:
        import os
        from openai import AzureOpenAI
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        key = os.environ.get("AZURE_OPENAI_API_KEY")
        deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")
        if not (endpoint and key and deployment):
            return None

        client = AzureOpenAI(api_key=key, api_version=api_version, azure_endpoint=endpoint)
        prompt = (
            "You are a senior data analyst. Read the context and produce 6–10 precise, "
            "non-obvious, actionable business insights with metrics, trends, and potential drivers. "
            "Summarize in bullet points. Avoid generic advice.\n\n"
            f"CONTEXT:\n{context_text}\n"
        )
        resp = client.chat.completions.create(
            model=deployment,
            messages=[{"role":"system","content":"You are a highly analytical data expert."},
                      {"role":"user","content":prompt}],
            temperature=0.2,
            max_tokens=800,
        )
        content = resp.choices[0].message.content
        return [line.strip("-• ").strip() for line in content.splitlines() if line.strip()]
    except Exception:
        return None

# -----------------------------
# Reporting (PDF)
# -----------------------------

def table_from_dataframe(df: pd.DataFrame, max_rows=20, max_cols=8):
    df2 = df.copy()
    if df2.shape[0] > max_rows:
        df2 = df2.head(max_rows)
    if df2.shape[1] > max_cols:
        df2 = df2.iloc[:, :max_cols]
    data = [ [str(c) for c in [""] + df2.columns.tolist()] ]
    for idx, row in df2.iterrows():
        data.append([str(idx)] + [str(v) for v in row.values])
    return data

def build_pdf(report_path, meta, miss_tbl, num_tbl, cat_summaries, chart_paths, insights):
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", fontSize=9, leading=11))
    doc = SimpleDocTemplate(report_path, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)

    story = []

    # Title
    story.append(Paragraph("Auto-EDA & Insights Report", styles["Title"]))
    story.append(Paragraph(meta.get("filename", ""), styles["Normal"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    story.append(Spacer(1, 12))

    # Dataset overview
    story.append(Paragraph("<b>Dataset Overview</b>", styles["Heading2"]))
    story.append(Paragraph(f"Rows: {meta['rows']:,} | Columns: {meta['cols']}", styles["Normal"]))
    story.append(Paragraph(f"Memory: {meta['memory']}", styles["Normal"]))
    story.append(Spacer(1, 8))

    # Insights
    story.append(Paragraph("<b>Key Insights</b>", styles["Heading2"]))
    for ins in insights:
        story.append(Paragraph(f"• {ins}", styles["Normal"]))
    story.append(Spacer(1, 12))

    # Missingness
    if miss_tbl is not None:
        story.append(Paragraph("<b>Missingness (Top)</b>", styles["Heading3"]))
        data = [["Column", "Missing %"]]
        for c, p in miss_tbl.head(15).items():
            data.append([c, f"{p*100:.1f}%"])
        t = Table(data, hAlign="LEFT")
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
            ('FONTSIZE', (0,0), (-1,-1), 9),
        ]))
        story.append(t)
        story.append(Spacer(1, 10))

    # Numeric summary
    if num_tbl is not None:
        story.append(Paragraph("<b>Numeric Summary (head)</b>", styles["Heading3"]))
        data = [["", *num_tbl.columns.tolist()]]
        for idx, row in num_tbl.head(15).iterrows():
            data.append([str(idx)] + [f"{v:.3f}" if isinstance(v, (int,float,np.floating)) else str(v) for v in row.values])
        t = Table(data, hAlign="LEFT", repeatRows=1)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
            ('FONTSIZE', (0,0), (-1,-1), 8),
        ]))
        story.append(t)
        story.append(Spacer(1, 10))

    # Categorical tops
    if cat_summaries:
        story.append(Paragraph("<b>Categorical Top Values</b>", styles["Heading3"]))
        for c, vc in list(cat_summaries.items())[:6]:
            story.append(Paragraph(f"<i>{c}</i>", styles["Normal"]))
            data = [["Value", "Count"]]
            for val, cnt in vc.items():
                data.append([str(val), str(int(cnt))])
            t = Table(data, hAlign="LEFT")
            t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
                ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
                ('FONTSIZE', (0,0), (-1,-1), 8),
            ]))
            story.append(t)
            story.append(Spacer(1, 6))

    # Charts
    if chart_paths:
        story.append(PageBreak())
        story.append(Paragraph("<b>Visuals</b>", styles["Heading2"]))
        for p, caption in chart_paths:
            story.append(Paragraph(caption, styles["Normal"]))
            story.append(Image(p, width=480, height=320))
            story.append(Spacer(1, 10))

    doc.build(story)

# -----------------------------
# Orchestrator
# -----------------------------

def run_pipeline(csv_path: str, out_dir: str = "artifacts", report_dir: str = "reports",
                 sample_rows: int = None, prefer_time_col: str = None, agg="mean"):
    ensure_dir(out_dir)
    ensure_dir(report_dir)

    df = load_csv(csv_path, sample_rows=sample_rows)
    raw_types = detect_types(df)
    time_col = prefer_time_col or pick_time_column(df, raw_types)

    miss = missingness_summary(df)
    num_desc = numeric_summary(df, raw_types)
    cat_summ = categorical_summary(df, raw_types, topn=12)
    out_iqr, out_idx = detect_outliers_iqr(df, raw_types)
    corr = correlation_matrix(df, raw_types, max_cols=25)

    # Cleaned copy for charts
    dfc = impute_dataframe(df, raw_types)

    # Time series aggregation (on cleaned)
    ts_df = None
    if time_col:
        ts_df = time_series_aggregate(dfc, time_col, raw_types, how=agg, max_numeric=5)

    # Charts
    charts = []
    charts += plot_histograms(dfc, raw_types, out_dir=out_dir, max_cols=6)
    charts += plot_boxplots(dfc, raw_types, out_dir=out_dir, max_cols=6)
    charts += plot_categorical_bars(dfc, raw_types, out_dir=out_dir, max_cols=5)
    charts += plot_correlation(corr, out_dir=out_dir)
    charts += plot_time_series(ts_df, out_dir=out_dir)

    # Insights (rule-based + optional LLM)
    rule_insights = rule_based_insights(df, dfc, raw_types, miss, num_desc, cat_summ, out_iqr, ts_df, corr)

    # Prepare context for LLM
    context = io.StringIO()
    context.write(f"Rows={len(df)}, Cols={df.shape[1]}\n")
    if miss is not None:
        context.write("Missingness (top): " + ", ".join([f"{c}:{p:.2%}" for c,p in miss.head(10).items()]) + "\n")
    if num_desc is not None:
        context.write("Numeric cols: " + ", ".join(num_desc.index.tolist()[:15]) + "\n")
    if cat_summ:
        context.write("Categorical cols: " + ", ".join(list(cat_summ.keys())[:10]) + "\n")
    if ts_df is not None:
        context.write(f"Time series points: {len(ts_df)} over {ts_df.index.min()} to {ts_df.index.max()}\n")
    if corr is not None and not corr.empty:
        # strongest pairs
        pairs = []
        cols = corr.columns
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                r = corr.iloc[i, j]
                pairs.append((cols[i], cols[j], r))
        pairs = sorted(pairs, key=lambda x: -abs(x[2]))[:5]
        context.write("Top correlations: " + ", ".join([f"{a}-{b}:{r:.2f}" for a,b,r in pairs]) + "\n")

    llm_ins = llm_insights_if_available(context.getvalue()) or []
    insights = (llm_ins[:10] if llm_ins else []) + rule_insights

    # PDF
    meta = {
        "filename": os.path.basename(csv_path),
        "rows": len(df),
        "cols": df.shape[1],
        "memory": human_mem(df.memory_usage(deep=True).sum()),
    }

    # Convert summary tables to compact forms for PDF
    miss_tbl = miss
    num_tbl = num_desc

    report_path = os.path.join(report_dir, os.path.splitext(os.path.basename(csv_path))[0] + "_report.pdf")
    build_pdf(report_path, meta, miss_tbl, num_tbl, cat_summ, charts, insights)

    return {
        "report_path": report_path,
        "charts": charts,
        "time_col": time_col,
        "insights_count": len(insights),
        "outliers_cols": list(out_iqr.keys())
    }

def main():
    parser = argparse.ArgumentParser(description="Analyst in a Box: Auto-EDA + Insights + PDF")
    parser.add_argument("--csv", required=True, help="Path to CSV file")
    parser.add_argument("--outdir", default="artifacts", help="Directory to save charts/images")
    parser.add_argument("--reports", default="reports", help="Directory to save PDF reports")
    parser.add_argument("--sample", type=int, default=None, help="Optional: read only first N rows for speed")
    parser.add_argument("--timecol", type=str, default=None, help="Optional: specify datetime column for trends")
    parser.add_argument("--agg", type=str, default="mean", choices=["mean","sum","median"], help="Aggregation for time trends")
    args = parser.parse_args()

    result = run_pipeline(args.csv, out_dir=args.outdir, report_dir=args.reports,
                          sample_rows=args.sample, prefer_time_col=args.timecol, agg=args.agg)
    print("✅ Report:", result["report_path"])
    print("🖼️ Charts:", len(result["charts"]))
    print("🕒 Time column:", result["time_col"])
    print("💡 Insights:", result["insights_count"])
    if result["outliers_cols"]:
        print("⚠️ Outlier columns:", ", ".join(result["outliers_cols"]))

if __name__ == "__main__":
    main()
