import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
from datetime import datetime

import gspread
from gspread.exceptions import WorksheetNotFound
from google.oauth2.service_account import Credentials

import networkx as nx
from pyvis.network import Network
import streamlit.components.v1 as components

SHORT_LABEL_MAP = {
    "Sweeping generalisations": "S. generalisations",
    "Sweeping generalizations": "S. generalisations",
    "Seeding doubt/distrust": "Seeding distrust",
    "Discriminatory/Exclusion": "Discriminatory",
    "Call for harm/violence": "Call for harm",
}


# -----------------------------
# Page
# -----------------------------
st.set_page_config(
    page_title="Adversarial Posts Analysis",
    layout="wide"
)

# -----------------------------
# UI polish (no analysis logic changes)
# -----------------------------
def _apply_ui_theme():
    # Streamlit CSS: cards, spacing, typography
    st.markdown(
        """
<style>
        /* Page width + breathing room */
        .block-container { padding-top: 1.2rem; padding-bottom: 2.5rem; }

        /* Card container */
        .pp-card {
            background: rgba(255,255,255,0.75);
            border: 1px solid rgba(49, 51, 63, 0.12);
            border-radius: 16px;
            padding: 14px 14px 6px 14px;
            margin: 0 0 14px 0;
            box-shadow: 0 6px 24px rgba(0,0,0,0.06);
            backdrop-filter: blur(6px);
        }
        .pp-card h3 { margin: 0 0 6px 0; font-size: 1.05rem; }
        .pp-subtle { color: rgba(49, 51, 63, 0.65); font-size: 0.9rem; margin-top: -2px; }

        /* KPI row */
        .pp-kpi-wrap {
            background: rgba(255,255,255,0.75);
            border: 1px solid rgba(49, 51, 63, 0.12);
            border-radius: 16px;
            padding: 10px 12px;
            box-shadow: 0 6px 24px rgba(0,0,0,0.06);
        }

        /* Make widgets feel tighter */
        div[data-testid="stMetric"] { padding: 6px 8px; border-radius: 14px; background: rgba(255,255,255,0.55); border: 1px solid rgba(49, 51, 63, 0.10); }
        div[data-testid="stMetricLabel"] > div { font-size: 0.9rem; color: rgba(49, 51, 63, 0.75); }
        div[data-testid="stMetricValue"] > div { font-size: 1.35rem; }

        /* Tabs: slightly cleaner */
        button[data-baseweb="tab"] { font-size: 0.92rem; }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Altair: a subtle consistent style
    def _altair_theme():
        return {
            "config": {
                "view": {"stroke": "transparent"},
                "axis": {
                    "labelFontSize": 11,
                    "titleFontSize": 11,
                    "gridColor": "rgba(49, 51, 63, 0.10)",
                    "domainColor": "rgba(49, 51, 63, 0.20)",
                    "tickColor": "rgba(49, 51, 63, 0.20)",
                },
                "legend": {"labelFontSize": 11, "titleFontSize": 11},
                "title": {"fontSize": 13, "anchor": "start"},
            }
        }

    alt.themes.register("pp_clean", _altair_theme)
    try:
        alt.themes.enable("pp_clean")
    except Exception:
        pass

    # Plotly: consistent baseline look (keep Plotly where requested)
    try:
        px.defaults.template = "plotly_white"
    except Exception:
        pass


def _card_open(title: str, subtitle: str | None = None):
    st.markdown('<div class="pp-card">', unsafe_allow_html=True)
    st.markdown(f"### {title}")
    if subtitle:
        st.markdown(f'<div class="pp-subtle">{subtitle}</div>', unsafe_allow_html=True)

def _card_close():
    st.markdown("</div>", unsafe_allow_html=True)

_apply_ui_theme()

# -----------------------------
# Helpers
# -----------------------------



def avg_eng_by(df_in: pd.DataFrame, group_col: str, only_adv: bool) -> pd.DataFrame:
    d = df_in.copy()
    if d.empty or group_col not in d.columns:
        return pd.DataFrame(columns=[group_col, "Avg engagement"])

    d["Total Engagement"] = pd.to_numeric(d.get("Total Engagement", 0), errors="coerce").fillna(0)

    if "Adversarial_bool" not in d.columns:
        if "Adversarial" in d.columns:
            d["Adversarial_bool"] = d["Adversarial"].astype(str).str.lower().str.contains("adversarial")
        else:
            d["Adversarial_bool"] = False
    d["Adversarial_bool"] = d["Adversarial_bool"].fillna(False).astype(bool)

    # clean group labels
    d[group_col] = d[group_col].replace("", "Unknown").fillna("Unknown").astype(str).str.strip()

    if only_adv:
        d = d[d["Adversarial_bool"] == True]

    if d.empty:
        return pd.DataFrame(columns=[group_col, "Avg engagement"])

    out = (
        d.groupby(group_col)["Total Engagement"]
         .mean()
         .reset_index(name="Avg engagement")
    )
    out["Avg engagement"] = out["Avg engagement"].round(0).astype(int)
    return out

def compare_avg(df_in: pd.DataFrame, group_col: str) -> pd.DataFrame:
    a = avg_eng_by(df_in, group_col, only_adv=True).rename(columns={"Avg engagement": "Adversarial avg"})
    t = avg_eng_by(df_in, group_col, only_adv=False).rename(columns={"Avg engagement": "Total avg"})

    m = pd.merge(t, a, on=group_col, how="outer").fillna(0)

    # ints
    m["Total avg"] = pd.to_numeric(m["Total avg"], errors="coerce").fillna(0).astype(int)
    m["Adversarial avg"] = pd.to_numeric(m["Adversarial avg"], errors="coerce").fillna(0).astype(int)

    # sort by adversarial avg
    m = m.sort_values("Adversarial avg", ascending=False)
    return m

def _clean_text(x):
    return "" if pd.isna(x) else str(x).strip()

def _as_bool(x):
    if pd.isna(x):
        return False
    s = str(x).strip().lower()
    return s in {"true", "yes", "y", "1", "adversarial"}

def safe_num(x):
    if pd.isna(x):
        return 0.0
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip().replace(",", "")
    try:
        return float(s)
    except:
        return 0.0

def norm_dt(series: pd.Series) -> pd.Series:
    # Normalize to UTC-naive for consistent filtering/grouping
    return pd.to_datetime(series, utc=True, errors="coerce").dt.tz_convert(None)

def top_with_others(df_counts: pd.DataFrame, label_col: str, value_col: str, top_n=8, others_label="Others"):
    if df_counts.empty:
        return df_counts
    df_counts = df_counts.sort_values(value_col, ascending=False).reset_index(drop=True)
    if len(df_counts) <= top_n + 1:
        return df_counts
    top = df_counts.head(top_n).copy()
    others_sum = df_counts[value_col].iloc[top_n:].sum()
    others = pd.DataFrame({label_col: [others_label], value_col: [others_sum]})
    return pd.concat([top, others], ignore_index=True)

def pie_with_table(
    df_counts: pd.DataFrame,
    *,
    name_col: str,
    value_col: str,
    title: str,
    n_value: int,
    show_labels: bool,
    show_table: bool,
    key_prefix: str,
    hole: float = 0.55,
    height: int = 420,
    table_value_name: str = "Posts",
    table_share_name: str = "Share %"
):
    if df_counts.empty:
        st.info("No data.")
        return

    d = df_counts.copy()
    d[name_col] = d[name_col].replace(SHORT_LABEL_MAP)

    total = d[value_col].sum()
    d[table_share_name] = (d[value_col] / total * 100).round(2)

    fig = px.pie(
        d,
        names=name_col,
        values=value_col,
        hole=hole
    )

    # Remove all Plotly titles to avoid "undefined"
    fig.update_layout(
        title=dict(text=""),
        height=height,
        margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(orientation="v")
    )

    fig.update_traces(
        textinfo="percent+label" if show_labels else "none",
        textposition="inside"
    )

    st.caption(f"N = {int(total)}")
    st.plotly_chart(fig, key=f"{key_prefix}_pie", width="stretch")

    if st.checkbox("Show table", key=f"{key_prefix}_table_toggle"):
        table = d[[name_col, value_col, table_share_name]].copy()
        table.columns = [name_col, table_value_name, table_share_name]
        st.dataframe(table, width="stretch")

def bar_with_table(
    df_bar: pd.DataFrame,
    *,
    x: str,
    y: str,
    title: str,
    show_table: bool,
    key_prefix: str,
    height: int = 380
):
    if df_bar.empty:
        st.info("No data.")
        return

    fig = px.bar(df_bar, x=x, y=y, title=title)
    fig.update_layout(height=height, margin=dict(t=60, b=10, l=10, r=10))
    st.plotly_chart(fig, width="stretch", key=f"{key_prefix}_chart")

    if st.checkbox("Show table", key=f"{key_prefix}_table_toggle"):
        st.dataframe(df_bar, width="stretch", key=f"{key_prefix}_table")

def _scope_df(platform_name: str) -> pd.DataFrame:
    if platform_name == "All":
        return pd.concat([fb, yt, tt], ignore_index=True)
    if platform_name == "Facebook":
        return fb
    if platform_name == "YouTube":
        return yt
    return tt

def _adv_only(d: pd.DataFrame) -> pd.DataFrame:
    if d.empty or "Adversarial_bool" not in d.columns:
        return d.iloc[0:0]
    return d[d["Adversarial_bool"] == True].copy()

def _stack_counts(d_adv: pd.DataFrame, dim_col: str, actor_col: str = "Actor Type") -> pd.DataFrame:
    if d_adv.empty:
        return pd.DataFrame(columns=[dim_col, actor_col, "Posts"])
    tmp = d_adv.copy()
    tmp[dim_col] = tmp[dim_col].replace("", "Unknown").fillna("Unknown")
    tmp[actor_col] = tmp[actor_col].replace("", "Unknown").fillna("Unknown")
    out = (
        tmp.groupby([dim_col, actor_col])
           .size()
           .reset_index(name="Posts")
    )
    return out


# -----------------------------
# Auth
# -----------------------------
def get_creds():
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    if "SERVICE_ACCOUNT_FILE" in st.secrets:
        return Credentials.from_service_account_file(st.secrets["SERVICE_ACCOUNT_FILE"], scopes=scopes)
    if "gcp_service_account" in st.secrets:
        return Credentials.from_service_account_info(dict(st.secrets["gcp_service_account"]), scopes=scopes)
    raise RuntimeError("No service account configured. Add SERVICE_ACCOUNT_FILE or [gcp_service_account] to secrets.toml.")

# -----------------------------
# Google Sheets loader (duplicate/blank headers safe + missing sheet safe)
# -----------------------------
@st.cache_data(show_spinner=False, ttl=1800)
def load_worksheet(sheet_id: str, worksheet: str) -> pd.DataFrame:
    creds = get_creds()
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(sheet_id)

    try:
        ws = sh.worksheet(worksheet)
    except WorksheetNotFound:
        st.warning(f"Worksheet not found: '{worksheet}'. Check exact tab name (case + spaces).")
        return pd.DataFrame()

    values = ws.get_all_values()
    if not values or len(values) < 2:
        return pd.DataFrame()

    headers = values[0]
    rows = values[1:]

    headers = [h.strip() if h and str(h).strip() else "Unnamed" for h in headers]

    seen = {}
    out = []
    for h in headers:
        if h not in seen:
            seen[h] = 1
            out.append(h)
        else:
            seen[h] += 1
            out.append(f"{h}__{seen[h]}")

    return pd.DataFrame(rows, columns=out)

# -----------------------------
# Normalizers (header-based)
# -----------------------------
UNIFIED_COLS = [
    "Platform",
    "Url",
    "Posted At",
    "PostedAt_dt",
    "Actor Type",
    "Adversarial",
    "Adversarial_bool",
    "Categories",
    "Narrative",
    "Misinformation/Hate",
    "Severity",
    "Reviewer's Comment",
    "Comment Count",
    "Reaction Count",
    "Shared Count",
    "Total Engagement",
]

def ensure_cols(df: pd.DataFrame, cols: list[str]):
    for c in cols:
        if c not in df.columns:
            df[c] = ""

def normalize_facebook(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw.empty:
        return pd.DataFrame(columns=UNIFIED_COLS)

    df = df_raw.copy()

    actor_cols = [c for c in df.columns if c.startswith("Actor Type")]
    if len(actor_cols) >= 2:
        df["Actor Type"] = df[actor_cols].replace("", np.nan).bfill(axis=1).iloc[:, 0].fillna("")

    ensure_cols(df, [
        "Url", "Posted At",
        "Comment Count", "Reaction Count", "Shared Count", "Total Engagement",
        "Actor Type", "Adversarial", "Categories", "Narrative",
        "Misinformation/Hate", "Severity", "Reviewer's Comment"
    ])

    out = pd.DataFrame()
    out["Platform"] = "Facebook"
    out["Url"] = df["Url"].apply(_clean_text)
    out["Posted At"] = df["Posted At"].apply(_clean_text)
    out["PostedAt_dt"] = norm_dt(out["Posted At"])

    out["Comment Count"] = df["Comment Count"].apply(safe_num)
    out["Reaction Count"] = df["Reaction Count"].apply(safe_num)
    out["Shared Count"] = df["Shared Count"].apply(safe_num)

    te = df["Total Engagement"].apply(safe_num) if "Total Engagement" in df.columns else 0.0
    out["Total Engagement"] = te

    out["Actor Type"] = df["Actor Type"].apply(_clean_text)
    out["Adversarial"] = df["Adversarial"].apply(_clean_text)
    out["Adversarial_bool"] = df["Adversarial"].apply(_as_bool)

    out["Categories"] = df["Categories"].apply(_clean_text)
    out["Narrative"] = df["Narrative"].apply(_clean_text)
    out["Misinformation/Hate"] = df["Misinformation/Hate"].apply(_clean_text)
    out["Severity"] = df["Severity"].apply(_clean_text)
    out["Reviewer's Comment"] = df["Reviewer's Comment"].apply(_clean_text)

    return out[UNIFIED_COLS]

def normalize_youtube(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw.empty:
        return pd.DataFrame(columns=UNIFIED_COLS)

    df = df_raw.copy()

    if "url" in df.columns and "Url" not in df.columns:
        df["Url"] = df["url"]
    if "publishedAt" in df.columns and "Posted At" not in df.columns:
        df["Posted At"] = df["publishedAt"]
    if "likeCount" in df.columns and "Reaction Count" not in df.columns:
        df["Reaction Count"] = df["likeCount"]
    if "commentCount" in df.columns and "Comment Count" not in df.columns:
        df["Comment Count"] = df["commentCount"]
    if "viewCount" in df.columns and "View Count" not in df.columns:
        df["View Count"] = df["viewCount"]

    ensure_cols(df, [
        "Url", "Posted At",
        "View Count", "Reaction Count", "Comment Count",
        "Actor Type", "Adversarial", "Categories", "Narrative",
        "Misinformation/Hate", "Severity", "Reviewer's Comment"
    ])

    out = pd.DataFrame()
    out["Platform"] = "YouTube"
    out["Url"] = df["Url"].apply(_clean_text)
    out["Posted At"] = df["Posted At"].apply(_clean_text)
    out["PostedAt_dt"] = norm_dt(out["Posted At"])

    view = df["View Count"].apply(safe_num)
    out["Reaction Count"] = df["Reaction Count"].apply(safe_num)
    out["Comment Count"] = df["Comment Count"].apply(safe_num)
    out["Shared Count"] = 0.0

    te = df["Total Engagement"].apply(safe_num) if "Total Engagement" in df.columns else view

    # Prefer sheet-provided Total Engagement; fallback to views if missing
    out["Total Engagement"] = te

    out["Actor Type"] = df["Actor Type"].apply(_clean_text)
    out["Adversarial"] = df["Adversarial"].apply(_clean_text)
    out["Adversarial_bool"] = df["Adversarial"].apply(_as_bool)

    out["Categories"] = df["Categories"].apply(_clean_text)
    out["Narrative"] = df["Narrative"].apply(_clean_text)
    out["Misinformation/Hate"] = df["Misinformation/Hate"].apply(_clean_text)
    out["Severity"] = df["Severity"].apply(_clean_text)
    out["Reviewer's Comment"] = df["Reviewer's Comment"].apply(_clean_text)

    return out[UNIFIED_COLS]

def normalize_tiktok(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw.empty:
        return pd.DataFrame(columns=UNIFIED_COLS)

    df = df_raw.copy()

    if "URL" in df.columns and "Url" not in df.columns:
        df["Url"] = df["URL"]
    if "Date" in df.columns and "Posted At" not in df.columns:
        df["Posted At"] = df["Date"]

    if "like_count" in df.columns and "Reaction Count" not in df.columns:
        df["Reaction Count"] = df["like_count"]

    if "comment_count" in df.columns and "Comment Count" not in df.columns:
        df["Comment Count"] = df["comment_count"]
    elif "comment_coun" in df.columns and "Comment Count" not in df.columns:
        df["Comment Count"] = df["comment_coun"]

    if "share_count" in df.columns and "Shared Count" not in df.columns:
        df["Shared Count"] = df["share_count"]

    ensure_cols(df, [
        "Url", "Posted At",
        "Reaction Count", "Comment Count", "Shared Count",
        "Actor Type", "Adversarial", "Categories", "Narrative",
        "Misinformation/Hate", "Severity", "Reviewer's Comment"
    ])

    out = pd.DataFrame()
    out["Platform"] = "TikTok"
    out["Url"] = df["Url"].apply(_clean_text)
    out["Posted At"] = df["Posted At"].apply(_clean_text)
    out["PostedAt_dt"] = norm_dt(out["Posted At"])

    out["Reaction Count"] = df["Reaction Count"].apply(safe_num)
    out["Comment Count"] = df["Comment Count"].apply(safe_num)
    out["Shared Count"] = df["Shared Count"].apply(safe_num)

    te = df["Total Engagement"].apply(safe_num) if "Total Engagement" in df.columns else (
        out["Reaction Count"] + out["Comment Count"] + out["Shared Count"]
    )
    # Prefer sheet-provided Total Engagement; fallback to computed sum if missing
    out["Total Engagement"] = te

    out["Actor Type"] = df["Actor Type"].apply(_clean_text)
    out["Adversarial"] = df["Adversarial"].apply(_clean_text)
    out["Adversarial_bool"] = df["Adversarial"].apply(_as_bool)

    out["Categories"] = df["Categories"].apply(_clean_text)
    out["Narrative"] = df["Narrative"].apply(_clean_text)
    out["Misinformation/Hate"] = df["Misinformation/Hate"].apply(_clean_text)
    out["Severity"] = df["Severity"].apply(_clean_text)
    out["Reviewer's Comment"] = df["Reviewer's Comment"].apply(_clean_text)

    return out[UNIFIED_COLS]

# -----------------------------
# UI: header + sidebar
# -----------------------------
st.title("Adversarial Posts Analysis")

with st.container():
    # -----------------------------
    # Data source + global display options (sidebar removed)
    # -----------------------------
    sheet_id = st.secrets.get("SHEET_ID", "")
    fb_ws, yt_ws, tt_ws = "Facebook", "YouTube", "TikTok"

    # Percentages are ALWAYS computed against the full dataset
    denom_mode = "Full dataset"

    # Chart options are fixed (no sidebar controls)
    SHOW_LABELS = True
    SHOW_TABLES = False


if not sheet_id:
    st.info("Enter Spreadsheet ID or set SHEET_ID in secrets.toml.")
    st.stop()

# -----------------------------
# Load + Normalize
# -----------------------------
fb_raw = load_worksheet(sheet_id, fb_ws)
yt_raw = load_worksheet(sheet_id, yt_ws)
tt_raw = load_worksheet(sheet_id, tt_ws)

fb = normalize_facebook(fb_raw)
yt = normalize_youtube(yt_raw)
tt = normalize_tiktok(tt_raw)

frames = [x for x in [fb, yt, tt] if not x.empty]
if not frames:
    st.warning("No data found in Facebook/YouTube/TikTok sheets.")
    st.stop()

df = pd.concat(frames, ignore_index=True)
df["Adversarial_bool"] = df["Adversarial_bool"].fillna(False)

df_f = df.copy()

# Denominator for percentages
df_denom = df if denom_mode == "Full dataset" else df_f
df_base = df_denom.copy()
df_base["Adversarial_bool"] = df_base["Adversarial_bool"].fillna(False)
adv_df = df_base[df_base["Adversarial_bool"] == True]

# -----------------------------
# KPIs
# -----------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total posts", f"{len(df_f):,}")
k2.metric("Adversarial posts", f"{int(df_f['Adversarial_bool'].sum()):,}")

avg_all = (df_f["Total Engagement"].mean() if len(df_f) else 0)
adv_only = df_f[df_f["Adversarial_bool"] == True]
avg_adv = (adv_only["Total Engagement"].mean() if len(adv_only) else 0)

k3.metric("Avg engagement per post", f"{avg_all:.0f}")
k4.metric("Avg engagement per adversarial post", f"{avg_adv:.0f}")
st.markdown('</div>', unsafe_allow_html=True)

st.divider()

# =========================
# ROW 1 — Overview (sheet-tab accurate)
# Left pie: counts from each sheet tab (Facebook / YouTube / TikTok)
# Right pie: Adversarial vs Non-Adversarial with tab filter (All/Facebook/YouTube/TikTok)
# =========================

# Safety: ensure bool exists
for _d in (fb, yt, tt):
    if not _d.empty and "Adversarial_bool" in _d.columns:
        _d["Adversarial_bool"] = _d["Adversarial_bool"].fillna(False).astype(bool)

# LEFT pie source: counts per sheet tab (not from unified df)
fb_n = 0 if fb.empty else len(fb)
yt_n = 0 if yt.empty else len(yt)
tt_n = 0 if tt.empty else len(tt)

plat_counts_global = pd.DataFrame(
    {"Platform": ["Facebook", "YouTube", "TikTok"], "Posts": [fb_n, yt_n, tt_n]}
)

left, right = st.columns(2)

with left:
    _card_open("Posts by Platform")

    lt_all, lt_adv = st.tabs(["All", "Adversarial"])

    # --- All tab (default)
    with lt_all:
        fb_n = 0 if fb.empty else len(fb)
        yt_n = 0 if yt.empty else len(yt)
        tt_n = 0 if tt.empty else len(tt)

        plat_counts_all = pd.DataFrame(
            {"Platform": ["Facebook", "YouTube", "TikTok"], "Posts": [fb_n, yt_n, tt_n]}
        )

        pie_with_table(
            plat_counts_all,
            name_col="Platform",
            value_col="Posts",
            title="Posts by Platforms",
            n_value=int(plat_counts_all["Posts"].sum()),
            show_labels=SHOW_LABELS,
            show_table=SHOW_TABLES,
            key_prefix="row1_left_all",
            hole=0.0,
            height=420,
            table_value_name="Posts",
        )

    # --- Adversarial tab
    with lt_adv:
        fb_a = 0 if fb.empty else int(fb["Adversarial_bool"].sum())
        yt_a = 0 if yt.empty else int(yt["Adversarial_bool"].sum())
        tt_a = 0 if tt.empty else int(tt["Adversarial_bool"].sum())

        plat_counts_adv = pd.DataFrame(
            {"Platform": ["Facebook", "YouTube", "TikTok"], "Posts": [fb_a, yt_a, tt_a]}
        )

        pie_with_table(
            plat_counts_adv,
            name_col="Platform",
            value_col="Posts",
            title="Adversarial Posts by Platforms",
            n_value=int(plat_counts_adv["Posts"].sum()),
            show_labels=SHOW_LABELS,
            show_table=SHOW_TABLES,
            key_prefix="row1_left_adv",
            hole=0.0,
            height=420,
            table_value_name="Posts",
        )
    _card_close()

with right:
    _card_open("Adversarial posts by Platform")

    t_all, t_fb, t_yt, t_tt = st.tabs(["All", "Facebook", "YouTube", "TikTok"])
    tab_defs = [
        ("All", t_all),
        ("Facebook", t_fb),
        ("YouTube", t_yt),
        ("TikTok", t_tt),
    ]

    for name, tab in tab_defs:
        with tab:
            if name == "All":
                df_scope = pd.concat([fb, yt, tt], ignore_index=True)
            elif name == "Facebook":
                df_scope = fb
            elif name == "YouTube":
                df_scope = yt
            else:
                df_scope = tt

            if df_scope.empty:
                st.info("No data.")
                continue

            adv_n = int(df_scope["Adversarial_bool"].sum())
            non_adv_n = int(len(df_scope) - adv_n)

            pt_df = pd.DataFrame(
                {"Post Type": ["Non Adversarial", "Adversarial"], "Posts": [non_adv_n, adv_n]}
            )

            pie_with_table(
                pt_df,
                name_col="Post Type",
                value_col="Posts",
                title="Post Type",
                n_value=int(len(df_scope)),
                show_labels=SHOW_LABELS,
                show_table=SHOW_TABLES,
                key_prefix=f"row1_right_{name.lower()}",
                hole=0.0,
                height=420,
                table_value_name="Posts",
            )

    _card_close()

st.divider()

# =========================
# ROW 2 — One row, two pies, separate tab filters
# Left tabs:  All/Facebook/YouTube/TikTok  -> Categories distribution (adversarial only)
# Right tabs: All/Facebook/YouTube/TikTok  -> Narrative distribution (adversarial only)
# =========================

row2_left, row2_right = st.columns(2)
r3c1, r3c2, r3c3 = st.columns(3)

def _counts(d: pd.DataFrame, col: str, top_n: int = 10) -> pd.DataFrame:
    if d.empty or col not in d.columns:
        return pd.DataFrame(columns=[col, "Posts"])
    s = d[col].replace("", "Unknown").fillna("Unknown").astype(str).str.strip()
    out = s.value_counts().reset_index()
    out.columns = [col, "Posts"]
    out = top_with_others(out, col, "Posts", top_n=top_n, others_label="Others")
    return out

def _narr_counts_for_category(d_adv: pd.DataFrame, category_name: str, top_n: int = 10) -> (pd.DataFrame, int):
    if d_adv.empty:
        return pd.DataFrame(columns=["Narrative", "Posts"]), 0

    if "Categories" not in d_adv.columns or "Narrative" not in d_adv.columns:
        return pd.DataFrame(columns=["Narrative", "Posts"]), 0

    cat = d_adv["Categories"].fillna("").astype(str).str.strip()
    sub = d_adv[cat == category_name].copy()

    if sub.empty:
        return pd.DataFrame(columns=["Narrative", "Posts"]), 0

    s = sub["Narrative"].replace("", "Unknown").fillna("Unknown").astype(str).str.strip()
    out = s.value_counts().reset_index()
    out.columns = ["Narrative", "Posts"]
    out = top_with_others(out, "Narrative", "Posts", top_n=top_n, others_label="Others")
    return out, int(len(sub))

def _render_category_pie(container, category_name: str, title: str, key_base: str, top_n: int = 10):
    with container:
        st.markdown(f"##### {title}")
        t_all, t_fb, t_yt, t_tt = st.tabs(["All", "Facebook", "YouTube", "TikTok"])
        tab_defs = [("All", t_all), ("Facebook", t_fb), ("YouTube", t_yt), ("TikTok", t_tt)]

        for platform_name, tab in tab_defs:
            with tab:
                scope = _scope_df(platform_name)
                adv_scope = _adv_only(scope)

                narr_counts, n_cat = _narr_counts_for_category(adv_scope, category_name, top_n=top_n)

                pie_with_table(
                    narr_counts,
                    name_col="Narrative",
                    value_col="Posts",
                    title=title,
                    n_value=n_cat,
                    show_labels=SHOW_LABELS,
                    show_table=SHOW_TABLES,
                    key_prefix=f"{key_base}_{platform_name.lower()}",
                    hole=0.55,      # donut like your screenshot
                    height=380,
                    table_value_name="Posts",
                )

_render_category_pie(
    r3c1,
    category_name="In and Out-group framing",
    title="Narratives of In and Out-group framing",
    key_base="row3_inout",
    top_n=10
)

_render_category_pie(
    r3c2,
    category_name="Negative targeting",
    title="Narratives of Negative targeting",
    key_base="row3_negative",
    top_n=10
)

_render_category_pie(
    r3c3,
    category_name="Sensational",
    title="Narratives of Sensationalized post",
    key_base="row3_sensational",
    top_n=10
)

st.divider()

# =========================
# ROW 4 — Stacked bars by actor  (Neutral dropped in this section only)
# =========================

r4c1, r4c2 = st.columns(2)

def _render_stacked(container, *, title: str, dim_col: str, key_base: str):
    with container:
        _card_open(title, "100% stacked distribution of adversarial posts across Actor Types (neutral excluded).")

        # Make tab keys stable across reruns
        t_all, t_fb, t_yt, t_tt = st.tabs(
            ["All", "Facebook", "YouTube", "TikTok"]
        )

        for platform_name, tab in [("All", t_all), ("Facebook", t_fb), ("YouTube", t_yt), ("TikTok", t_tt)]:
            with tab:
                scope = _scope_df(platform_name)
                adv_scope = _adv_only(scope)

                sc = _stack_counts(adv_scope, dim_col=dim_col)
                n_val = int(len(adv_scope))

                if sc.empty:
                    st.info("No data.")
                    continue

                # ✅ SPECIAL RULE (this section only): drop Neutral theme rows
                # (display-only; does not change upstream logic, just hides Neutral from the chart)
                if dim_col in sc.columns:
                    _s = sc[dim_col].astype(str).str.strip().str.lower()
                    sc = sc[~_s.eq("neutral")].copy()

                if sc.empty:
                    st.info("No data (after excluding Neutral).")
                    continue

                # Convert to 100% stacked (same as before)
                totals = sc.groupby(dim_col)["Posts"].transform("sum")
                sc["Pct"] = (sc["Posts"] / totals * 100).round(1)

                # Altair stacked bars (analysis logic unchanged, only rendering)
                base = alt.Chart(sc).encode(
                    y=alt.Y(f"{dim_col}:N", sort="-x", title=None),
                    x=alt.X("Pct:Q", title=None, scale=alt.Scale(domain=[0, 100])),
                    color=alt.Color("Actor Type:N", title="Actor Type"),
                    tooltip=[
                        alt.Tooltip(f"{dim_col}:N", title=dim_col),
                        alt.Tooltip("Actor Type:N"),
                        alt.Tooltip("Posts:Q", format=",.0f"),
                        alt.Tooltip("Pct:Q", format=".1f", title="% of theme"),
                    ],
                )

                bars = base.mark_bar()

                # Label only meaningful segments to reduce clutter
                labels = base.transform_filter("datum.Pct >= 7").mark_text(dx=8, color="white").encode(
                    text=alt.Text("Pct:Q", format=".1f")
                )

                chart = (bars + labels).properties(height=420).configure_legend(orient="top")

                st.caption(f"N = {n_val}")
                st.altair_chart(chart, width="stretch")

        _card_close()

# Left: Categories by actors
_render_stacked(
    r4c1,
    title="Adversarial post Categories by actors",
    dim_col="Categories",
    key_base="row4_cat_actor"
)

# Right: Narratives by actors
_render_stacked(
    r4c2,
    title="Adversarial Post Narratives by actors",
    dim_col="Narrative",
    key_base="row4_nar_actor"
)

st.divider()



# =========================
# CHART 1 — Actor totals + % adversarial (by platform tabs)
# =========================

# =========================
# CHART 1 — Actor totals + % adversarial (by platform tabs)
# =========================

import plotly.graph_objects as go
import altair as alt

def actor_summary(df_scope: pd.DataFrame) -> pd.DataFrame:
    if df_scope.empty:
        return pd.DataFrame(columns=["Actor Type", "TotalPosts", "AdversarialPosts", "AdversarialPct"])

    d = df_scope.copy()
    d["Actor Type"] = d["Actor Type"].replace("", "Unknown").fillna("Unknown").astype(str).str.strip()
    d["Adversarial_bool"] = d["Adversarial_bool"].fillna(False).astype(bool)

    g = d.groupby("Actor Type", as_index=False).agg(
        TotalPosts=("Adversarial_bool", "size"),
        AdversarialPosts=("Adversarial_bool", "sum"),
    )
    g["AdversarialPct"] = (g["AdversarialPosts"] / g["TotalPosts"] * 100).round(1)
    g = g.sort_values("TotalPosts", ascending=False)
    return g

def _scope_df(platform_name: str) -> pd.DataFrame:
    if platform_name == "All":
        return pd.concat([fb, yt, tt], ignore_index=True)
    if platform_name == "Facebook":
        return fb
    if platform_name == "YouTube":
        return yt
    return tt

st.markdown("### Actor activity and adversarial rate")
st.write("Bar: Total posts by actor. Line/markers: % of adversarial posts within each actor.")

t_all, t_fb, t_yt, t_tt = st.tabs(["All", "Facebook", "YouTube", "TikTok"])
for platform_name, tab in [("All", t_all), ("Facebook", t_fb), ("YouTube", t_yt), ("TikTok", t_tt)]:
    with tab:
        scope = _scope_df(platform_name)
        summ = actor_summary(scope)

        if summ.empty:
            st.info("No data.")
            continue

        # Primary: Total posts (bar). Secondary: % adversarial (line/markers).
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                y=summ["Actor Type"],
                x=summ["TotalPosts"],
                name="Total posts",
                orientation="h",
                text=summ["TotalPosts"],
                textposition="auto",
            )
        )

        fig.add_trace(
            go.Scatter(
                y=summ["Actor Type"],
                x=summ["AdversarialPct"],
                name="% adversarial",
                mode="markers+lines",
                xaxis="x2",
                text=summ["AdversarialPct"].astype(str) + "%",
                hovertemplate="Actor=%{y}<br>% adversarial=%{x}%<extra></extra>",
            )
        )

        fig.update_layout(
            height=420,
            margin=dict(t=10, b=10, l=10, r=10),
            barmode="overlay",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(title="Total posts"),
            xaxis2=dict(
                title="% adversarial",
                overlaying="x",
                side="top",
                range=[0, 100],
            ),
            yaxis=dict(title=None),
        )

        st.caption(f"N = {int(len(scope))}")
        st.plotly_chart(fig, key=f"chart1_actor_{platform_name.lower()}", width="stretch")

        if st.checkbox("Show table", key=f"chart1_actor_{platform_name.lower()}_table_toggle"):
            st.dataframe(summ, width="stretch")

st.divider()


# =========================
# Avg engagement comparison (ALTair grouped bars, side-by-side)
# =========================

def actor_summary(df_scope: pd.DataFrame) -> pd.DataFrame:
    if df_scope.empty:
        return pd.DataFrame(columns=["Actor Type", "TotalPosts", "AdversarialPosts", "AdversarialPct"])

    d = df_scope.copy()
    d["Actor Type"] = d["Actor Type"].replace("", "Unknown").fillna("Unknown").astype(str).str.strip()
    d["Adversarial_bool"] = d["Adversarial_bool"].fillna(False).astype(bool)

    g = d.groupby("Actor Type", as_index=False).agg(
        TotalPosts=("Adversarial_bool", "size"),
        AdversarialPosts=("Adversarial_bool", "sum"),
    )
    g["AdversarialPct"] = (g["AdversarialPosts"] / g["TotalPosts"] * 100).round(1)
    g = g.sort_values("TotalPosts", ascending=False)
    return g

def adv_cat_by_actor(df_scope: pd.DataFrame) -> (pd.DataFrame, int):
    if df_scope.empty:
        return pd.DataFrame(columns=["Actor Type", "Categories", "Posts", "Pct"]), 0

    d = df_scope.copy()
    d["Actor Type"] = d["Actor Type"].replace("", "Unknown").fillna("Unknown").astype(str).str.strip()
    d["Categories"] = d["Categories"].replace("", "Unknown").fillna("Unknown").astype(str).str.strip()
    d["Adversarial_bool"] = d["Adversarial_bool"].fillna(False).astype(bool)

    adv = d[d["Adversarial_bool"] == True].copy()
    n_adv = int(len(adv))
    if adv.empty:
        return pd.DataFrame(columns=["Actor Type", "Categories", "Posts", "Pct"]), 0

    g = adv.groupby(["Actor Type", "Categories"], as_index=False).size()
    g = g.rename(columns={"size": "Posts"})

    totals = g.groupby("Actor Type")["Posts"].transform("sum")
    g["Pct"] = (g["Posts"] / totals * 100).round(1)

    return g, n_adv

def adv_narr_by_actor(df_scope: pd.DataFrame) -> (pd.DataFrame, int):
    if df_scope.empty:
        return pd.DataFrame(columns=["Actor Type", "Narrative", "Posts", "Pct"]), 0

    d = df_scope.copy()
    d["Actor Type"] = d["Actor Type"].replace("", "Unknown").fillna("Unknown").astype(str).str.strip()
    d["Narrative"] = d["Narrative"].replace("", "Unknown").fillna("Unknown").astype(str).str.strip()
    d["Adversarial_bool"] = d["Adversarial_bool"].fillna(False).astype(bool)

    adv = d[d["Adversarial_bool"] == True].copy()
    n_adv = int(len(adv))
    if adv.empty:
        return pd.DataFrame(columns=["Actor Type", "Narrative", "Posts", "Pct"]), 0

    g = adv.groupby(["Actor Type", "Narrative"], as_index=False).size()
    g = g.rename(columns={"size": "Posts"})

    totals = g.groupby("Actor Type")["Posts"].transform("sum")
    g["Pct"] = (g["Posts"] / totals * 100).round(1)

    return g, n_adv

def overall_avg_compare(df_in: pd.DataFrame) -> dict:
    if df_in is None or df_in.empty:
        return {"total_avg": 0, "adv_avg": 0, "lift": 0.0, "n_total": 0, "n_adv": 0}

    d = df_in.copy()
    d["Total Engagement"] = pd.to_numeric(d.get("Total Engagement", 0), errors="coerce").fillna(0)

    if "Adversarial_bool" not in d.columns:
        if "Adversarial" in d.columns:
            d["Adversarial_bool"] = d["Adversarial"].astype(str).str.lower().str.contains("adversarial")
        else:
            d["Adversarial_bool"] = False
    d["Adversarial_bool"] = d["Adversarial_bool"].fillna(False).astype(bool)

    total_avg = float(d["Total Engagement"].mean()) if len(d) else 0.0
    adv = d[d["Adversarial_bool"] == True]
    adv_avg = float(adv["Total Engagement"].mean()) if len(adv) else 0.0
    lift = (adv_avg / total_avg) if total_avg > 0 else 0.0

    return {
        "total_avg": round(total_avg, 2),
        "adv_avg": round(adv_avg, 2),
        "lift": round(lift, 3),
        "n_total": int(len(d)),
        "n_adv": int(len(adv)),
    }

def compare_adv_only(df_in: pd.DataFrame, group_col: str) -> pd.DataFrame:
    a = avg_eng_by(df_in, group_col, only_adv=True).rename(columns={"Avg engagement": "Adversarial avg"})
    a["Adversarial avg"] = pd.to_numeric(a["Adversarial avg"], errors="coerce").fillna(0).astype(int)
    return a.sort_values("Adversarial avg", ascending=False)

# --- helper: grouped bar chart (side-by-side) ---
def _grouped_compare_chart(plot_df: pd.DataFrame, dim_col: str, height: int = 420):
    """
    plot_df columns:
      - dim_col (e.g. "Categories" or "Narrative")
      - "Series" (Baseline..., Adversarial avg)
      - "Avg engagement" (numeric)
    """
    if plot_df is None or plot_df.empty:
        return None

    # Make sort stable & meaningful: sort by Adversarial avg first (if present), else by max value
    sort_df = plot_df.copy()
    if "Adversarial avg" in set(sort_df["Series"].astype(str).unique()):
        tmp = (
            sort_df[sort_df["Series"] == "Adversarial avg"]
            [[dim_col, "Avg engagement"]]
            .rename(columns={"Avg engagement": "_sort"})
        )
        sort_order = tmp.sort_values("_sort", ascending=False)[dim_col].tolist()
    else:
        tmp = sort_df.groupby(dim_col, as_index=False)["Avg engagement"].max().rename(columns={"Avg engagement": "_sort"})
        sort_order = tmp.sort_values("_sort", ascending=False)[dim_col].tolist()

    chart = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X(f"{dim_col}:N", sort=sort_order, title=None, axis=alt.Axis(labelAngle=90)),
            xOffset=alt.XOffset("Series:N"),
            y=alt.Y("Avg engagement:Q", title="Avg engagement"),
            color=alt.Color("Series:N", title=None, legend=alt.Legend(orient="top")),
            tooltip=[
                alt.Tooltip(f"{dim_col}:N", title=dim_col),
                alt.Tooltip("Series:N"),
                alt.Tooltip("Avg engagement:Q", format=",.0f"),
            ],
        )
        .properties(height=height)
    )

    labels = (
        alt.Chart(plot_df)
        .mark_text(dy=-8)
        .encode(
            x=alt.X(f"{dim_col}:N", sort=sort_order),
            xOffset=alt.XOffset("Series:N"),
            y=alt.Y("Avg engagement:Q"),
            text=alt.Text("Avg engagement:Q", format=",.0f"),
            detail="Series:N",
        )
    )

    return (chart + labels).configure_view(strokeOpacity=0).configure_axis(grid=True)

# =========================
# UI
# =========================
t_all, t_fb, t_yt, t_tt = st.tabs(["All", "Facebook", "YouTube", "TikTok"])

for platform_name, tab in [("All", t_all), ("Facebook", t_fb), ("YouTube", t_yt), ("TikTok", t_tt)]:
    with tab:
        df_tab = _scope_df(platform_name)

        # KPI compare (overall)
        k = overall_avg_compare(df_tab)
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total avg engagement", k["total_avg"])
        k2.metric("Adversarial avg engagement", k["adv_avg"])
        k3.metric("Lift (adv/total)", k["lift"])
        k4.metric("Total N", k["n_total"])
        k5.metric("Adversarial N", k["n_adv"])

        show_both = st.toggle(
            "Compare with total posts average",
            value=True,
            key=f"cmp_total_{platform_name}"
        )

        cat_adv = compare_adv_only(df_tab, "Categories")
        nar_adv = compare_adv_only(df_tab, "Narrative")

        # adversarial avg by group (what you actually want to compare against a constant baseline)
        cat_adv = compare_adv_only(df_tab, "Categories")
        nar_adv = compare_adv_only(df_tab, "Narrative")

        # ✅ SPECIAL RULE (this section only): hide Neutral
        def _drop_neutral(df_in: pd.DataFrame, col: str) -> pd.DataFrame:
            if df_in is None or df_in.empty or col not in df_in.columns:
                return df_in
            s = df_in[col].astype(str).str.strip().str.lower()
            return df_in[~s.eq("neutral")].copy()

        cat_adv = _drop_neutral(cat_adv, "Categories")
        nar_adv = _drop_neutral(nar_adv, "Narrative")


        c1, c2 = st.columns([1, 2])

        with c1:
            st.subheader("Average engagement of Categories")
            if cat_adv.empty:
                st.info("No data.")
            else:
                plot_w = cat_adv.copy()
                if show_both:
                    plot_w["Baseline (overall total avg)"] = float(k["total_avg"])
                    use_cols = ["Baseline (overall total avg)", "Adversarial avg"]
                else:
                    use_cols = ["Adversarial avg"]

                plot_df = plot_w.melt(
                    id_vars=["Categories"],
                    value_vars=use_cols,
                    var_name="Series",
                    value_name="Avg engagement",
                )

                fig = _grouped_compare_chart(plot_df, dim_col="Categories", height=420)
                st.altair_chart(fig, width="stretch")

                if st.checkbox("Show numbers table", key=f"tbl_cmp_cat_{platform_name}"):
                    out = plot_w.copy()
                    if show_both and "Baseline (overall total avg)" in out.columns:
                        out = out[["Categories", "Baseline (overall total avg)", "Adversarial avg"]]
                    st.dataframe(out, width="stretch")

        with c2:
            st.subheader("Average engagement of Narratives")
            if nar_adv.empty:
                st.info("No data.")
            else:
                plot_w = nar_adv.copy()
                if show_both:
                    plot_w["Baseline (overall total avg)"] = float(k["total_avg"])
                    use_cols = ["Baseline (overall total avg)", "Adversarial avg"]
                else:
                    use_cols = ["Adversarial avg"]

                plot_df = plot_w.melt(
                    id_vars=["Narrative"],
                    value_vars=use_cols,
                    var_name="Series",
                    value_name="Avg engagement",
                )

                fig = _grouped_compare_chart(plot_df, dim_col="Narrative", height=420)
                st.altair_chart(fig, width="stretch")

                if st.checkbox("Show numbers table", key=f"tbl_cmp_nar_{platform_name}"):
                    out = plot_w.copy()
                    if show_both and "Baseline (overall total avg)" in out.columns:
                        out = out[["Narrative", "Baseline (overall total avg)", "Adversarial avg"]]
                    st.dataframe(out, width="stretch")

st.divider()


# =========================
# ROW 6 — (Actor Intersection Analysis)  ✅ SAME LOGIC
# ✅ Only changes:
#   1) Actor+Category and Actor+Narrative are side-by-side in one row (33% / 67%)
#   2) Drop Neutral from BOTH charts (display-only, this section only)
# =========================

st.markdown("## Actor Intersection Analysis")
st.write("This section shows how different actors perform when they publish adversarial content, broken down by category and narrative. The bars represent the average engagement per adversarial post for the actors, allowing comparison of which actor–category or actor–narrative combinations attract more attention relative to each actor’s usual engagement level.")

def actor_narrative_profile(df_in: pd.DataFrame) -> pd.DataFrame:
    if df_in is None or df_in.empty:
        return pd.DataFrame()

    d = df_in.copy()
    d["Total Engagement"] = pd.to_numeric(d.get("Total Engagement", 0), errors="coerce").fillna(0)

    if "Adversarial_bool" not in d.columns:
        d["Adversarial_bool"] = d.get("Adversarial", "").astype(str).str.lower().str.contains("adversarial")
    d["Adversarial_bool"] = d["Adversarial_bool"].fillna(False).astype(bool)

    if "Actor Type" not in d.columns or "Narrative" not in d.columns:
        return pd.DataFrame()

    d["Actor Type"] = d["Actor Type"].fillna("Unknown").astype(str).str.strip()
    d["Narrative"] = d["Narrative"].fillna("Unknown").astype(str).str.strip()

    # Actor-level totals (ALL posts)
    actor_totals = (
        d.groupby("Actor Type")
         .agg(
            **{
                "Total posts": ("Actor Type", "size"),
                "Overall avg engagement": ("Total Engagement", "mean"),
            }
         )
         .reset_index()
    )

    # Actor + Narrative (adversarial only)
    adv = d[d["Adversarial_bool"] == True].copy()
    adv_slice = (
        adv.groupby(["Actor Type", "Narrative"])
           .agg(
                **{
                    "Adversarial posts": ("Narrative", "size"),
                    "Adversarial avg engagement": ("Total Engagement", "mean"),
                }
           )
           .reset_index()
    )

    out = adv_slice.merge(actor_totals, on="Actor Type", how="left")

    out["Overall avg engagement"] = out["Overall avg engagement"].fillna(0)
    out["Adversarial avg engagement"] = out["Adversarial avg engagement"].fillna(0)

    out["Lift (adv/overall)"] = out.apply(
        lambda r: (r["Adversarial avg engagement"] / r["Overall avg engagement"]) if r["Overall avg engagement"] > 0 else 0,
        axis=1
    )

    out["Overall avg engagement"] = out["Overall avg engagement"].round(1)
    out["Adversarial avg engagement"] = out["Adversarial avg engagement"].round(1)
    out["Lift (adv/overall)"] = out["Lift (adv/overall)"].round(2)

    out = out.sort_values(["Actor Type", "Adversarial avg engagement"], ascending=[True, False]).reset_index(drop=True)
    return out


def actor_category_profile(df_in: pd.DataFrame) -> pd.DataFrame:
    if df_in is None or df_in.empty:
        return pd.DataFrame()

    d = df_in.copy()
    d["Total Engagement"] = pd.to_numeric(d.get("Total Engagement", 0), errors="coerce").fillna(0)

    if "Adversarial_bool" not in d.columns:
        d["Adversarial_bool"] = d.get("Adversarial", "").astype(str).str.lower().str.contains("adversarial")
    d["Adversarial_bool"] = d["Adversarial_bool"].fillna(False).astype(bool)

    if "Actor Type" not in d.columns or "Categories" not in d.columns:
        return pd.DataFrame()

    d["Actor Type"] = d["Actor Type"].fillna("Unknown").astype(str).str.strip()
    d["Categories"] = d["Categories"].fillna("Unknown").astype(str).str.strip()

    # Actor-level totals (ALL posts)
    actor_totals = (
        d.groupby("Actor Type")
         .agg(
            **{
                "Total posts": ("Actor Type", "size"),
                "Overall avg engagement": ("Total Engagement", "mean"),
            }
         )
         .reset_index()
    )

    # Actor + Category (adversarial only)
    adv = d[d["Adversarial_bool"] == True].copy()
    adv_slice = (
        adv.groupby(["Actor Type", "Categories"])
           .agg(
                **{
                    "Adversarial posts": ("Categories", "size"),
                    "Adversarial avg engagement": ("Total Engagement", "mean"),
                }
           )
           .reset_index()
           .rename(columns={"Categories": "Category"})
    )

    out = adv_slice.merge(actor_totals, on="Actor Type", how="left")

    out["Overall avg engagement"] = out["Overall avg engagement"].fillna(0)
    out["Adversarial avg engagement"] = out["Adversarial avg engagement"].fillna(0)

    out["Lift (adv/overall)"] = out.apply(
        lambda r: (r["Adversarial avg engagement"] / r["Overall avg engagement"]) if r["Overall avg engagement"] > 0 else 0,
        axis=1
    )

    out["Overall avg engagement"] = out["Overall avg engagement"].round(1)
    out["Adversarial avg engagement"] = out["Adversarial avg engagement"].round(1)
    out["Lift (adv/overall)"] = out["Lift (adv/overall)"].round(2)

    out = out.sort_values(["Actor Type", "Adversarial avg engagement"], ascending=[True, False]).reset_index(drop=True)
    return out


# Display-only neutral drop (this section only)
def _drop_neutral_display(df_in: pd.DataFrame, col: str) -> pd.DataFrame:
    if df_in is None or df_in.empty or col not in df_in.columns:
        return df_in
    s = df_in[col].astype(str).str.strip().str.lower()
    return df_in[~s.eq("neutral")].copy()


t_all, t_fb, t_yt, t_tt = st.tabs(["All", "Facebook", "YouTube", "TikTok"])

for platform_name, tab in [
    ("All", t_all),
    ("Facebook", t_fb),
    ("YouTube", t_yt),
    ("TikTok", t_tt),
]:
    with tab:
        df_tab = _scope_df(platform_name)

        # Compute both tables once (same as your original)
        nar_df = actor_narrative_profile(df_tab)
        cat_df = actor_category_profile(df_tab)

        # Drop Neutral (display-only)
        nar_df = _drop_neutral_display(nar_df, "Narrative")
        cat_df = _drop_neutral_display(cat_df, "Category")

        # ✅ One row, weighted widths: Category left (33%), Narrative right (67%)
        c_left, c_right = st.columns([1, 2])

        # -------------------------
        # Actor + Category (LEFT)
        # -------------------------
        with c_left:
            st.subheader("Actor + Category")

            if cat_df.empty:
                st.info("No category intersection data.")
            else:
                selected_actor_cat = st.selectbox(
                    "Select Actor Type",
                    options=["All"] + sorted(cat_df["Actor Type"].unique().tolist()),
                    key=f"actor_cat_{platform_name}",
                )

                view_df_cat = cat_df.copy()
                if selected_actor_cat != "All":
                    view_df_cat = view_df_cat[view_df_cat["Actor Type"] == selected_actor_cat]

                chart2 = alt.Chart(view_df_cat).mark_bar().encode(
                    x=alt.X("Category:N", sort="-y", title=None, axis=alt.Axis(labelAngle=90)),
                    y=alt.Y("Adversarial avg engagement:Q", title="Adversarial avg engagement"),
                    color=alt.Color("Actor Type:N", title="Actor Type"),
                    tooltip=[
                        alt.Tooltip("Actor Type:N"),
                        alt.Tooltip("Category:N"),
                        alt.Tooltip("Adversarial posts:Q", format=",.0f"),
                        alt.Tooltip("Adversarial avg engagement:Q", format=",.1f"),
                        alt.Tooltip("Overall avg engagement:Q", format=",.1f"),
                        alt.Tooltip("Lift (adv/overall):Q", format=".2f"),
                    ],
                ).properties(height=420)

                st.altair_chart(chart2.configure_legend(orient="top"), width="stretch")

                if st.checkbox("Show numbers table", key=f"tbl_actor_cat_{platform_name}"):
                    st.dataframe(view_df_cat, width="stretch")

        # -------------------------
        # Actor + Narrative (RIGHT)
        # -------------------------
        with c_right:
            st.subheader("Actor + Narrative")

            if nar_df.empty:
                st.info("No narrative intersection data.")
            else:
                selected_actor = st.selectbox(
                    "Select Actor Type",
                    options=["All"] + sorted(nar_df["Actor Type"].unique().tolist()),
                    key=f"actor_nar_{platform_name}",
                )

                view_df = nar_df.copy()
                if selected_actor != "All":
                    view_df = view_df[view_df["Actor Type"] == selected_actor]

                chart = alt.Chart(view_df).mark_bar().encode(
                    x=alt.X("Narrative:N", sort="-y", title=None, axis=alt.Axis(labelAngle=90)),
                    y=alt.Y("Adversarial avg engagement:Q", title="Adversarial avg engagement"),
                    color=alt.Color("Actor Type:N", title="Actor Type"),
                    tooltip=[
                        alt.Tooltip("Actor Type:N"),
                        alt.Tooltip("Narrative:N"),
                        alt.Tooltip("Adversarial posts:Q", format=",.0f"),
                        alt.Tooltip("Adversarial avg engagement:Q", format=",.1f"),
                        alt.Tooltip("Overall avg engagement:Q", format=",.1f"),
                        alt.Tooltip("Lift (adv/overall):Q", format=".2f"),
                    ],
                ).properties(height=520)

                st.altair_chart(chart.configure_legend(orient="top"), width="stretch")

                if st.checkbox("Show numbers table", key=f"tbl_actor_nar_{platform_name}"):
                    st.dataframe(view_df, width="stretch")


st.divider()

# =========================
# ROW 7 — Actor Lift Heatmap (Actor vs Adversarial Narrative/Category)
# Drop this block AFTER you already have: fb, yt, tt dataframes
# and AFTER you already have: _scope_df(platform_name) helper
# Requires columns:
#   - "Actor Type"
#   - "Narrative" and/or "Categories"
#   - "Total Engagement" (numeric or convertible)
#   - either "Adversarial_bool" OR "Adversarial" (text: Adversarial/Non Adversarial)
# =========================

st.markdown("## Actor lift vs adversarial themes (who overperforms where)")
st.write("This heatmap shows where different actors overperform or underperform across adversarial narratives or categories. Each cell represents an actor–theme pair, with values indicating how much engagement that theme receives relative to a baseline (actor-level or theme-level). A lift above 1 means the actor received more engagement on that theme than their average.")

def _ensure_adv_bool(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    if "Adversarial_bool" not in d.columns:
        if "Adversarial" in d.columns:
            # Accept: "Adversarial", "Non Adversarial", etc.
            d["Adversarial_bool"] = (
                d["Adversarial"].astype(str).str.strip().str.lower().isin(["adversarial", "true", "yes", "1"])
                | d["Adversarial"].astype(str).str.lower().str.contains("adversarial")
            )
        else:
            d["Adversarial_bool"] = False
    d["Adversarial_bool"] = d["Adversarial_bool"].fillna(False).astype(bool)
    return d

def _prep_for_lift(df_in: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if df_in is None or df_in.empty:
        return pd.DataFrame()

    d = df_in.copy()
    # engagement
    d["Total Engagement"] = pd.to_numeric(d.get("Total Engagement", 0), errors="coerce").fillna(0)

    # actor + group labels
    if "Actor Type" not in d.columns or group_col not in d.columns:
        return pd.DataFrame()

    d["Actor Type"] = d["Actor Type"].replace("", "Unknown").fillna("Unknown").astype(str).str.strip()
    d[group_col] = d[group_col].replace("", "Unknown").fillna("Unknown").astype(str).str.strip()

    d = _ensure_adv_bool(d)
    return d

def compute_actor_lift_table(
    df_in: pd.DataFrame,
    *,
    group_col: str,              # "Narrative" or "Categories"
    baseline_mode: str,          # "Actor baseline" or "Theme baseline"
    adv_only_for_intersection: bool = True,
    min_total_posts: int = 1,
    min_adv_posts: int = 1,
) -> pd.DataFrame:
    """
    Returns tidy table with:
      Actor Type, Theme (group_col), Total posts, Adversarial posts,
      Overall avg engagement, Adversarial avg engagement, Lift (adv/baseline)
    """

    d = _prep_for_lift(df_in, group_col)
    if d.empty:
        return pd.DataFrame()

    # --- counts ---
    total_counts = (
        d.groupby(["Actor Type", group_col], dropna=False)
         .size()
         .reset_index(name="Total posts")
    )

    adv_d = d[d["Adversarial_bool"] == True]
    adv_counts = (
        adv_d.groupby(["Actor Type", group_col], dropna=False)
             .size()
             .reset_index(name="Adversarial posts")
    )

    # --- averages ---
    overall_avg = (
        d.groupby(["Actor Type", group_col], dropna=False)["Total Engagement"]
         .mean()
         .reset_index(name="Overall avg engagement")
    )
    adv_avg = (
        adv_d.groupby(["Actor Type", group_col], dropna=False)["Total Engagement"]
             .mean()
             .reset_index(name="Adversarial avg engagement")
    )

    out = total_counts.merge(adv_counts, on=["Actor Type", group_col], how="left")
    out = out.merge(overall_avg, on=["Actor Type", group_col], how="left")
    out = out.merge(adv_avg, on=["Actor Type", group_col], how="left")
    out["Adversarial posts"] = out["Adversarial posts"].fillna(0).astype(int)

    # If a pair has no adversarial posts, adv avg will be NaN → 0
    out["Adversarial avg engagement"] = out["Adversarial avg engagement"].fillna(0)

    # --- baselines ---
    # Actor baseline: actor's average engagement over ALL their posts (not just adversarial)
    actor_base = (
        d.groupby("Actor Type")["Total Engagement"]
         .mean()
         .reset_index(name="Baseline")
    )

    # Theme baseline: theme's average engagement over ALL actors (not just adversarial)
    theme_base = (
        d.groupby(group_col)["Total Engagement"]
         .mean()
         .reset_index(name="Baseline")
    )

    if baseline_mode == "Actor baseline":
        out = out.merge(actor_base, on="Actor Type", how="left")
    else:
        out = out.merge(theme_base, on=group_col, how="left")

    out["Baseline"] = out["Baseline"].fillna(0)

    # Choose intersection metric
    if adv_only_for_intersection:
        intersection = out["Adversarial avg engagement"]
    else:
        intersection = out["Overall avg engagement"]

    out["Lift (intersection/baseline)"] = np.where(
        out["Baseline"] > 0,
        intersection / out["Baseline"],
        0.0
    )

    # --- filtering to avoid noisy tiny samples ---
    out = out[out["Total posts"] >= int(min_total_posts)]
    out = out[out["Adversarial posts"] >= int(min_adv_posts)]

    # cleanup
    out["Overall avg engagement"] = out["Overall avg engagement"].fillna(0).round(1)
    out["Adversarial avg engagement"] = out["Adversarial avg engagement"].fillna(0).round(1)
    out["Lift (intersection/baseline)"] = out["Lift (intersection/baseline)"].round(3)

    return out

def render_lift_heatmap(
    lift_df: pd.DataFrame,
    *,
    group_col: str,
    top_themes: int = 15,
    key_prefix: str = "lift"
):
    if lift_df.empty:
        st.info("No data for lift heatmap (try lowering thresholds).")
        return

    # Keep top themes by adversarial posts (more meaningful)
    theme_order = (
        lift_df.groupby(group_col)["Adversarial posts"]
              .sum()
              .sort_values(ascending=False)
              .head(top_themes)
              .index.tolist()
    )
    d = lift_df[lift_df[group_col].isin(theme_order)].copy()

    # Pivot -> long (Altair friendly)
    heat = d.pivot_table(
        index="Actor Type",
        columns=group_col,
        values="Lift (intersection/baseline)",
        aggfunc="mean",
        fill_value=0.0
    )

    # Sort actors by strongest overperformance somewhere
    heat = heat.loc[heat.max(axis=1).sort_values(ascending=False).index]

    long_df = (
        heat.reset_index()
            .melt(id_vars=["Actor Type"], var_name=group_col, value_name="Lift")
    )

    # Altair heatmap (analysis logic unchanged, only rendering)
    base = alt.Chart(long_df).encode(
        x=alt.X(f"{group_col}:N", title=None, sort=list(heat.columns)),
        y=alt.Y("Actor Type:N", title=None, sort=list(heat.index)),
        tooltip=[
            alt.Tooltip("Actor Type:N"),
            alt.Tooltip(f"{group_col}:N", title=group_col),
            alt.Tooltip("Lift:Q", format=".3f"),
        ],
    )

    rect = base.mark_rect().encode(
        color=alt.Color("Lift:Q", title="Lift", scale=alt.Scale(scheme="viridis"))
    )

    text = base.transform_filter("datum.Lift != 0").mark_text(fontSize=10).encode(
        text=alt.Text("Lift:Q", format=".2f"),
        color=alt.value("white"),
    )

    chart = (rect + text).properties(height=520).configure_legend(orient="right")

    st.altair_chart(chart, width="stretch")

    with st.expander("Show lift table"):
        st.dataframe(d.sort_values("Lift (intersection/baseline)", ascending=False), width="stretch")

# ---------- UI controls + platform tabs ----------
t_all, t_fb, t_yt, t_tt = st.tabs(["All", "Facebook", "YouTube", "TikTok"])

for platform_name, tab in [("All", t_all), ("Facebook", t_fb), ("YouTube", t_yt), ("TikTok", t_tt)]:
    with tab:
        df_tab = _scope_df(platform_name)

        left, right = st.columns([1, 1])

        with left:
            analysis_target = st.selectbox(
                "Analyze by",
                ["Narrative", "Categories"],
                key=f"lift_group_{platform_name}"
            )

        with right:
            baseline_mode = st.selectbox(
                "Compare against",
                ["Actor baseline", "Theme baseline"],
                help="Actor baseline = actor’s normal avg engagement. Theme baseline = theme’s ecosystem avg engagement.",
                key=f"lift_base_{platform_name}"
            )

        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            top_themes = st.slider("Top themes", 5, 30, 15, key=f"lift_top_{platform_name}")
        with c2:
            min_total_posts = st.number_input("Min total posts (actor+theme)", 1, 5000, 1, key=f"lift_min_total_{platform_name}")
        with c3:
            min_adv_posts = st.number_input("Min adversarial posts (actor+theme)", 0, 5000, 1, key=f"lift_min_adv_{platform_name}")

        adv_only_for_intersection = True  # keep it strict to your request: adversarial theme performance
        lift_df = compute_actor_lift_table(
            df_tab,
            group_col=analysis_target,
            baseline_mode=baseline_mode,
            adv_only_for_intersection=adv_only_for_intersection,
            min_total_posts=int(min_total_posts),
            min_adv_posts=int(min_adv_posts),
        )

        st.caption("Lift > 1 = overperforms; Lift < 1 = underperforms.")
        render_lift_heatmap(
            lift_df,
            group_col=analysis_target,
            top_themes=int(top_themes),
            key_prefix=f"lift_{platform_name}"
        )
st.divider()

# =========================
# MISINFORMATION/HATE — Expander with:
# - independent platform tabs per chart
# - multi-select bucket logic (HS / Misinfo / Both / Other Adversarial)
# - exclude N/A completely
# - Other Adversarial = adversarial posts NOT tagged HS/Misinfo
# =========================

import re

def _parse_mh_cell(x) -> list[str]:
    if pd.isna(x):
        return []
    s = str(x).strip()
    if not s:
        return []
    parts = re.split(r"[,\n;|]+", s)
    toks = []
    for p in parts:
        t = p.strip()
        if not t:
            continue
        tl = t.lower()
        if tl in {"hate", "hate speech", "hatespeech"}:
            toks.append("Hate Speech")
        elif tl in {"misinformation", "misinfo"}:
            toks.append("Misinformation")
        elif tl in {"n/a", "na", "none"}:
            toks.append("N/A")
        else:
            # keep any other token (we won't make buckets from it)
            toks.append(t)
    # de-dupe
    out, seen = [], set()
    for t in toks:
        if t not in seen:
            out.append(t); seen.add(t)
    return out

def _mh_bucket(tokens: list[str], is_adv: bool) -> str | None:
    """
    Strict logic:
    - If tagged HS/Misinfo/Both -> return that bucket (regardless of adversarial flag)
    - Else if NOT tagged but adversarial -> Other Adversarial
    - Else -> exclude (None)
    - N/A-only -> exclude (None)
    """
    # N/A-only => drop
    if tokens and all(t == "N/A" for t in tokens):
        return None

    # ignore N/A if mixed
    tokens = [t for t in tokens if t != "N/A"]

    has_hate = "Hate Speech" in tokens
    has_misinfo = "Misinformation" in tokens

    if has_hate and has_misinfo:
        return "Hate Speech + Misinformation"
    if has_hate:
        return "Hate Speech"
    if has_misinfo:
        return "Misinformation"

    # Only adversarial leftovers become Other Adversarial
    if bool(is_adv):
        return "Other Adversarial"

    # Non-adversarial leftovers are not part of this section
    return None

def _apply_common_filters_without_platform(d: pd.DataFrame) -> pd.DataFrame:
    """Apply all your current sidebar filters EXCEPT platform."""
    out = d.copy()

    if f_actor:
        out = out[out["Actor Type"].isin(f_actor)]
    if f_cat:
        out = out[out["Categories"].isin(f_cat)]
    if f_narr:
        out = out[out["Narrative"].isin(f_narr)]
    if f_mh:
        out = out[out["Misinformation/Hate"].isin(f_mh)]
    if f_sev:
        out = out[out["Severity"].isin(f_sev)]

    # Date filter
    if isinstance(date_range, tuple) and len(date_range) == 2:
        d0, d1 = date_range
        out = out[
            (out["PostedAt_dt"].isna())
            | ((out["PostedAt_dt"] >= pd.to_datetime(d0)) & (out["PostedAt_dt"] <= pd.to_datetime(d1) + pd.Timedelta(days=1)))
        ]

    return out

def _mh_scope(platform_name: str) -> pd.DataFrame:
    # 1) Start from the sheet-tab dataframes (same as other charts)
    base = _scope_df(platform_name)  # returns fb/yt/tt or concat for "All"

    if base.empty:
        return base

    # 2) If denom_mode is "Filtered data", apply the SAME global filters
    #    (but do NOT apply the global f_adv — this expander is adversarial-only by design)
    if denom_mode != "Full dataset":
        if f_actor:
            base = base[base["Actor Type"].isin(f_actor)]
        if f_cat:
            base = base[base["Categories"].isin(f_cat)]
        if f_narr:
            base = base[base["Narrative"].isin(f_narr)]
        if f_mh:
            base = base[base["Misinformation/Hate"].isin(f_mh)]
        if f_sev:
            base = base[base["Severity"].isin(f_sev)]

        if isinstance(date_range, tuple) and len(date_range) == 2:
            d0, d1 = date_range
            base = base[
                (base["PostedAt_dt"].isna())
                | ((base["PostedAt_dt"] >= pd.to_datetime(d0)) &
                   (base["PostedAt_dt"] <= pd.to_datetime(d1) + pd.Timedelta(days=1)))
            ]

    # 3) Adversarial only for this section
    #base = base[base["Adversarial_bool"] == True].copy()
    return base

# =========================
# Misinformation/Hate Analysis (ONE expander, TWO rows)
# Row 1: Share donut (left) + Avg engagement bars (right)
# Row 2: Actor activity (left) + 100% stacked distribution (right)
# =========================

with st.expander("Misinformation/Hate Analysis", expanded=True):

    ORDER = ["Hate Speech", "Misinformation", "Hate Speech + Misinformation", "Other Adversarial"]

    def _mh_bucket_from_tokens(tokens) -> str:
        """
        tokens: output of _parse_mh_cell (list/iterable or None)
        Rule:
          - if tokens imply both hate & misinfo -> Both bucket
          - else hate -> Hate Speech
          - else misinfo -> Misinformation
          - else -> Other Adversarial
        """
        if tokens is None:
            return "Other Adversarial"
        try:
            toks = [str(t).strip().lower() for t in tokens if str(t).strip()]
        except Exception:
            toks = [str(tokens).strip().lower()] if str(tokens).strip() else []

        has_hate = any(("hate" in t) or ("hs" == t) or ("hate speech" in t) for t in toks)
        has_misinfo = any(("misinfo" in t) or ("misinformation" in t) for t in toks)

        if has_hate and has_misinfo:
            return "Hate Speech + Misinformation"
        if has_hate:
            return "Hate Speech"
        if has_misinfo:
            return "Misinformation"
        return "Other Adversarial"

    def _mh_adv_bucketed_scope(platform_name: str) -> pd.DataFrame:
        """
        Your rule implemented:
        - Filter adversarial FIRST.
        - Then bucket Misinformation/Hate; N/A/empty => Other Adversarial.
        Returns ONLY adversarial rows with a bucket in ORDER.
        """
        dfp = _scope_df(platform_name)
        if dfp is None or dfp.empty:
            return pd.DataFrame()

        needed = ["Actor Type", "Adversarial_bool", "Misinformation/Hate"]
        if any(c not in dfp.columns for c in needed):
            return pd.DataFrame()

        d = dfp.copy()
        d["Adversarial_bool"] = d["Adversarial_bool"].fillna(False).astype(bool)

        # ✅ 1) Filter adversarial FIRST
        adv = d[d["Adversarial_bool"] == True].copy()
        if adv.empty:
            return pd.DataFrame()

        # ✅ 2) Bucket within adversarial, with N/A/empty => Other Adversarial
        adv["_mh_tokens"] = adv["Misinformation/Hate"].apply(_parse_mh_cell)
        adv["_mh_bucket"] = adv["_mh_tokens"].apply(_mh_bucket_from_tokens)

        adv["_mh_bucket"] = pd.Categorical(adv["_mh_bucket"], categories=ORDER, ordered=True)
        return adv

    # -------------------------
    # ROW 1 — Donut + Avg engagement
    # -------------------------
    left, right = st.columns(2)

    # -------- LEFT donut with its own tabs --------
    with left:
        st.markdown("##### Share of hate speech, misinformation, and other adversarial narratives")
        lt_all, lt_fb, lt_yt, lt_tt = st.tabs(["All", "Facebook", "YouTube", "TikTok"])

        for platform_name, tab in [("All", lt_all), ("Facebook", lt_fb), ("YouTube", lt_yt), ("TikTok", lt_tt)]:
            with tab:
                mh = _mh_adv_bucketed_scope(platform_name)
                if mh is None or mh.empty:
                    st.info("No adversarial data for this platform (or missing required columns).")
                    continue

                counts = mh["_mh_bucket"].value_counts().reset_index()
                counts.columns = ["Post Type", "Posts"]

                counts["Post Type"] = pd.Categorical(counts["Post Type"], categories=ORDER, ordered=True)
                counts = counts.sort_values("Post Type")

                pie_with_table(
                    counts,
                    name_col="Post Type",
                    value_col="Posts",
                    title="",
                    n_value=int(counts["Posts"].sum()),
                    show_labels=SHOW_LABELS,
                    show_table=SHOW_TABLES,
                    key_prefix=f"mh_share_{platform_name.lower()}",
                    hole=0.55,
                    height=420,
                    table_value_name="Posts",
                )

    # -------- RIGHT bar with its own tabs --------
    with right:
        st.markdown("##### Average engagement of Hate & Misinformation post types")
        rt_all, rt_fb, rt_yt, rt_tt = st.tabs(["All", "Facebook", "YouTube", "TikTok"])

        for platform_name, tab in [("All", rt_all), ("Facebook", rt_fb), ("YouTube", rt_yt), ("TikTok", rt_tt)]:
            with tab:
                dfp = _scope_df(platform_name)
                if dfp is None or dfp.empty:
                    st.info("No data.")
                    continue

                required_cols = ["Adversarial_bool", "Misinformation/Hate", "Total Engagement"]
                missing = [c for c in required_cols if c not in dfp.columns]
                if missing:
                    st.info(f"Missing required columns: {', '.join(missing)}")
                    continue

                # baseline should be overall total avg over ALL posts in the platform scope
                k = overall_avg_compare(dfp)
                baseline_avg = float(k.get("total_avg", 0) or 0)

                mh = _mh_adv_bucketed_scope(platform_name)
                if mh is None or mh.empty:
                    st.info("No adversarial posts for this platform.")
                    continue

                mh["Total Engagement"] = pd.to_numeric(mh.get("Total Engagement", 0), errors="coerce").fillna(0)

                mh_adv = (
                    mh.groupby("_mh_bucket", as_index=False)["Total Engagement"]
                      .mean()
                      .rename(columns={"_mh_bucket": "Post Type", "Total Engagement": "Adversarial avg"})
                )
                mh_adv["Adversarial avg"] = (
                    pd.to_numeric(mh_adv["Adversarial avg"], errors="coerce")
                      .fillna(0).round(0).astype(int)
                )
                mh_adv["Baseline (overall total avg)"] = float(baseline_avg)

                mh_adv["Post Type"] = pd.Categorical(mh_adv["Post Type"], categories=ORDER, ordered=True)
                mh_adv = mh_adv.sort_values("Post Type")

                show_both = st.toggle(
                    "Compare with total posts average",
                    value=True,
                    key=f"cmp_mh_{platform_name}",
                )

                use_cols = ["Adversarial avg"]
                if show_both:
                    use_cols = ["Baseline (overall total avg)", "Adversarial avg"]

                plot_df = mh_adv.melt(
                    id_vars=["Post Type"],
                    value_vars=use_cols,
                    var_name="Series",
                    value_name="Avg engagement",
                )

                fig = px.bar(
                    plot_df,
                    x="Post Type",
                    y="Avg engagement",
                    color="Series",
                    barmode="group",
                    text="Avg engagement",
                )
                fig.update_traces(textposition="outside")
                fig.update_layout(
                    height=420,
                    margin=dict(t=70, b=10, l=10, r=10),
                    yaxis_title="Average engagement",
                    xaxis_title="Post Type",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.15,
                        xanchor="left",
                        x=0
                    ),
                )

                st.plotly_chart(fig, width="stretch")

                if st.checkbox("Show numbers table", key=f"tbl_mh_{platform_name}"):
                    st.dataframe(mh_adv, width="stretch")

    st.divider()

    # -------------------------
    # ROW 2 — Actor activity + distribution
    # -------------------------
    st.markdown("## Actor activity (Hate/Misinformation)")

    left2, right2 = st.columns(2)

    # -------- LEFT: Actor activity + MH rate (own tabs) --------
    with left2:
        st.markdown("### Actor activity and MH rate")
        st.caption("Bar: total adversarial posts by actor type. Line: % of those adversarial posts that are Hate Speech / Misinformation / Both.")

        lt_all2, lt_fb2, lt_yt2, lt_tt2 = st.tabs(["All", "Facebook", "YouTube", "TikTok"])

        for platform_name, tab in [("All", lt_all2), ("Facebook", lt_fb2), ("YouTube", lt_yt2), ("TikTok", lt_tt2)]:
            with tab:
                mh2 = _mh_adv_bucketed_scope(platform_name)
                if mh2 is None or mh2.empty:
                    st.info("No adversarial rows for this platform.")
                    continue

                # ✅ BAR: total adversarial posts by actor type
                a_total = (
                    mh2.groupby("Actor Type", as_index=False)
                       .size()
                       .rename(columns={"size": "Adversarial posts"})
                )

                # ✅ LINE numerator: typed MH buckets only
                typed = mh2[mh2["_mh_bucket"].isin(["Hate Speech", "Misinformation", "Hate Speech + Misinformation"])].copy()
                a_typed = (
                    typed.groupby("Actor Type", as_index=False)
                         .size()
                         .rename(columns={"size": "Typed MH posts"})
                )

                actor_rate = a_total.merge(a_typed, on="Actor Type", how="left").fillna({"Typed MH posts": 0})
                actor_rate["MH rate (%)"] = (actor_rate["Typed MH posts"] / actor_rate["Adversarial posts"] * 100).round(1)
                actor_rate = actor_rate.sort_values("Adversarial posts", ascending=True)

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=actor_rate["Actor Type"],
                    x=actor_rate["Adversarial posts"],
                    name="Adversarial posts",
                    orientation="h",
                    text=actor_rate["Adversarial posts"],
                    textposition="inside",
                ))
                fig.add_trace(go.Scatter(
                    y=actor_rate["Actor Type"],
                    x=actor_rate["MH rate (%)"],
                    name="% typed MH",
                    mode="lines+markers",
                    xaxis="x2",
                ))

                fig.update_layout(
                    height=420,
                    margin=dict(t=70, b=20, l=20, r=20),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.15,
                        xanchor="left",
                        x=0
                    ),
                    xaxis=dict(title="Adversarial posts"),
                    xaxis2=dict(title="% typed MH", overlaying="x", side="top", range=[0, 100]),
                    yaxis=dict(title="Actor Type"),
                )

                st.caption(f"N = {len(mh2):,}")
                st.plotly_chart(fig, width="stretch", key=f"mh_actor_rate_{platform_name.lower()}")

                if st.checkbox("Show table", key=f"mh_actor_rate_{platform_name.lower()}_table_toggle"):
                    st.dataframe(actor_rate, width="stretch")

    # -------- RIGHT: 100% stacked distribution (own tabs) --------
    with right2:
        st.markdown("### MH post type distribution by Actor Type")
        st.caption("100% stacked: each Actor Type sums to 100% across Hate Speech / Misinformation / Both / Other Adversarial.")

        rt_all2, rt_fb2, rt_yt2, rt_tt2 = st.tabs(["All", "Facebook", "YouTube", "TikTok"])

        for platform_name, tab in [("All", rt_all2), ("Facebook", rt_fb2), ("YouTube", rt_yt2), ("TikTok", rt_tt2)]:
            with tab:
                mh2 = _mh_adv_bucketed_scope(platform_name)
                if mh2 is None or mh2.empty:
                    st.info("No adversarial rows for this platform.")
                    continue

                pivot = (
                    mh2.groupby(["Actor Type", "_mh_bucket"], as_index=False)
                       .size()
                       .rename(columns={"size": "Posts", "_mh_bucket": "Post Type"})
                )

                pivot["Posts"] = pd.to_numeric(pivot["Posts"], errors="coerce").fillna(0)
                totals = pivot.groupby("Actor Type")["Posts"].transform("sum")
                pivot["Percent"] = (pivot["Posts"] / totals * 100).fillna(0)

                actor_order = mh2.groupby("Actor Type").size().sort_values(ascending=True).index.tolist()

                fig2 = px.bar(
                    pivot,
                    y="Actor Type",
                    x="Percent",
                    color="Post Type",
                    orientation="h",
                    barmode="stack",
                    category_orders={"Post Type": ORDER, "Actor Type": actor_order},
                    text=pivot["Percent"].round(1).astype(str) + "%"
                )

                fig2.update_traces(textposition="inside", cliponaxis=False)
                fig2.update_layout(
                    height=420,
                    margin=dict(t=70, b=20, l=20, r=20),
                    xaxis_title="Percent of adversarial posts",
                    yaxis_title="Actor Type",
                    legend_title="Post Type",
                    xaxis=dict(range=[0, 100]),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.15,
                        xanchor="left",
                        x=0
                    ),
                )

                st.plotly_chart(fig2, width="stretch", key=f"mh_actor_pct_{platform_name.lower()}")

                if st.checkbox("Show table", key=f"mh_actor_pct_{platform_name.lower()}_table_toggle"):
                    st.dataframe(pivot, width="stretch")


st.divider()

# =========================
# When adversarial narratives spike (daily %)
# =========================
st.divider()
st.markdown("## When adversarial narratives spike (daily %)")

import pandas as pd
import plotly.graph_objects as go
import altair as alt

# ---------- helpers ----------
def _pick_date_source(d: pd.DataFrame):
    for c in [
        "PostedAt_dt",
        "Posted At", "PostedAt",
        "publishedAt", "Published At",
        "Date", "Timestamp", "Datetime",
    ]:
        if c in d.columns:
            return c
    return None

def ensure_posted_dt(d: pd.DataFrame) -> pd.DataFrame:
    if d is None or d.empty:
        return d
    x = d.copy()
    src = _pick_date_source(x)
    if src is None:
        x["PostedAt_dt"] = pd.NaT
        return x
    if src == "PostedAt_dt":
        x["PostedAt_dt"] = pd.to_datetime(x["PostedAt_dt"], errors="coerce")
        if x["PostedAt_dt"].notna().mean() >= 0.1:
            return x
        for alt in ["Posted At", "publishedAt", "Published At", "Date"]:
            if alt in x.columns:
                src = alt
                break
    x["PostedAt_dt"] = pd.to_datetime(x[src], errors="coerce", utc=False)
    return x

def ensure_adv_bool(d: pd.DataFrame) -> pd.DataFrame:
    if d is None or d.empty:
        return d
    x = d.copy()
    if "Adversarial_bool" not in x.columns:
        if "Adversarial" in x.columns:
            x["Adversarial_bool"] = (
                x["Adversarial"]
                .astype(str)
                .str.strip()
                .str.lower()
                .str.contains("adversarial")
            )
        else:
            x["Adversarial_bool"] = False
    else:
        x["Adversarial_bool"] = x["Adversarial_bool"].astype(bool)
    return x

def daily_adv_share(d: pd.DataFrame) -> pd.DataFrame:
    if d is None or d.empty:
        return pd.DataFrame(columns=["Date", "Total posts", "Adversarial posts", "Adversarial %"])

    x = ensure_posted_dt(d)
    x = ensure_adv_bool(x)
    x = x[~x["PostedAt_dt"].isna()].copy()
    if x.empty:
        return pd.DataFrame(columns=["Date", "Total posts", "Adversarial posts", "Adversarial %"])

    x["Date"] = x["PostedAt_dt"].dt.date

    out = (
        x.groupby("Date", as_index=False)
         .agg(
            **{
                "Total posts": ("Adversarial_bool", "size"),
                "Adversarial posts": ("Adversarial_bool", "sum"),
            }
         )
    )
    out["Adversarial %"] = (out["Adversarial posts"] / out["Total posts"] * 100).round(1)
    return out.sort_values("Date")

def _counts(d: pd.DataFrame):
    x = ensure_adv_bool(d)
    total_n = len(x)
    adv_n = int(x["Adversarial_bool"].sum()) if total_n else 0
    return total_n, adv_n

# ---------- datasets (from 3 tabs) ----------
DATASETS = {
    "Facebook": fb,
    "YouTube": yt,
    "TikTok": tt,
}
ALL_DF = pd.concat([fb, yt, tt], ignore_index=True)

t_all, t_fb, t_yt, t_tt = st.tabs(["All", "Facebook", "YouTube", "TikTok"])
TAB_MAP = {
    "All": (t_all, ALL_DF),
    "Facebook": (t_fb, DATASETS["Facebook"]),
    "YouTube": (t_yt, DATASETS["YouTube"]),
    "TikTok": (t_tt, DATASETS["TikTok"]),
}

# ---------- render ----------
for name, (tab, d0) in TAB_MAP.items():
    with tab:
        total_n, adv_n = _counts(d0)

        k1, k2, k3 = st.columns(3)
        k1.metric("Total posts (N)", f"{total_n:,}")
        k2.metric("Adversarial posts (N)", f"{adv_n:,}")
        k3.metric(
            "Adversarial share (overall)",
            f"{(adv_n / total_n * 100):.1f}%" if total_n else "0.0%"
        )

        dday = daily_adv_share(d0)
        if dday.empty:
            st.info("No dated posts available for this view.")
            continue

        fig = go.Figure()

        # Bars: total posts
        fig.add_trace(go.Bar(
            x=dday["Date"],
            y=dday["Total posts"],
            name="Total posts",
            opacity=0.35,
        ))

        # Line: adversarial %
        fig.add_trace(go.Scatter(
            x=dday["Date"],
            y=dday["Adversarial %"],
            name="Adversarial %",
            mode="lines+markers",
            yaxis="y2",
            line=dict(color="#1C4D8D", width=2),
            marker=dict(color="#1C4D8D", size=6),
            hovertemplate="Date=%{x}<br>Adversarial %=%{y}%<extra></extra>",
        ))

        fig.update_layout(
            height=420,
            margin=dict(t=10, b=10, l=10, r=10),
            xaxis=dict(
                title="Date",
                range=[dday["Date"].min(), dday["Date"].max()],  # removes empty space
            ),
            yaxis=dict(title="Total posts"),
            yaxis2=dict(
                title="Adversarial %",
                overlaying="y",
                side="right",
                range=[0, 100],
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )

        st.plotly_chart(fig, width="stretch", key=f"adv_daily_pct_{name}")

        with st.expander("Show daily table"):
            st.dataframe(dday, width="stretch")

# =========================
# Narrative × Platform (normalized adversarial rate %) — grouped bars
# - Only adversarial narratives (derived from Adversarial posts)
# - Drop Neutral
# - No All button
# - Default multiselect empty => show ALL narratives
# =========================
st.divider()
st.markdown("## Narrative amplification by platform (normalized %)")
st.write("This chart normalizes adversarial narratives by total platform activity, showing what share of all posts on each platform contain a given adversarial narrative, highlighting which narratives are more prevalent on which platforms.")

import pandas as pd
import plotly.express as px

# ---------- helpers ----------
def ensure_adv_bool(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    if "Adversarial_bool" not in d.columns:
        if "Adversarial" in d.columns:
            d["Adversarial_bool"] = (
                d["Adversarial"].astype(str).str.strip().str.lower().str.contains("adversarial")
            )
        else:
            d["Adversarial_bool"] = False
    else:
        d["Adversarial_bool"] = d["Adversarial_bool"].fillna(False).astype(bool)
    return d

def norm_str(s: pd.Series) -> pd.Series:
    return s.fillna("Unknown").astype(str).str.strip().replace("", "Unknown")

def is_neutral_series(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().str.lower().eq("neutral")

# ---------- platform datasets ----------
platform_data = {
    "Facebook": ensure_adv_bool(fb),
    "YouTube": ensure_adv_bool(yt),
    "TikTok": ensure_adv_bool(tt),
}

# Ensure Narrative exists + normalize
for p, dfp in platform_data.items():
    dfp = dfp.copy()
    if "Narrative" not in dfp.columns:
        dfp["Narrative"] = "Unknown"
    dfp["Narrative"] = norm_str(dfp["Narrative"])
    platform_data[p] = dfp

# ---------- compute platform totals (denominator stays ALL posts) ----------
platform_totals = {p: int(len(dfp)) for p, dfp in platform_data.items()}

# ---------- build adversarial-only narrative universe (excludes Neutral) ----------
adv_narratives_series = []
for p, dfp in platform_data.items():
    advp = dfp[dfp["Adversarial_bool"] == True].copy()
    if advp.empty:
        continue
    advp = advp[~is_neutral_series(advp["Narrative"])].copy()
    if advp.empty:
        continue
    adv_narratives_series.append(advp["Narrative"])

all_narratives = sorted(
    pd.concat(adv_narratives_series, ignore_index=True).dropna().unique().tolist()
) if adv_narratives_series else []

if not all_narratives:
    st.info("No adversarial narratives available (after excluding Neutral).")
else:
    # ---------- UI: empty default selection; empty means "show all" ----------
    selected_narratives = st.multiselect(
        "Filter narratives (optional)",
        options=all_narratives,
        default=[],
        key="narplat_filter_advonly",
        help="Select one or more to filter."
    )

    # empty selection => no filter (show all)
    active_narratives = selected_narratives if selected_narratives else all_narratives

    # ---------- compute normalized rates ----------
    rows = []
    for platform, dfp in platform_data.items():
        total_posts = platform_totals.get(platform, 0)
        if total_posts <= 0:
            continue

        adv = dfp[dfp["Adversarial_bool"] == True].copy()
        if adv.empty:
            continue

        adv = adv[~is_neutral_series(adv["Narrative"])].copy()
        adv = adv[adv["Narrative"].isin(active_narratives)].copy()

        if adv.empty:
            continue

        counts = adv.groupby("Narrative").size().reset_index(name="Adversarial posts")
        counts["Platform"] = platform
        counts["Total posts"] = total_posts
        counts["Rate (%)"] = (counts["Adversarial posts"] / total_posts * 100).round(2)
        rows.append(counts)

    rate_long = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["Narrative", "Adversarial posts", "Platform", "Total posts", "Rate (%)"]
    )

    # Full grid so missing platform-narrative combos appear as 0
    platforms_kept = list(platform_data.keys())
    grid = pd.MultiIndex.from_product(
        [active_narratives, platforms_kept],
        names=["Narrative", "Platform"]
    ).to_frame(index=False)

    plot_df = grid.merge(rate_long, on=["Narrative", "Platform"], how="left")
    plot_df["Adversarial posts"] = plot_df["Adversarial posts"].fillna(0).astype(int)
    plot_df["Total posts"] = plot_df["Total posts"].fillna(plot_df["Platform"].map(platform_totals)).fillna(0).astype(int)
    plot_df["Rate (%)"] = plot_df["Rate (%)"].fillna(0.0)

    # Grouped bar chart
    fig = px.bar(
        plot_df,
        x="Narrative",
        y="Rate (%)",
        color="Platform",
        barmode="group",
        text="Rate (%)",
        hover_data={
            "Adversarial posts": True,
            "Total posts": True,
            "Rate (%)": True,
            "Platform": True,
            "Narrative": True,
        },
    )

    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")

    fig.update_layout(
        height=520,
        margin=dict(t=70, b=10, l=10, r=10),
        xaxis_title="Narrative",
        yaxis_title="Adversarial rate (% of all posts on that platform)",
        legend_title_text="Platform",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.15,
            xanchor="left",
            x=0
        ),
    )
    fig.update_xaxes(tickangle=-30)

    st.plotly_chart(fig, width="stretch")

    with st.expander("Show underlying table"):
        st.dataframe(plot_df.sort_values(["Narrative", "Platform"]), width="stretch")

# =========================
# Top Adversarial Posters (3 tables side-by-side)
# - NO "All" filter
# - FB: AuthorId/Author
# - YT: channelId/channelTitle
# - TT: author_id/author_username
# MH Post = ONLY HS / Misinfo / Both (excludes Other Adversarial + excludes N/A)
# Avg Engagement = adversarial posts only
# =========================

st.markdown("## Top Adversarial Posters")

MH_TYPED = {"Hate Speech", "Misinformation", "Hate Speech + Misinformation"}  # only these count as MH


def _as_bool_series(s: pd.Series) -> pd.Series:
    s2 = s.astype(str).str.strip().str.lower()
    truthy = {"adversarial", "true", "yes", "y", "1", "adv"}
    return s2.isin(truthy)


def _num_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0)


def _prep_platform(platform_name: str, df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    if "Adversarial" not in df.columns:
        df["Adversarial"] = ""
    if "Misinformation/Hate" not in df.columns:
        df["Misinformation/Hate"] = ""

    # boolean adversarial
    df["Adversarial_bool"] = _as_bool_series(df["Adversarial"])

    # total engagement (Avg Engagement later averaged only over adversarial posts)
    if platform_name == "Facebook":
        if "Total Engagement" not in df.columns:
            for c in ["Comment Count", "Reaction Count", "Shared Count"]:
                if c not in df.columns:
                    df[c] = 0
            df["Total Engagement"] = (
                _num_series(df["Comment Count"]) +
                _num_series(df["Reaction Count"]) +
                _num_series(df["Shared Count"])
            )
        else:
            df["Total Engagement"] = _num_series(df["Total Engagement"])

    elif platform_name == "YouTube":
        # use explicit if present, else compute
        if "Total Engagement" in df.columns:
            df["Total Engagement"] = _num_series(df["Total Engagement"])
        else:
            for c in ["viewCount", "likeCount", "commentCount"]:
                if c not in df.columns:
                    df[c] = 0
            df["Total Engagement"] = (
                _num_series(df["viewCount"]) +
                _num_series(df["likeCount"]) +
                _num_series(df["commentCount"])
            )

    else:  # TikTok
        if "Total Engagement" in df.columns:
            df["Total Engagement"] = _num_series(df["Total Engagement"])
        else:
            if "like_count" not in df.columns:
                df["like_count"] = 0
            if "view_count" not in df.columns:
                df["view_count"] = 0
            if "comment_count" not in df.columns and "comment_coun" not in df.columns:
                df["comment_count"] = 0
            comm_col = "comment_count" if "comment_count" in df.columns else "comment_coun"
            df["Total Engagement"] = (
                _num_series(df["view_count"]) +
                _num_series(df["like_count"]) +
                _num_series(df[comm_col])
            )

    return df


def _top_adv_one(platform_name: str) -> pd.DataFrame:
    # These MUST exist in your script: fb_raw / yt_raw / tt_raw
    if platform_name == "Facebook":
        df_raw = fb_raw.copy()
        id_col, name_col = "AuthorId", "Author"
    elif platform_name == "YouTube":
        df_raw = yt_raw.copy()
        id_col, name_col = "channelId", "channelTitle"
    else:
        df_raw = tt_raw.copy()
        id_col, name_col = "author_id", "author_username"

    if df_raw is None or df_raw.empty:
        return pd.DataFrame()

    df = _prep_platform(platform_name, df_raw)

    actor_col = "Actor Type"

    # Must exist
    for c in [id_col, name_col, actor_col, "Adversarial_bool", "Misinformation/Hate", "Total Engagement"]:
        if c not in df.columns:
            return pd.DataFrame()

    # adversarial universe only
    adv = df[df["Adversarial_bool"] == True].copy()
    if adv.empty:
        return pd.DataFrame()

    # MH bucket on adversarial posts
    adv["_mh_tokens"] = adv["Misinformation/Hate"].apply(_parse_mh_cell)
    adv["_mh_bucket"] = adv.apply(lambda r: _mh_bucket(r["_mh_tokens"], True), axis=1)

    # MH posts = ONLY typed buckets (HS/Misinfo/Both)
    adv["_is_mh_typed"] = adv["_mh_bucket"].isin(MH_TYPED)

    # clean strings
    adv[id_col] = adv[id_col].astype(str).str.strip()
    adv[name_col] = adv[name_col].astype(str).str.strip()
    adv[actor_col] = adv[actor_col].astype(str).str.strip().replace("", "Unknown").fillna("Unknown")

    out = (
        adv.groupby(id_col, dropna=False)
           .agg(
               **{
                   name_col: (name_col, lambda s: s.dropna().iloc[0] if len(s.dropna()) else ""),
                   "Actor Type": (actor_col, lambda s: s.dropna().iloc[0] if len(s.dropna()) else "Unknown"),
                   "No of Adv Post": (id_col, "size"),
                   "No of MH Post": ("_is_mh_typed", "sum"),
                   "Avg Engagement": ("Total Engagement", "mean"),
               }
           )
           .reset_index()
    )

    out["Avg Engagement"] = pd.to_numeric(out["Avg Engagement"], errors="coerce").fillna(0).round(1)
    out = out.sort_values(["No of Adv Post", "No of MH Post", "Avg Engagement"], ascending=[False, False, False])
    return out


def _render_table_in_col(platform_name: str, df_out: pd.DataFrame, key_prefix: str):
    st.markdown(f"### {platform_name}")

    if df_out is None or df_out.empty:
        st.info("No adversarial data found (or required columns missing).")
        return

    top_n = st.slider(
        "Show top N",
        10, 300, 50,
        step=10,
        key=f"{key_prefix}_topn"
    )

    show = df_out.head(top_n)
    st.dataframe(show, width="stretch", height=520)


# ---------- compute once ----------
fb_out = _top_adv_one("Facebook")
yt_out = _top_adv_one("YouTube")
tt_out = _top_adv_one("TikTok")

# ---------- layout: 3 tables in one row ----------
c1, c2, c3 = st.columns(3)

with c1:
    _render_table_in_col("Facebook", fb_out, "topadv_fb")

with c2:
    _render_table_in_col("YouTube", yt_out, "topadv_yt")

with c3:
    _render_table_in_col("TikTok", tt_out, "topadv_tt")

st.divider()

# =========================
# Content Block (ONE Platform selector in Row 1 + 2-row filters + controls row + cards)
# Filters here affect ONLY this block.
# Requires raw dfs: fb_raw / yt_raw / tt_raw
# =========================

import math
import re
import numpy as np
import pandas as pd
from html import unescape

st.markdown("## Content")

# ---------- helpers ----------
def _to_num(x):
    return pd.to_numeric(x, errors="coerce").fillna(0)

def _norm_text(s):
    return s.astype(str).fillna("").str.strip()

def _adv_bool_from_text(col: pd.Series) -> pd.Series:
    s = _norm_text(col).str.lower()
    truthy = {"adversarial", "true", "yes", "y", "1", "adv"}
    return s.isin(truthy)

def _ensure_total_engagement_for_block(platform_name: str, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Total Engagement" in df.columns:
        df["Total Engagement"] = _to_num(df["Total Engagement"])
        return df

    if platform_name == "Facebook":
        for c in ["Comment Count", "Reaction Count", "Shared Count"]:
            if c not in df.columns:
                df[c] = 0
        df["Total Engagement"] = _to_num(df["Comment Count"]) + _to_num(df["Reaction Count"]) + _to_num(df["Shared Count"])
        return df

    if platform_name == "YouTube":
        for c in ["viewCount", "likeCount", "commentCount"]:
            if c not in df.columns:
                df[c] = 0
        df["Total Engagement"] = _to_num(df["viewCount"]) + _to_num(df["likeCount"]) + _to_num(df["commentCount"])
        return df

    # TikTok
    if "like_count" not in df.columns:
        df["like_count"] = 0
    if "view_count" not in df.columns:
        df["view_count"] = 0
    comm_col = "comment_count" if "comment_count" in df.columns else ("comment_coun" if "comment_coun" in df.columns else None)
    if comm_col is None:
        df["comment_count"] = 0
        comm_col = "comment_count"
    df["Total Engagement"] = _to_num(df["view_count"]) + _to_num(df["like_count"]) + _to_num(df[comm_col])
    return df

def _mh_show_only_typed(cell) -> str:
    """Show only if includes Misinformation or Hate/Hate Speech; else blank."""
    if cell is None:
        return ""
    txt = str(cell).strip()
    if not txt or txt.lower() in {"na", "n/a", "none", "nan"}:
        return ""
    low = txt.lower()
    has_mis = "misinformation" in low
    has_hate = "hate" in low
    return txt if (has_mis or has_hate) else ""

def _explode_multiselect_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([], dtype=str)
    vals = (
        df[col].dropna().astype(str)
        .apply(lambda x: [p.strip() for p in x.split(",")] if "," in x else [x.strip()])
        .explode()
        .dropna()
        .astype(str)
        .str.strip()
    )
    vals = vals[vals.ne("") & ~vals.str.lower().isin({"na", "n/a", "none", "nan"})]
    return vals

def _safe_multiselect_options(df: pd.DataFrame, col: str) -> list:
    vals = _explode_multiselect_col(df, col)
    return sorted(vals.unique().tolist())

def _apply_list_filter(df: pd.DataFrame, col: str, selected: list) -> pd.DataFrame:
    if not selected or col not in df.columns:
        return df

    s = df[col].fillna("").astype(str)

    def _cell_has_any(cell: str) -> bool:
        parts = [p.strip() for p in str(cell).split(",")] if "," in str(cell) else [str(cell).strip()]
        parts = [p for p in parts if p and p.lower() not in {"na", "n/a", "none", "nan"}]
        return any(sel in parts for sel in selected)

    return df[s.apply(_cell_has_any)].copy()

# --- HTML sanitization for FB content that sometimes contains markup ---
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")

def _strip_html(text: str) -> str:
    if text is None:
        return ""
    t = str(text)
    if "<" not in t and "&" not in t:
        return t
    t = unescape(t)
    t = _HTML_TAG_RE.sub(" ", t)
    t = _WHITESPACE_RE.sub(" ", t).strip()
    return t

def _card_escape(text: str) -> str:
    t = _strip_html(text)
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def _pick_first_url(row: pd.Series, candidates: list[str]) -> str:
    for c in candidates:
        if c in row.index:
            v = str(row.get(c, "")).strip()
            if v and v.lower() not in {"na", "n/a", "none", "nan"}:
                return v
    return ""

def _maybe_link(url: str, label: str = "Open link") -> str:
    u = str(url or "").strip()
    if not u or u.lower() in {"na", "n/a", "none", "nan"}:
        return ""
    u_esc = u.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f'<a href="{u_esc}" target="_blank">{label}</a>'


st.markdown("#### Content filters")

# =========================
# Row 1: Platform, Actor Type, Categories, Narrative
# =========================
r1 = st.columns(4)
with r1[0]:
    platform_sel = st.selectbox("Platform", ["Facebook", "YouTube", "TikTok"], index=0, key="content_platform")

# Load raw df AFTER platform selection
if platform_sel == "Facebook":
    df0 = fb_raw.copy()
    date_col = "Posted At"
elif platform_sel == "YouTube":
    df0 = yt_raw.copy()
    date_col = "publishedAt"
else:
    df0 = tt_raw.copy()
    date_col = "Date"

if df0 is None or df0.empty:
    st.info("No data for this platform.")
else:
    df0 = _ensure_total_engagement_for_block(platform_sel, df0)

    # Ensure Adversarial_bool exists
    if "Adversarial_bool" not in df0.columns:
        if "Adversarial" in df0.columns:
            df0["Adversarial_bool"] = _adv_bool_from_text(df0["Adversarial"])
        else:
            df0["Adversarial_bool"] = False

    # Parse date
    if date_col in df0.columns:
        df0["_date"] = pd.to_datetime(df0[date_col], errors="coerce")
    else:
        df0["_date"] = pd.NaT

    # Options
    actor_opts = sorted(
        df0.get("Actor Type", pd.Series([], dtype=str))
           .dropna().astype(str).str.strip().replace("", np.nan).dropna().unique().tolist()
    )
    cat_opts = _safe_multiselect_options(df0, "Categories")
    nar_opts = _safe_multiselect_options(df0, "Narrative")

    sev_opts = _safe_multiselect_options(df0, "Severity")
    mh_opts = _safe_multiselect_options(df0, "Misinformation/Hate")
    mh_opts = sorted({v for v in mh_opts if ("misinformation" in v.lower() or "hate" in v.lower())})

    # Remaining row 1 filters
    with r1[1]:
        actor_type_sel = st.multiselect("Actor Type", actor_opts, default=[], key="content_actor_type")
    with r1[2]:
        categories_sel = st.multiselect("Categories", cat_opts, default=[], key="content_categories")
    with r1[3]:
        narrative_sel = st.multiselect("Narrative", nar_opts, default=[], key="content_narrative")

    # =========================
    # Row 2: Adversarial, Misinformation/Hate, Severity, Date Range
    # =========================
    r2 = st.columns(4)
    with r2[0]:
        adversarial_sel = st.multiselect("Adversarial", ["Adversarial", "Non-Adversarial"], default=[], key="content_adv")
    with r2[1]:
        mh_sel = st.multiselect("Misinformation/Hate", mh_opts, default=[], key="content_mh")
    with r2[2]:
        severity_sel = st.multiselect("Severity", sev_opts, default=[], key="content_severity")
    with r2[3]:
        dmin, dmax = df0["_date"].min(), df0["_date"].max()
        has_dates = pd.notna(dmin) and pd.notna(dmax)
        if has_dates:
            date_range = st.date_input(
                f"{date_col} range",
                value=(dmin.date(), dmax.date()),
                key="content_date_range"
            )
        else:
            st.caption(f"{date_col} range")
            st.write("No dates detected.")
            date_range = None

    # =========================
    # Apply filters (local only)
    # =========================
    df = df0.copy()

    # adversarial filter
    if adversarial_sel:
        if "Adversarial" in adversarial_sel and "Non-Adversarial" not in adversarial_sel:
            df = df[df["Adversarial_bool"] == True].copy()
        elif "Non-Adversarial" in adversarial_sel and "Adversarial" not in adversarial_sel:
            df = df[df["Adversarial_bool"] == False].copy()

    # actor type filter
    if actor_type_sel and "Actor Type" in df.columns:
        df = df[df["Actor Type"].astype(str).isin(actor_type_sel)].copy()

    # multiselect cols
    df = _apply_list_filter(df, "Categories", categories_sel)
    df = _apply_list_filter(df, "Narrative", narrative_sel)
    df = _apply_list_filter(df, "Severity", severity_sel)
    df = _apply_list_filter(df, "Misinformation/Hate", mh_sel)

    # date range filter
    if date_range and "_date" in df.columns and len(date_range) == 2:
        start, end = date_range
        df = df[(df["_date"].dt.date >= start) & (df["_date"].dt.date <= end)].copy()

    # =========================
    # Platform-specific card fields
    # =========================
    if platform_sel == "Facebook":
        need = ["Author", "AuthorId", "Content", "Posted At", "Total Engagement",
                "Actor Type", "Adversarial", "Categories", "Narrative", "Misinformation/Hate"]
        for c in need:
            if c not in df.columns:
                df[c] = ""

        df["Misinformation/Hate"] = df["Misinformation/Hate"].apply(_mh_show_only_typed)

        df["_card_title"] = df["Author"].astype(str)
        df["_card_sub"] = "AuthorId: " + df["AuthorId"].astype(str)
        df["_card_body"] = df["Content"].astype(str)
        df["_card_url"] = df.apply(lambda r: _pick_first_url(r, ["Url", "URL", "Link", "link"]), axis=1)
        df["_card_date"] = df["Posted At"].astype(str)

    elif platform_sel == "YouTube":
        need = ["channelId", "channelTitle", "Title", "url", "publishedAt", "Total Engagement",
                "Actor Type", "Adversarial", "Categories", "Narrative", "Misinformation/Hate"]
        for c in need:
            if c not in df.columns:
                df[c] = ""

        df["Misinformation/Hate"] = df["Misinformation/Hate"].apply(_mh_show_only_typed)

        df["_card_title"] = df["channelTitle"].astype(str)
        df["_card_sub"] = "ChannelId: " + df["channelId"].astype(str)
        df["_card_body"] = df["Title"].astype(str)
        df["_card_url"] = df.apply(lambda r: _pick_first_url(r, ["url", "URL", "Link", "link"]), axis=1)
        df["_card_date"] = df["publishedAt"].astype(str)

    else:  # TikTok
        need = ["author_id", "author_username", "Snippet", "URL", "Date", "Total Engagement",
                "Actor Type", "Adversarial", "Categories", "Narrative", "Misinformation/Hate"]
        for c in need:
            if c not in df.columns:
                df[c] = ""

        df["Misinformation/Hate"] = df["Misinformation/Hate"].apply(_mh_show_only_typed)

        df["_card_title"] = df["author_username"].astype(str)
        df["_card_sub"] = "AuthorId: " + df["author_id"].astype(str)
        df["_card_body"] = df["Snippet"].astype(str)
        df["_card_url"] = df.apply(lambda r: _pick_first_url(r, ["URL", "Url", "url", "Link", "link"]), axis=1)
        df["_card_date"] = df["Date"].astype(str)

    df["Total Engagement"] = _to_num(df["Total Engagement"])

    # =========================
    # Row 3 controls: Sort, Cards/row, Cards/page, Page
    # =========================
    r3 = st.columns(4)
    with r3[0]:
        sort_by = st.selectbox("Sort by", ["Newest", "Highest engagement"], index=0, key="content_sort_by")
    with r3[1]:
        cols_per_row = st.selectbox("Cards/row", [2, 3, 4], index=2, key="content_cards_per_row")
    with r3[2]:
        page_size = st.selectbox("Cards/page", [20, 40, 80, 120], index=1, key="content_cards_page_size")

    # sort
    if sort_by == "Highest engagement":
        df = df.sort_values("Total Engagement", ascending=False)
    else:
        if "_date" in df.columns:
            df = df.sort_values("_date", ascending=False)

    total_pages = max(1, math.ceil(len(df) / page_size))
    with r3[3]:
        page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1, key="content_cards_page")

    start_i = (page - 1) * page_size
    end_i = start_i + page_size
    df_page = df.iloc[start_i:end_i].copy()

    st.caption(f"Posts shown: {len(df):,} • Page {page}/{total_pages}")

    # =========================
    # Card CSS (Engagement is visible)
    # =========================
    st.markdown("""
    <style>
    .post-card{
        border:1px solid #e6e6e6;
        border-radius:12px;
        padding:12px 12px;
        margin:8px 0;
        background:#fff;
        height: 100%;
    }
    .post-title{
        font-weight:700;
        font-size:14px;
        line-height:1.25;
        margin-bottom:4px;
    }
    .post-sub{
        font-size:12px;
        color:#666;
        margin-bottom:6px;
        white-space:nowrap;
        overflow:hidden;
        text-overflow:ellipsis;
    }
    .post-date{
        font-size:11px;
        color:#888;
        margin-bottom:8px;
    }
    .post-body{
        font-size:13px;
        color:#111;
        white-space:pre-wrap;
        overflow:hidden;
        display:-webkit-box;
        -webkit-line-clamp: 8;
        -webkit-box-orient: vertical;
        margin-bottom:10px;
        min-height: 110px;
    }
    .tag{
        display:inline-block;
        padding:3px 8px;
        margin:2px 4px 2px 0;
        border-radius:999px;
        font-size:11px;
        background:#f2f2f2;
    }
    .tag-adv{ background:#ffe3e3; }
    .tag-mh{ background:#e3f2ff; }
    .tag-eng{ background:#e8f5e9; font-weight:700; }
    .meta{
        font-size:11px;
        color:#666;
        margin-top:8px;
    }
    .meta a{ text-decoration: none; }
    </style>
    """, unsafe_allow_html=True)

    # =========================
    # Render cards
    # =========================
    rows = math.ceil(len(df_page) / cols_per_row)

    for r_i in range(rows):
        cols = st.columns(cols_per_row)
        for c_i in range(cols_per_row):
            idx = r_i * cols_per_row + c_i
            if idx >= len(df_page):
                break

            r = df_page.iloc[idx]

            title = _card_escape(r.get("_card_title", ""))
            sub = _card_escape(r.get("_card_sub", ""))
            date_txt = _card_escape(r.get("_card_date", ""))
            body = _card_escape(r.get("_card_body", ""))

            actor_type = _card_escape(r.get("Actor Type", ""))
            adv_txt = str(r.get("Adversarial", "")).strip().lower()
            is_adv = adv_txt in {"adversarial", "true", "yes", "y", "1", "adv"}

            cats = _card_escape(r.get("Categories", ""))
            nar = _card_escape(r.get("Narrative", ""))
            mh = _card_escape(r.get("Misinformation/Hate", ""))

            eng = int(_to_num(pd.Series([r.get("Total Engagement", 0)])).iloc[0])
            link_html = _maybe_link(r.get("_card_url", ""), "Open link")

            tags_html = f'<span class="tag tag-eng">Engagement: {eng}</span>'
            if actor_type:
                tags_html += f'<span class="tag">{actor_type}</span>'
            if is_adv:
                tags_html += f'<span class="tag tag-adv">Adversarial</span>'
            if mh:
                tags_html += f'<span class="tag tag-mh">{mh}</span>'

            meta_bits = []
            if cats:
                meta_bits.append(f"<b>Categories:</b> {cats}")
            if nar:
                meta_bits.append(f"<b>Narrative:</b> {nar}")
            if link_html:
                meta_bits.append(link_html)
            else:
                meta_bits.append("<span class='tag'>No link</span>")
            meta_html = "<br/>".join(meta_bits)

            with cols[c_i]:
                st.markdown(
                    f"""
                    <div class="post-card">
                        <div class="post-title">{title}</div>
                        <div class="post-sub">{sub}</div>
                        <div class="post-date">{date_txt}</div>
                        <div class="post-body">{body}</div>
                        <div>{tags_html}</div>
                        <div class="meta">{meta_html}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

st.divider()
