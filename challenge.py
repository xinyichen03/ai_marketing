# %%
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import math
import re

# Set up logging
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )
logger = logging.getLogger(__name__)

# ------------------------------
# Config
# ------------------------------
DATA_DIR = Path("./challenge")  # Directory containing parquet files
VALUE_COL = "attribute_value_local"  # Column to pivot on
SPARSE_THRESHOLD = 0.60  # Thredshold for dropping columns with missing values over 60%
SECTION_NAME_MAPPING = {'de': 'Allgemein', 'en': 'General'}  # Section name by local language

# Selected attribute_keys to include in product_detail
selected_cols = [
    'materialDetail','colorDetail',
    'weight_clean_num','depth_clean_num','height_clean_num','width_clean_num',
    'styleFilter','shippingCondition','guarantee'
]

# Mapping of attribute keys to Germannames 
ATTRIBUTE_NAME_MAPPING = {
    'shippingCondition': 'Lieferzustand',
    'styleFilter': 'Stil',
    'guarantee': 'Garantie (Jahre)',
    'weight_clean_num': 'Gewicht',
    'depth_clean_num': 'Tiefe',
    'height_clean_num': 'Höhe',
    'width_clean_num': 'Breite',
    'materialDetail': 'Material',
    'colorDetail': 'Farbe'
}

# local language
LANG = 'de'

# Rules for cleaning/scaling dimension columns
DIMENSION_RULES = {
    'weight': {'threshold': 600, 'divisor': 1000, 'round': 'ceil', 'unit': 'kg'},
    'height': {'threshold': 300, 'divisor': 10,   'round': 'ceil', 'unit': 'cm'},
    'width':  {'threshold': 500, 'divisor': 10,   'round': 'ceil', 'unit': 'cm'},
    'depth':  {'threshold': 200, 'divisor': 10,   'round': 'ceil', 'unit': 'cm'},
}


# ------------------------------
# Helpers
# ------------------------------
# Load all parquet files in a directory
def load_parquet_dir(dir_path: Path) -> pd.DataFrame:
    files = sorted(dir_path.glob("*.parquet"))
    logger.info("Loading %d parquet files from %s", len(files), dir_path)
    if not files:
        raise FileNotFoundError("No parquet files found.")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True, copy=False)

# Pivot attributes to wide format
def pivot_attributes(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    tmp = df[['item_code', 'attribute_key', value_col]].drop_duplicates()
    attr_wide = (
        tmp.pivot_table(index='item_code',
                        columns='attribute_key',
                        values=value_col,
                        aggfunc='first')
           .rename_axis(None, axis=1)
    )
    names = (
        df[['item_code', 'product_name']]
          .drop_duplicates()
          .groupby('item_code')['product_name']
          .first()
          .to_frame()
    )
    return names.join(attr_wide).reset_index()

# Drop columns with high NaN ratio, except protected columns
def drop_sparse_columns(df: pd.DataFrame, threshold: float, protected=None):
    protected = protected or set()
    nan_ratio = df.isna().mean()
    drop_cols = [c for c, r in nan_ratio.items() if r > threshold and c not in protected]
    logger.info("Dropping %d sparse columns (> %.0f%% NaN)", len(drop_cols), threshold * 100)
    return df.drop(columns=drop_cols), drop_cols

# Calculate non-null metrics and return a summary
def compute_non_null_metrics(df: pd.DataFrame,
                             id_cols=('item_code', 'product_name')) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add per-row and per-attribute non-null metrics; return (df, attribute_summary)."""
    base_attr_cols = [c for c in df.columns if c not in id_cols]
    if not base_attr_cols:
        logger.warning("No attribute columns found for non-null metric calculation.")
        return df, pd.DataFrame()
    df['non_null_attribute_count'] = df[base_attr_cols].notna().sum(axis=1)
    df['non_null_attribute_pct'] = (
        df['non_null_attribute_count'] / len(base_attr_cols) * 100
    ).round(2)
    _non_null_counts = df[base_attr_cols].notna().sum()
    _total_rows = len(df)
    attribute_non_null_summary = (
        pd.DataFrame({
            'non_null_count': _non_null_counts,
            'non_null_pct': (_non_null_counts / _total_rows * 100).round(2)
        })
        .sort_values('non_null_count', ascending=False)
    )
    logger.info("       Attribute non-null summary (top 10 shown):\n%s",
                attribute_non_null_summary.head(10).to_string())
    return df, attribute_non_null_summary

# Scale and clean dimension columns
def _scale_series(s, rule):
    if s is None:
        return None, None
    ser = pd.to_numeric(s, errors='coerce')
    mask = ser >= rule['threshold']
    if rule.get('divisor'):
        ser.loc[mask] = ser.loc[mask] / rule['divisor']
    rmode = rule.get('round')
    if rmode == 'ceil':
        ser.loc[mask] = ser.loc[mask].apply(math.ceil)
    elif rmode == 'round':
        ser.loc[mask] = ser.loc[mask].round()
    return ser, mask

# Clean dimension columns in the DataFrame
def clean_dimensions(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    for dim, rule in rules.items():
        if dim not in df.columns:
            continue
        cleaned_numeric, mask_scaled = _scale_series(df[dim], rule)
        if cleaned_numeric is None:
            continue
        num_col = f"{dim}_clean_num"
        unit_col = f"{dim}_clean"
        df[num_col] = cleaned_numeric
        df[unit_col] = df[num_col].apply(
            lambda x: f"{int(x) if float(x).is_integer() else x} {rule['unit']}" if pd.notna(x) else x
        )
        logger.info("       %s: scaled %d values (>= %s -> /%s, round=%s, unit=%s)",
                    dim,
                    int(mask_scaled.sum()),
                    rule['threshold'],
                    rule.get('divisor'),
                    rule.get('round'),
                    rule['unit'])
    return df

# Log top-10 extremes for specified dimensions
def log_top_extremes(df, dims=("weight","height","width","depth"), n=10):
    results = {}
    for dim in dims:
        if dim not in df.columns:
            continue
        clean_col = f"{dim}_clean"
        numeric_col = f"{dim}_clean_num"
        # Determine source column preference
        for src in (clean_col, numeric_col, dim):
            if src in df.columns:
                value_col = src
                break
        else:
            continue
        numeric_sort = pd.to_numeric(df[value_col], errors='coerce')
        top = (
            df.loc[numeric_sort.notna(), ['product_name', value_col]]
              .assign(_sort_key=numeric_sort[numeric_sort.notna()])
              .sort_values('_sort_key', ascending=False)
              .head(n)
              .drop(columns='_sort_key')
              .rename(columns={value_col: f"{dim}_value"})
        )
        results[dim] = top
        if not top.empty:
            logger.info("       Top %d %s values (source=%s):\n%s", len(top), dim, value_col, top.to_string(index=False))
    return results


# Format product detail string
def format_product_detail(raw_text: str, section_name: str, attribute_name: str) -> str:
    pattern = r'(\w+):\s*([^:]+?)(?=\s+\w+:|$)'
    matches = list(re.finditer(pattern, raw_text))
    out = []
    if matches:
        first_start = matches[0].start()
        leading = raw_text[:first_start].strip()
        if leading:
            out.append(f"{section_name}:{attribute_name}:{leading}")
        for m in matches:
            k, v = m.group(1), m.group(2).strip()
            if ',' in v:
                v = f'"{v}"'
            out.append(f"{attribute_name}:{k}:{v}")
    else:
        out.append(f"{section_name}:{attribute_name}:{raw_text.strip()}")
    return ",".join(out)

# Build product_detail column
def build_product_detail(df: pd.DataFrame,
                         selected_cols,
                         lang='de') -> pd.Series:
    section = SECTION_NAME_MAPPING[lang]
    def row_builder(row):
        parts = []
        for col in selected_cols:
            if col not in df.columns:
                continue
            val = row.get(col)
            if pd.isna(val):
                continue
            attr_label = ATTRIBUTE_NAME_MAPPING.get(col, col)
            parts.append(format_product_detail(str(val).strip(), section, attr_label))
        return ",".join(parts) if parts else None
    return df.apply(row_builder, axis=1)

# ------------------------------
# Pipeline
# ------------------------------
logger.info("Step 1: Load data")
df = load_parquet_dir(DATA_DIR)
logger.info("       Rows loaded: %d", len(df))

logger.info("Step 2: Pivot attributes")
df_wide = pivot_attributes(df, VALUE_COL)
logger.info("       Wide shape: %s", df_wide.shape)

logger.info("       Dropping sparse columns")
df_wide, dropped_cols = drop_sparse_columns(df_wide, SPARSE_THRESHOLD, protected={'item_code','product_name'})
logger.info("       After drop shape: %s", df_wide.shape)


logger.info("       Computing non-null attribute metrics")
df_wide, attribute_non_null_summary = compute_non_null_metrics(df_wide)

logger.info("Step 3: Dimension cleanup")
extremes_summary = log_top_extremes(df_wide)
logger.info("       Cleaning dimensions:")
df_wide = clean_dimensions(df_wide, DIMENSION_RULES)

logger.info("Step 4: Build product_detail")
selected_cols = [c for c in selected_cols if c in df_wide.columns]
df_selected = df_wide[['item_code'] + selected_cols].copy()
df_selected['product_detail'] = build_product_detail(df_selected, selected_cols, lang=LANG)

logger.info("Step 5: Export")
export_df = df_selected[['item_code','product_detail']].rename(columns={'item_code':'sku'})
export_path = Path("product_detail.csv")
export_df.to_csv(export_path, index=False)
logger.info("   Exported %d rows to %s", len(export_df), export_path)
# %%