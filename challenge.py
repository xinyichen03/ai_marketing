# %%
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import math
import re

# Set up logging
logger = logging.getLogger(__name__)

# ------------------------------
# Config
# ------------------------------
DATA_DIR = Path("./challenge")  # Directory containing parquet files
VALUE_COL = "attribute_value_local"  # Column to pivot on
SPARSE_THRESHOLD = 0.60  # Thredshold for dropping columns with missing values over 60%
SECTION_NAME_MAPPING = {'de': 'Allgemein', 
                        'en': 'General'}  # Section name by local language

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
    # Collect every parquet file in the target folder (non‑recursive)
    files = sorted(dir_path.glob("*.parquet"))
    # Log how many files we are about to stitch together
    logger.info("Loading %d parquet files from %s", len(files), dir_path)
    # Read each file and stack them vertically into one DataFrame
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True, copy=False)

# Pivot attributes to wide format
def pivot_attributes(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    # Keep only the columns we need for widening, drop duplicates so pivot is clean
    tmp = df[['item_code', 'attribute_key', value_col]].drop_duplicates()
    attr_wide = (
        tmp.pivot_table(index='item_code',
                        columns='attribute_key',
                        values=value_col,
                        aggfunc='first')
           .rename_axis(None, axis=1)
    )
    # Keep a single product_name per item_code (there can be repeats in raw data)
    names = (
        df[['item_code', 'product_name']]
          .drop_duplicates()
          .groupby('item_code')['product_name']
          .first()
          .to_frame()
    )
    # Return item_code + product_name + one column per attribute_key
    return names.join(attr_wide).reset_index()

# Drop columns with high NaN ratio, except protected columns
def drop_sparse_columns(df: pd.DataFrame, threshold: float, protected=None):
    protected = protected or set()
    # Share of missing values per column
    nan_ratio = df.isna().mean()
    # Decide which columns to drop (too many gaps, not in protected set)
    drop_cols = [c for c, r in nan_ratio.items() if r > threshold and c not in protected]
    logger.info("       Dropping %d sparse columns (> %.0f%% NaN)", len(drop_cols), threshold * 100)
    return df.drop(columns=drop_cols), drop_cols

# Calculate non-null metrics and return a summary
def compute_non_null_metrics(df: pd.DataFrame,
                             id_cols=('item_code', 'product_name')) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add per-row and per-attribute non-null metrics; return (df, attribute_summary)."""
    base_attr_cols = [c for c in df.columns if c not in id_cols]
    # Metrics per row
    df['non_null_attribute_count'] = df[base_attr_cols].notna().sum(axis=1)
    df['non_null_attribute_pct'] = (
        df['non_null_attribute_count'] / len(base_attr_cols) * 100
    ).round(2)
    non_null_counts = df[base_attr_cols].notna().sum()
    total_rows = len(df)
    attribute_non_null_summary = (
        pd.DataFrame({
            'non_null_count': non_null_counts,
            'non_null_pct': (non_null_counts / total_rows * 100).round(2)
        })
        .sort_values('non_null_count', ascending=False)
    )
    logger.info("       Attribute non-null summary:\n%s",
                attribute_non_null_summary.to_string())
    return df, pd.DataFrame()

# Scale and clean dimension columns
def scale_series(series, rule):
    # Empty guard: if no series passed, bail out
    if series is None:
        return None, None
    # Convert to numeric so we can compare and scale; non‑numeric becomes NaN
    ser = pd.to_numeric(series, errors='coerce')
    # Values at or above the threshold are candidates for scaling (e.g. mm -> cm)
    mask = ser >= rule['threshold']
    # Apply divisor only to values flagged by the mask
    if rule.get('divisor'):
        ser.loc[mask] = ser.loc[mask] / rule['divisor']
    rmode = rule.get('round')
    # Optional rounding mode for scaled values
    if rmode == 'ceil':
        ser.loc[mask] = ser.loc[mask].apply(math.ceil)
    elif rmode == 'round':
        ser.loc[mask] = ser.loc[mask].round()
    return ser, mask

# Clean dimension columns in the DataFrame
def clean_dimensions(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    for dim, rule in rules.items():
        # Skip if this dimension is not present in the wide table
        if dim not in df.columns:
            continue
        # Scale raw numeric values according to rule (threshold/divisor/round)
        cleaned_numeric, mask_scaled = scale_series(df[dim], rule)
        if cleaned_numeric is None:
            continue
        num_col = f"{dim}_clean_num"
        unit_col = f"{dim}_clean"
        # Store pure numeric value
        df[num_col] = cleaned_numeric
        # Store display value with unit (e.g. 50 cm). Keep original NaN as NaN.
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
        # Prefer already cleaned display, then cleaned numeric, then raw column
        for src in (clean_col, numeric_col, dim):
            if src in df.columns:
                value_col = src
                break
        else:
            continue
        # Convert to numeric for sorting; ignore rows that still fail conversion
        numeric_sort = pd.to_numeric(df[value_col], errors='coerce')
        top = (
            df.loc[numeric_sort.notna(), ['product_name', value_col]]
              .assign(sort_key=numeric_sort[numeric_sort.notna()])
              .sort_values('sort_key', ascending=False)
              .head(n)
              .drop(columns='sort_key')
              .rename(columns={value_col: f"{dim}_value"})
        )
        results[dim] = top
        if not top.empty:
            logger.info("       Top %d %s values (source=%s):\n%s", len(top), dim, value_col, top.to_string(index=False))
    return results


# Analyze color-related columns and log summary + sample rows
def analyze_color_fields(df: pd.DataFrame, sample_size: int = 10):
    """Find rows where a base 'color' or 'colorSubcolor' values exist but 'colorDetail' is empty.
    This helps judge if we could expand colorDetail. 
    """
    # colorDetail is empty
    mask_cd_nan = df['colorDetail'].isna()
    # At least one alternative color source is present
    mask_color_or_sub = df['color'].notna() | df['colorSubcolor'].notna()

    total = len(df)
    cd_nan = int(mask_cd_nan.sum())
    # Missing detail but some color info available
    cd_nan_with_color = int((mask_cd_nan & mask_color_or_sub).sum())
    # Missing detail and also no fallback color info
    cd_nan_without_color = int((mask_cd_nan & ~mask_color_or_sub).sum())

    logger.info(
        "       Color analysis: total=%d colorDetail_NaN=%d with_alt_color=%d no_color_info=%d",
        total, cd_nan, cd_nan_with_color, cd_nan_without_color
    )

    # Show a few rows where we could potentially enrich colorDetail
    sample = df.loc[mask_cd_nan & mask_color_or_sub,
                    ['item_code', 'product_name', 'color', 'colorDetail', 'colorSubcolor']].head(sample_size)
    if not sample.empty:
        logger.info("       Sample rows (colorDetail NaN but color/colorSubcolor present):\n%s",
                    sample.to_string(index=False))
    else:
        logger.info("       No rows where colorDetail is NaN but color/colorSubcolor present")

    return {
        'total': total,
        'colorDetail_nan': cd_nan,
        'colorDetail_nan_with_color_or_sub': cd_nan_with_color,
        'colorDetail_nan_without_color_or_sub': cd_nan_without_color,
        'sample': sample
    }


# Analyze material/materialDetail coverage analogous to color analysis
def analyze_material_fields(df: pd.DataFrame, sample_size: int = 10):
    """Find rows where a base 'material' value exists but 'materialDetail' is empty.
    This helps judge if we could expand materialDetail and how that impacts data quality.
    """
    # All required columns ('material', 'materialDetail') are guaranteed present
    mask_material_present = df['material'].notna()
    mask_detail_missing = df['materialDetail'].isna()
    # Rows with material but no materialDetail
    mask_material_no_detail = mask_material_present & mask_detail_missing

    # Columns we will inspect
    subset = df.loc[mask_material_no_detail, ['item_code', 'product_name', 'material', 'materialDetail']]
    count_missing = len(subset)
    total = len(df)
    pct = (count_missing / total * 100) if total else 0
    logger.info("       Material analysis: material_present_no_detail=%d (%.2f%% of products)", count_missing, pct)

    # Small sample for inspection in logs
    sample = subset.head(sample_size)
    if not sample.empty:
        logger.info("       Sample rows (material present, materialDetail missing):\n%s", sample.to_string(index=False))
    else:
        logger.info("       No rows with material present and materialDetail missing")

    return {
        'total': total,
        'material_present_no_detail_count': count_missing,
        'material_present_no_detail_pct': pct,
        'sample': sample
    }


# Analyze dimensionDetail presence when specific dimension columns are missing
def analyze_dimension_detail_fields(df: pd.DataFrame, sample_size: int = 20):
    """Analyze products that have 'dimensionDetail' text but lack ALL individual
    numeric dimension columns (depth, height, weight, width).
    """
    required_col = 'dimensionDetail'
    dim_cols = ['depth', 'height', 'weight', 'width']

    # Rows where dimensionDetail present AND all individual dims are NaN
    mask_only_detail = df[required_col].notna() & df[dim_cols].isna().all(axis=1)

    # Columns we will output / inspect
    cols_out = ['item_code', 'product_name', required_col] + dim_cols
    subset = df.loc[mask_only_detail, cols_out].copy()

    # Summary stats
    n = len(subset)
    total = len(df)
    pct = (n / total * 100) if total else 0
    logger.info("       Dimension detail analysis: only_dimensionDetail=%d (%.2f%% of products)", n, pct)

    # Small sample for inspection in logs
    sample = subset.head(sample_size)
    if not sample.empty:
        logger.info("       Sample rows (only dimensionDetail, no raw dims):\n%s", sample.to_string(index=False))
    else:
        logger.info("       No rows with only dimensionDetail present")

    return {
        'total': total,
        'only_dimensionDetail_count': n,
        'only_dimensionDetail_pct': pct,
        'sample': sample,
        'dimension_columns_present': dim_cols
    }


# Format product detail string
def format_product_detail(raw_text: str, section_name: str, attribute_name: str) -> str:
    # Pattern: capture simple key:value pairs. Stops before next key or end of string.
    pattern = r'(\w+):\s*([^:]+?)(?=\s+\w+:|$)'
    matches = list(re.finditer(pattern, raw_text))
    out = []
    if matches:
        # Anything before first key:value pair is treated as a free text intro
        first_start = matches[0].start()
        leading = raw_text[:first_start].strip()
        if leading:
            out.append(f"{section_name}:{attribute_name}:{leading}")
        for m in matches:
            k, v = m.group(1), m.group(2).strip()
            # Quote values that contain commas so they remain grouped when later parsed
            if ',' in v:
                v = f'"{v}"'
            out.append(f"{attribute_name}:{k}:{v}")
    else:
        # No structured pairs, keep whole string as a single line
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
            # Column might be missing if dropped earlier
            if col not in df.columns:
                continue
            val = row.get(col)
            # Skip blanks / NaN
            if pd.isna(val):
                continue
            attr_label = ATTRIBUTE_NAME_MAPPING.get(col, col)
            # Build one or more tokens from the raw cell text
            parts.append(format_product_detail(str(val).strip(), section, attr_label))
        # Join all tokens for this row; return None if empty so downstream can drop
        return ",".join(parts) if parts else None
    return df.apply(row_builder, axis=1)

# ------------------------------
# Workflow
# ------------------------------
if __name__ == "__main__":
    # Step 1: Load data
    logger.info("Step 1: Load data")
    df = load_parquet_dir(DATA_DIR)
    logger.info("       Rows loaded: %d", len(df))

    # Step 2: Exploration and Analysis
    logger.info("Step 2: Exploration and Analysis")
    df_wide = pivot_attributes(df, VALUE_COL)
    logger.info("       Wide shape: %s", df_wide.shape)

    logger.info("       Dropping sparse columns")
    df_wide, dropped_cols = drop_sparse_columns(df_wide, SPARSE_THRESHOLD, protected={'item_code','product_name'})
    logger.info("       After drop shape: %s", df_wide.shape)

    logger.info("       Computing non-null attribute metrics")
    df_wide, attribute_non_null_summary = compute_non_null_metrics(df_wide)

    logger.info("       --- Analyzing color fields ---")
    color_analysis = analyze_color_fields(df_wide)

    logger.info("       --- Analyzing dimension detail fields ---")
    dimension_detail_analysis = analyze_dimension_detail_fields(df_wide)

    logger.info("       --- Analyzing material fields ---")
    material_analysis = analyze_material_fields(df_wide)

    # Step 3: Data Cleaning
    logger.info("Step 3: Dimension cleanup")
    extremes_summary = log_top_extremes(df_wide)
    logger.info("       Cleaning dimensions:")
    df_wide = clean_dimensions(df_wide, DIMENSION_RULES)

    # Step 4: Generate Product Detail
    logger.info("Step 4: Build product_detail")
    selected_cols = [c for c in selected_cols if c in df_wide.columns]
    df_selected = df_wide[['item_code'] + selected_cols].copy()
    df_selected['product_detail'] = build_product_detail(df_selected, selected_cols, lang=LANG)

    # Step 5: Export
    logger.info("Step 5: Export")
    export_df = df_selected[['item_code','product_detail']].rename(columns={'item_code':'sku'})
    export_path = Path("product_detail.csv")
    export_df.to_csv(export_path, index=False)
    logger.info("       Exported %d rows to %s", len(export_df), export_path)