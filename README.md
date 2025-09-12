# Product Detail Feed Builder

Minimal pipeline to turn raw product attribute parquet parts into a Google Shopping‑ready `product_detail` field.

## What It Does
1. Load all parquet files in `./challenge/`.
2. Pivot attribute rows to one wide row per product.
3. Drop columns with >60% missing values (keep IDs).
4. Normalize dimensions (heuristic unit fixes) using simple thresholds.
5. Parse selected marketing attributes into tokens.
6. Write `product_detail.csv` with columns: `sku,product_detail`.

## Quick Start
```bash
pip install -r requirements.txt
python challenge.py
```
Output: `product_detail.csv` appears in repo root.

## Input (expected columns)
`item_code`, `product_name`, `attribute_key`, `attribute_value_local` plus various attribute keys (e.g. material, color, dimensions).

## Output Format
Each row: `sku,product_detail` where `product_detail` is a comma‑separated list of tokens:
`Allgemein:Material:Korpus:Massivholz,Material:Bezug:"100% Baumwolle",Allgemein:Stil:Modern`

Patterns:
- Section/Attribute: `SectionName:AttributeName:Value`
- Sub‑pairs inside free‑text fields (`materialDetail`, `colorDetail`): `AttributeName:SubKey:SubValue`

## Selected Attributes
`materialDetail, colorDetail, weight, height, width, depth, styleFilter, shippingCondition, guarantee`

## Dimension Rules (scale if value >= threshold)
Weight ≥600 → /1000 (g→kg) ceil    → kg
Height ≥300 → /10   (mm→cm) ceil   → cm
Width  ≥500 → /10   (mm→cm) ceil   → cm
Depth  ≥200 → /10   (mm→cm) ceil   → cm

## Key Assumptions
- Thresholds are heuristic (no statistical detection yet).
- Language fixed to German (`de`).
- Columns >60% NaN dropped globally (simple relevance proxy).
- Regex for sub‑pairs: `key: value` sequences only.

## Files
`challenge.py` main pipeline
`new.py` exploratory earlier version
`challenge/` parquet inputs
`product_detail.csv` generated output
`requirements.txt` dependencies
`non-technical_summary.pdf` plain‑language explanation

## Future (short list)
- Add CLI args (input dir, language, thresholds)
- Unit tests for parsing & scaling
- Optional schema validation

For extended rationale see non-technical_summary.pdf
