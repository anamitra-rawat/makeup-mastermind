#!/usr/bin/env python3
"""
safety_filter.py
================
Keyword-based ingredient safety filter for the Day 6 chatbot.

Public API
----------
filter_products(catalog_df, user_sensitivities,
                sensitivity_keywords=None,
                ingredient_col='ingredients_text')
    → pd.DataFrame
    Returns catalog_df copy with two new columns:
      - matched_sensitivities : list[str] — which sensitivities triggered
      - passes_filter         : bool      — True = no violations

load_precomputed_flags(flags_path)
    → pd.DataFrame
    Loads safety_filter_catalog.csv produced by Day 5 notebook.

apply_precomputed_flags(flags_df, user_sensitivities)
    → pd.Series[bool]
    Returns a boolean mask (True = passes) for fast query-time filtering.

Design notes
------------
- Deliberately scoped from NLP (status report commitment). Exact
  substring matching is the documented baseline approach.
- Keyword vocabulary is verbatim from notebook_3a_feature_engineering.py
  and the Day 2 training label generation. Do NOT change keywords without
  retraining the Day 3 GBM — the filter's logic is encoded as the top
  permutation-importance feature (score ≈ 0.94).
- Validated in Day 5 Part 3: recall = 1.0, FPR = 0 on 30-product sample,
  Cohen's kappa = 1.0 between annotators.
"""

import os
import pandas as pd
import numpy as np


# ── keyword vocabulary ────────────────────────────────────────────────────────
# Verbatim from training pipeline — DO NOT MODIFY without retraining the GBM.
SENSITIVITY_KEYWORDS = {
    'fragrance'  : ['fragrance', 'parfum'],
    'parabens'   : ['paraben'],
    'sulfates'   : ['sulfate', 'sls', 'sles'],
    'phthalates' : ['phthalate'],
    'alcohol'    : ['denatured alcohol', 'alcohol denat'],
    'silicones'  : ['dimethicone', 'cyclomethicone', 'cyclopentasiloxane'],
}

VALID_SENSITIVITIES = list(SENSITIVITY_KEYWORDS.keys())


# ── core API ──────────────────────────────────────────────────────────────────

def filter_products(catalog_df, user_sensitivities,
                    sensitivity_keywords=None,
                    ingredient_col='ingredients_text'):
    """
    Filter a product catalog based on a user's ingredient sensitivities.

    Parameters
    ----------
    catalog_df : pd.DataFrame
        The merged product catalog. Must contain `ingredient_col`.
    user_sensitivities : list[str]
        Sensitivities declared by this user, e.g. ['fragrance', 'silicones'].
        Must be a subset of SENSITIVITY_KEYWORDS keys:
            fragrance, parabens, sulfates, phthalates, alcohol, silicones
    sensitivity_keywords : dict, optional
        Keyword vocabulary override. Defaults to SENSITIVITY_KEYWORDS.
    ingredient_col : str, optional
        Column name containing ingredient text. Default: 'ingredients_text'.

    Returns
    -------
    pd.DataFrame
        Copy of catalog_df with two new columns appended:
          - matched_sensitivities : list[str] — sensitivity classes that matched
                                    (empty list if no violations found)
          - passes_filter         : bool — True if product passes the filter
                                    (no violations for this user's sensitivities)

    Raises
    ------
    ValueError : if user_sensitivities contains unknown sensitivity names
    KeyError   : if ingredient_col is not in catalog_df
    """
    if sensitivity_keywords is None:
        sensitivity_keywords = SENSITIVITY_KEYWORDS

    # validate
    unknown = set(user_sensitivities) - set(sensitivity_keywords.keys())
    if unknown:
        raise ValueError(
            f"Unknown sensitivity classes: {unknown}. "
            f"Valid options: {list(sensitivity_keywords.keys())}"
        )

    if ingredient_col not in catalog_df.columns:
        raise KeyError(
            f"Column '{ingredient_col}' not found. "
            f"Available: {list(catalog_df.columns)}"
        )

    df = catalog_df.copy()

    def _row_matches(ingredient_text):
        if pd.isna(ingredient_text):
            return []
        text_lower = str(ingredient_text).lower()
        return [
            sens for sens in user_sensitivities
            if any(kw.lower() in text_lower for kw in sensitivity_keywords[sens])
        ]

    df['matched_sensitivities'] = df[ingredient_col].apply(_row_matches)
    df['passes_filter']         = df['matched_sensitivities'].apply(lambda x: len(x) == 0)

    return df


def load_precomputed_flags(flags_path):
    """
    Load the pre-computed safety flags CSV produced by the Day 5 notebook.

    Parameters
    ----------
    flags_path : str
        Path to safety_filter_catalog.csv

    Returns
    -------
    pd.DataFrame
        Columns: source, source_id, flagged_fragrance, flagged_parabens,
                 flagged_sulfates, flagged_phthalates, flagged_alcohol,
                 flagged_silicones
    """
    if not os.path.exists(flags_path):
        raise FileNotFoundError(
            f"Pre-computed flags not found at: {flags_path}\n"
            f"Run the Day 5 Part 5 notebook to generate it."
        )
    df = pd.read_csv(flags_path)
    expected = [f'flagged_{s}' for s in SENSITIVITY_KEYWORDS]
    missing  = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Missing flag columns in {flags_path}: {missing}")
    return df


def apply_precomputed_flags(flags_df, user_sensitivities):
    """
    Apply pre-computed flags for fast query-time filtering.

    Parameters
    ----------
    flags_df : pd.DataFrame
        Output of load_precomputed_flags().
    user_sensitivities : list[str]
        Sensitivities for this query, e.g. ['fragrance', 'silicones'].

    Returns
    -------
    pd.Series[bool]
        Boolean mask aligned with flags_df index.
        True  = product passes (no violations for this user)
        False = product violates at least one of the user's sensitivities

    Example
    -------
    flags_df = load_precomputed_flags('safety_filter_catalog.csv')
    mask     = apply_precomputed_flags(flags_df, user['sensitivities'])
    safe_catalog = catalog_df[mask.values]
    """
    unknown = set(user_sensitivities) - set(SENSITIVITY_KEYWORDS.keys())
    if unknown:
        raise ValueError(f"Unknown sensitivity classes: {unknown}")

    if len(user_sensitivities) == 0:
        return pd.Series([True] * len(flags_df), index=flags_df.index)

    flag_cols = [f'flagged_{s}' for s in user_sensitivities]
    missing   = [c for c in flag_cols if c not in flags_df.columns]
    if missing:
        raise KeyError(f"Missing columns in flags_df: {missing}")

    # passes if NONE of the user's sensitivity columns are True
    violates = flags_df[flag_cols].any(axis=1)
    return ~violates


# ── convenience ───────────────────────────────────────────────────────────────

def summarise_filter_coverage(flags_df):
    """
    Print a summary of flag prevalence across the catalog.
    Useful for the chatbot startup log.
    """
    n = len(flags_df)
    print(f"Safety filter coverage summary ({n:,} products):")
    for sens in SENSITIVITY_KEYWORDS:
        col = f'flagged_{sens}'
        if col in flags_df.columns:
            k   = flags_df[col].sum()
            pct = 100 * k / n
            print(f"  {sens:<12s}: {k:>5,} flagged  ({pct:5.1f}%)")
    all_clear = (~flags_df[[f'flagged_{s}' for s in SENSITIVITY_KEYWORDS]].any(axis=1)).sum()
    print(f"  {'(all-clear)':<12s}: {all_clear:>5,} products pass every filter  "
          f"({100*all_clear/n:.1f}%)")
