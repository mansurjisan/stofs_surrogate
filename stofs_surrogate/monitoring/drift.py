"""Input-forcing and prediction drift reporting.

Always writes a self-contained drift summary (JSON + HTML) computed with a two-sample
Kolmogorov-Smirnov test and Population Stability Index, so the pipeline runs in CI without
heavy dependencies. When Evidently is installed (``pip install '.[monitoring]'``) an
additional, richer Evidently HTML report is generated as well.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


def _psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two samples (0 = identical)."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    ref_frac = np.clip(np.histogram(reference, bins=edges)[0] / max(len(reference), 1), 1e-6, None)
    cur_frac = np.clip(np.histogram(current, bins=edges)[0] / max(len(current), 1), 1e-6, None)
    return float(np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac)))


def compute_drift(reference: pd.DataFrame, current: pd.DataFrame,
                  columns: Optional[List[str]] = None) -> Dict:
    """Per-column KS test + PSI between a reference and a current dataframe.

    A column is flagged as drifted when the KS p-value < 0.05 or PSI > 0.2.
    """
    if columns is None:
        columns = [c for c in reference.columns
                   if np.issubdtype(reference[c].dtype, np.number)]
    results = {}
    for col in columns:
        ref = reference[col].dropna().to_numpy()
        cur = current[col].dropna().to_numpy()
        if len(ref) < 2 or len(cur) < 2:
            continue
        ks = stats.ks_2samp(ref, cur)
        psi = _psi(ref, cur)
        results[col] = {
            "ks_stat": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
            "psi": psi,
            "drifted": bool(ks.pvalue < 0.05 or psi > 0.2),
        }
    n_drifted = sum(1 for r in results.values() if r["drifted"])
    return {"columns": results, "n_drifted": n_drifted, "n_columns": len(results),
            "dataset_drift": n_drifted > 0}


def drift_report(reference: pd.DataFrame, current: pd.DataFrame,
                 output_dir: str = "outputs/monitoring",
                 columns: Optional[List[str]] = None, use_evidently: bool = True) -> Dict:
    """Compute drift and write ``drift_report.json`` + ``drift_report.html`` to ``output_dir``.

    When ``use_evidently`` and Evidently is installed, also writes ``evidently_drift.html``.
    Returns the drift summary dict (with ``evidently_html`` set if generated).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = compute_drift(reference, current, columns)
    (out / "drift_report.json").write_text(json.dumps(summary, indent=2))
    _write_html(summary, out / "drift_report.html")
    if use_evidently:
        evidently_path = _maybe_evidently(reference, current, out / "evidently_drift.html")
        if evidently_path:
            summary["evidently_html"] = evidently_path
    return summary


def _write_html(summary: Dict, path: Path) -> None:
    rows = "".join(
        f"<tr><td>{col}</td><td>{d['ks_stat']:.3f}</td><td>{d['ks_pvalue']:.3g}</td>"
        f"<td>{d['psi']:.3f}</td><td>{'YES' if d['drifted'] else 'no'}</td></tr>"
        for col, d in summary["columns"].items()
    )
    path.write_text(
        "<html><body><h2>STOFS-GNN drift report</h2>"
        f"<p>Dataset drift: <b>{summary['dataset_drift']}</b> "
        f"({summary['n_drifted']}/{summary['n_columns']} columns)</p>"
        "<table border='1' cellpadding='4'><tr><th>column</th><th>KS</th><th>p-value</th>"
        f"<th>PSI</th><th>drifted</th></tr>{rows}</table></body></html>"
    )


def _maybe_evidently(reference: pd.DataFrame, current: pd.DataFrame,
                     out_path: Path) -> Optional[str]:
    """Best-effort Evidently HTML report (optional dependency; API varies by version)."""
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
    except ImportError:
        logger.info("Evidently not installed; skipping the Evidently report "
                    "(pip install '.[monitoring]' to enable).")
        return None
    try:
        snapshot = Report(metrics=[DataDriftPreset()]).run(
            current_data=current, reference_data=reference)
        snapshot.save_html(str(out_path))
        return str(out_path)
    except Exception as exc:  # version differences -> non-fatal; built-in report still written
        logger.warning("Evidently report failed (%s); built-in report was still written.", exc)
        return None
