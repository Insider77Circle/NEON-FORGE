"""Visualization hooks.

Two renderers: an ASCII console board (safe on cp1252 terminals and when stdout
is redirected) and a self-contained neon HTML artifact with no external assets.
"""

from __future__ import annotations

from typing import Dict, List, Optional


def _bar(value: float, width: int = 20) -> str:
    value = max(0.0, min(1.0, value))
    filled = int(round(value * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def _sparkline(values: List[float]) -> str:
    if not values:
        return ""
    levels = " .:-=+*#@"
    low, high = min(values), max(values)
    span = high - low
    if span < 1e-9:
        return "=" * len(values)
    out = []
    for value in values:
        index = int(round((value - low) / span * (len(levels) - 1)))
        out.append(levels[index])
    return "".join(out)


class Dashboard:
    """Renders pipeline state for humans."""

    def render(
        self,
        metrics: Dict[str, object],
        alerts: Optional[List[object]] = None,
        title: str = "NEON FORGE",
    ) -> str:
        refusal = metrics.get("refusal_rate", {}) or {}
        preservation = float(metrics.get("capability_preservation", 0.0) or 0.0)
        drift = float(metrics.get("drift", 0.0) or 0.0)
        before = float(refusal.get("before", 0.0) or 0.0)  # type: ignore[union-attr]
        after = float(refusal.get("after", 0.0) or 0.0)  # type: ignore[union-attr]

        lines = [
            "=" * 62,
            f" {title} :: MODEL OUTPUT MONITOR",
            "=" * 62,
            f" refusal rate   {_bar(after)}  {before:.3f} -> {after:.3f}",
            f" capability     {_bar(preservation)}  {preservation:.3f} retained",
            f" drift          {_bar(1.0 - drift)}  {drift:.3f} cosine distance",
            "-" * 62,
        ]

        per_category = metrics.get("per_category", {}) or {}
        series: List[float] = []
        for category, stats in per_category.items():  # type: ignore[union-attr]
            if "error" in stats:
                lines.append(f" {category:<16} ERROR: {stats['error']}")
                continue
            cat_before = float(stats.get("refusal_rate_before", 0.0))
            cat_after = float(stats.get("refusal_rate_after", 0.0))
            series.append(cat_after)
            lines.append(
                f" {category:<16} refusal {cat_before:.2f}->{cat_after:.2f}   "
                f"drift {float(stats.get('drift', 0.0)):.3f}   n={int(stats.get('probes', 0))}"
            )

        if series:
            lines.append(f" refusal trend   {_sparkline(series)}")
        lines.append("-" * 62)

        if alerts:
            lines.append(f" alerts ({len(alerts)})")
            for alert in alerts:
                entry = alert.as_dict() if hasattr(alert, "as_dict") else dict(alert)  # type: ignore[call-overload]
                lines.append(
                    f"   [{str(entry.get('severity', '?')).upper():<8}] "
                    f"{entry.get('rule')}: {entry.get('message')}"
                )
        else:
            lines.append(" alerts (0) - all gates passed")

        lines.append("=" * 62)
        return "\n".join(lines)

    def write_html(
        self,
        path: str,
        metrics: Dict[str, object],
        alerts: Optional[List[object]] = None,
        session: Optional[str] = None,
    ) -> str:
        """Write a standalone neon board. No external CSS, fonts, or scripts."""
        refusal = metrics.get("refusal_rate", {}) or {}
        preservation = float(metrics.get("capability_preservation", 0.0) or 0.0)
        drift = float(metrics.get("drift", 0.0) or 0.0)
        before = float(refusal.get("before", 0.0) or 0.0)  # type: ignore[union-attr]
        after = float(refusal.get("after", 0.0) or 0.0)  # type: ignore[union-attr]

        rows = []
        for category, stats in (metrics.get("per_category", {}) or {}).items():  # type: ignore[union-attr]
            if "error" in stats:
                cells = [category, "-", "-", "-", str(stats["error"])]
            else:
                cells = [
                    category,
                    f"{float(stats.get('refusal_rate_before', 0)):.2f}",
                    f"{float(stats.get('refusal_rate_after', 0)):.2f}",
                    f"{float(stats.get('drift', 0)):.3f}",
                    f"n={int(stats.get('probes', 0))}",
                ]
            rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")

        alert_rows = []
        for alert in alerts or []:
            entry = alert.as_dict() if hasattr(alert, "as_dict") else dict(alert)  # type: ignore[call-overload]
            alert_rows.append(
                f"<li class='{entry.get('severity')}'><b>{entry.get('severity')}</b> "
                f"{entry.get('rule')} &mdash; {entry.get('message')}</li>"
            )

        table = "".join(rows) or "<tr><td colspan='5'>no per-category data</td></tr>"
        alerts_html = "".join(alert_rows) or "<li>none &mdash; all gates passed</li>"

        html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>NEON FORGE :: {session or 'session'}</title>
<style>
body{{background:#05060a;color:#cfe8ff;font-family:ui-monospace,Consolas,monospace;padding:2rem}}
h1{{color:#39f6ff;letter-spacing:.35em;text-shadow:0 0 18px #39f6ff88;margin:0}}
.sub{{color:#7a5cff;letter-spacing:.25em;margin:.4rem 0 2rem}}
.card{{border:1px solid #1b3a5c;border-radius:10px;padding:1rem 1.25rem;margin-bottom:1rem;
background:linear-gradient(180deg,#0a1220,#070b12)}}
.metric{{font-size:1.7rem;color:#9dff5c}}
table{{width:100%;border-collapse:collapse;margin-top:.5rem}}
td,th{{padding:.4rem .6rem;border-bottom:1px solid #14263a;text-align:left}}
th{{color:#39f6ff}}
li{{margin:.35rem 0}} li.critical{{color:#ff4d6d}} li.high{{color:#ff9f43}}
li.medium{{color:#ffe066}} li.low{{color:#7cc4ff}}
</style></head><body>
<h1>NEON FORGE</h1>
<div class="sub">MODEL OUTPUT MONITOR :: {session or 'session'}</div>
<div class="card"><div>refusal rate</div>
<div class="metric">{before:.3f} &rarr; {after:.3f}</div></div>
<div class="card"><div>capability preserved</div><div class="metric">{preservation:.3f}</div></div>
<div class="card"><div>activation drift</div><div class="metric">{drift:.3f}</div></div>
<div class="card"><div>per-category</div><table>
<tr><th>category</th><th>refusal before</th><th>refusal after</th><th>drift</th><th>probes</th></tr>
{table}</table></div>
<div class="card"><div>alerts</div><ul>{alerts_html}</ul></div>
</body></html>
"""
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(html)
        return path
