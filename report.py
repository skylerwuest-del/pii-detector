"""Self-contained HTML summary of the latest scan. Counts and paths only, never matched values."""
import csv
import datetime
import html
from pathlib import Path
import config
from inference import folder_risk, query_results, read_meta

STYLE = """
:root { --bg:#ffffff; --fg:#1c1f24; --muted:#5f6670; --line:#dfe3e8; --card:#f6f8fa;
        --high:#c62828; --med:#b26a00; --low:#2e7d32; --bar:#3b6fd8; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#14171b; --fg:#e7eaee; --muted:#9aa3ad; --line:#2c3238; --card:#1c2126;
          --high:#ef6b6b; --med:#e0a43c; --low:#6cc070; --bar:#7aa2f7; } }
* { box-sizing:border-box; }
body { margin:0; padding:24px 16px 48px; background:var(--bg); color:var(--fg);
       font:15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width:1100px; margin:0 auto; }
h1 { font-size:24px; margin:0 0 4px; } h2 { font-size:18px; margin:32px 0 8px; }
.muted { color:var(--muted); }
.cards { display:grid; grid-template-columns:repeat(auto-fit, minmax(160px, 1fr)); gap:12px; margin-top:16px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px 14px; }
.card b { display:block; font-size:22px; font-variant-numeric:tabular-nums; }
.wrap { overflow-x:auto; border:1px solid var(--line); border-radius:8px; }
table { border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }
th, td { text-align:left; padding:7px 10px; border-bottom:1px solid var(--line); vertical-align:top; }
th { background:var(--card); font-weight:600; position:sticky; top:0; }
td.num, th.num { text-align:right; }
td.path { word-break:break-all; font-family:ui-monospace, Consolas, monospace; font-size:13px; }
.bar { display:inline-block; height:8px; border-radius:4px; background:var(--bar); vertical-align:middle; }
.High { color:var(--high); font-weight:600; } .Medium { color:var(--med); font-weight:600; }
.Low { color:var(--med); } .None { color:var(--low); }
"""


def esc(value):
    return html.escape(str(value), quote=True)


def table(headers, rows, numeric=()):
    head = "".join(f'<th class="num">{esc(h)}</th>' if i in numeric else f"<th>{esc(h)}</th>" for i, h in enumerate(headers))
    body = "".join("<tr>" + "".join(cell for cell in row) + "</tr>" for row in rows)
    return f'<div class="wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def read_inspections(roots):
    report = config.OUTPUT_DIRECTORY / "inspection_report.csv"
    try:
        with report.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError:
        return []
    roots = {str(r) for r in roots}
    return [r for r in rows if not roots or r.get("scan_root") in roots]


def write_html_report(full=False, output=None, top=50):
    from inference import database_for
    database = database_for(full)
    if not database.is_file():
        raise OSError("No scan results yet. Run a scan first.")
    meta = read_meta(database)
    rows, total, threshold = query_results(database, limit=top)
    flagged = query_results(database, high=True, limit=0)[1]
    folders = folder_risk(database, depth=1, limit=15)
    inspections = read_inspections(meta.get("roots", []))
    statuses = meta.get("statuses", {})
    when = datetime.datetime.fromtimestamp(meta.get("finished", meta.get("started", 0))).strftime("%Y-%m-%d %H:%M")
    confirmed = sum(1 for r in inspections if r.get("indicator_counts"))

    cards = [("Files evaluated", f"{total:,}"), ("Above threshold", f"{flagged:,}"),
             ("Inspected", f"{len(inspections):,}"), ("With PII indicators", f"{confirmed:,}"),
             ("New / changed", f"{statuses.get('new', 0):,} / {statuses.get('changed', 0):,}")]
    parts = [f"<h1>PII scan report</h1><p class='muted'>{esc(when)} &middot; "
             f"{'Full computer audit' if meta.get('full') else 'Controlled directory scan'} &middot; roots: "
             f"{esc(', '.join(meta.get('roots', [])))} &middot; threshold {threshold:.2f}"
             f"{'' if meta.get('complete') else ' &middot; <b>scan was cancelled</b>'}</p>",
             '<div class="cards">' + "".join(f'<div class="card"><span class="muted">{esc(k)}</span><b>{esc(v)}</b></div>' for k, v in cards) + "</div>"]
    if inspections:
        parts.append("<h2>Verified files</h2><p class='muted'>Deterministic checks of file contents. Counts only; matched values are never stored.</p>")
        parts.append(table(["Severity", "File", "Indicators", "CNN score"], [[
            f'<td class="{esc(r.get("severity", "None"))}">{esc(r.get("severity") if r.get("indicator_counts") else r.get("status"))}</td>',
            f'<td class="path">{esc(r["absolute_path"])}</td>',
            f'<td>{esc(r.get("indicator_counts", "").replace(";", ", ").replace("=", " x") or "-")}</td>',
            f'<td class="num">{esc(r.get("cnn_score"))}</td>'] for r in inspections], numeric=(3,)))
    if folders:
        parts.append("<h2>Riskiest folders</h2>")
        parts.append(table(["Folder", "Files", "Above threshold", "Mean score", "Max score"], [[
            f'<td class="path">{esc(f["folder"])}</td>', f'<td class="num">{f["files"]:,}</td>',
            f'<td class="num">{f["flagged"]:,}</td>', f'<td class="num">{f["mean"]:.3f}</td>',
            f'<td class="num">{f["max"]:.3f}</td>'] for f in folders], numeric=(1, 2, 3, 4)))
    parts.append(f"<h2>Highest-risk paths (top {min(top, total)} of {total:,})</h2>"
                 "<p class='muted'>CNN scores rank file names; they are not proof that a file contains PII.</p>")
    parts.append(table(["#", "Score", "", "Path", "Change"], [[
        f'<td class="num">{rank}</td>', f'<td class="num">{score:.4f}</td>',
        f'<td><span class="bar" style="width:{max(2, int(score * 80))}px"></span></td>',
        f'<td class="path">{esc(path)}</td>', f'<td>{esc(status)}</td>']
        for rank, (path, _, score, status) in enumerate(rows, 1)], numeric=(0, 1)))
    parts.append("<p class='muted'>Generated locally by the PII Detection Agent. This file lists paths on your "
                 "computer; keep it private. No indicators does not prove a file is free of PII.</p>")
    document = ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
                "<meta name='viewport' content='width=device-width, initial-scale=1'>"
                f"<title>PII Scan Report</title><style>{STYLE}</style></head><body><main>"
                + "".join(parts) + "</main></body></html>")
    output = Path(output or config.OUTPUT_DIRECTORY / f"{'full_scan_' if full else ''}report.html")
    output.write_text(document, encoding="utf-8")
    return output
