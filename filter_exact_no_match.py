import argparse
import json
import csv
import sys
from pathlib import Path

VALID_MATCH_TYPES = {"exactNoMatch", "exactMatch", "fuzzyNoMatch", "fuzzyMatch"}

SKIP_SCORES = {"100%", "100+%"}


def extract_text(content):
    if not content:
        return ""
    return "".join(item["text"] for item in content if isinstance(item, dict) and "text" in item)


def filter_report(html_path: Path, match_types: set):
    raw = html_path.read_text(encoding="utf-8", errors="replace")

    marker = '"gs4trTranscheckReportData"'
    pos = raw.find(marker)
    if pos < 0:
        print("Could not find gs4trTranscheckReportData", file=sys.stderr)
        sys.exit(1)
    brace = raw.find("{", pos + len(marker))
    data, _ = json.JSONDecoder().raw_decode(raw, brace)

    rows = []
    for f in data.get("files", []):
        header = f.get("header", "")
        for seg in f.get("segments", []):
            score_val = (seg.get("score") or {}).get("value", "")
            if score_val in SKIP_SCORES:
                continue
            seg_id = seg.get("segmentId", "")
            src_text = extract_text(seg.get("source", {}).get("content", []))
            tgt_text = extract_text(seg.get("target", {}).get("content", []))
            term = seg.get("errors", {}).get("terminology", {})
            comment = term.get("comment", "")
            for hl in term.get("highlights", []):
                sh = hl.get("sourceHighlight", {})
                if sh.get("matchType") not in match_types:
                    continue
                for t in sh.get("terms", []):
                    rows.append({
                        "file": header,
                        "segmentId": seg_id,
                        "score": score_val,
                        "sourceTerm": t.get("sourceTerm", ""),
                        "expectedTargetTerm": t.get("targetTerm", ""),
                        "comment": comment,
                        "source": src_text,
                        "target": tgt_text,
                        "terminology": t.get("terminologyName", ""),
                        "sourceFile": (t.get("targetAttributes") or {}).get("Source File", ""),
                    })

    suffix = "_".join(sorted(match_types))
    out_csv = html_path.with_name(html_path.stem + f"_{suffix}_filtered.csv")
    out_txt = html_path.with_name(html_path.stem + f"_{suffix}_filtered.txt")

    with out_csv.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "file", "segmentId", "score", "sourceTerm", "expectedTargetTerm",
            "comment", "source", "target", "terminology", "sourceFile",
        ])
        w.writeheader()
        w.writerows(rows)

    with out_txt.open("w", encoding="utf-8") as fh:
        label = ", ".join(sorted(match_types))
        lines = [f"Terminology errors ({label}): {len(rows)}\n", "=" * 80 + "\n\n"]
        for r in rows:
            lines.append(
                f"[{r['segmentId']}] ({r['score']}) {r['sourceTerm']} -> expected: {r['expectedTargetTerm']}\n"
                f"  Comment : {r['comment']}\n"
                f"  Source  : {r['source']}\n"
                f"  Target  : {r['target']}\n"
                f"  Glossary: {r['sourceFile']}\n"
                f"  File    : {r['file']}\n\n"
            )
        fh.writelines(lines)

    print(f"Wrote {len(rows)} entries ({', '.join(sorted(match_types))})")
    print(f"  - {out_csv}")
    print(f"  - {out_txt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter TranscheckReport HTML by terminology match type.")
    parser.add_argument("html", nargs="?", help="Path to TranscheckReport HTML file")
    parser.add_argument(
        "--match-type", "-m",
        nargs="+",
        default=["exactNoMatch"],
        choices=sorted(VALID_MATCH_TYPES),
        metavar="TYPE",
        help=f"Match type(s) to include. Choices: {', '.join(sorted(VALID_MATCH_TYPES))}. Default: exactNoMatch",
    )
    args = parser.parse_args()

    if args.html:
        path = Path(args.html)
    else:
        downloads = Path.home() / "Downloads"
        candidates = sorted(downloads.glob("*TranscheckReport*.html"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            print("Pass HTML path as argument.", file=sys.stderr)
            sys.exit(1)
        path = candidates[0]
        print(f"Using: {path}")

    filter_report(path, set(args.match_type))
