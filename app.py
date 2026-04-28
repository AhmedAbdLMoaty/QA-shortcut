import io
import csv
import json
import urllib.request
from collections import Counter
from pathlib import Path

import pandas as pd
import streamlit as st

from filter_exact_no_match import extract_text, SKIP_SCORES, VALID_MATCH_TYPES

UPDATE_URLS = {
    "app.py": "https://raw.githubusercontent.com/AhmedAbdLMoaty/QA-shortcut/main/app.py",
    "filter_exact_no_match.py": "https://raw.githubusercontent.com/AhmedAbdLMoaty/QA-shortcut/main/filter_exact_no_match.py",
}

CLAUDE_MODELS = [
    "claude-opus-4-7",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
]

SCORE_ORDER = [
    "MT", "75%", "76%", "77%", "78%", "79%", "80%", "81%", "82%", "83%",
    "84%", "85%", "86%", "87%", "88%", "89%", "90%", "91%", "92%", "93%",
    "94%", "95%", "96%", "97%", "98%", "99%",
]


def score_rank(val: str) -> int:
    try:
        return SCORE_ORDER.index(val)
    except ValueError:
        return -1


def parse_report(raw: str, match_types: set) -> list:
    marker = '"gs4trTranscheckReportData"'
    pos = raw.find(marker)
    if pos < 0:
        raise ValueError("Could not find gs4trTranscheckReportData in file.")
    brace = raw.find("{", pos + len(marker))
    data, _ = json.JSONDecoder().raw_decode(raw, brace)

    rows = []
    for f in data.get("files", []):
        header = f.get("header", "")
        for seg in f.get("segments", []):
            score_obj = seg.get("score") or {}
            score_val = score_obj.get("value", "")
            if score_val in SKIP_SCORES:
                continue
            status = score_obj.get("status", "")
            state_qualifier = score_obj.get("stateQualifier", "")
            seg_id = seg.get("segmentId", "")
            src_text = extract_text(seg.get("source", {}).get("content", []))
            tgt_text = extract_text(seg.get("target", {}).get("content", []))
            term = seg.get("errors", {}).get("terminology", {})
            comment = term.get("comment", "")
            for hl in term.get("highlights", []):
                sh = hl.get("sourceHighlight", {})
                th = hl.get("targetHighlight", {})
                if sh.get("matchType") not in match_types:
                    continue
                for t in sh.get("terms", []):
                    rows.append({
                        "file": header,
                        "segmentId": seg_id,
                        "score": score_val,
                        "status": status,
                        "stateQualifier": state_qualifier,
                        "srcMatchType": sh.get("matchType", ""),
                        "tgtMatchType": th.get("matchType", ""),
                        "sourceTerm": t.get("sourceTerm", ""),
                        "expectedTargetTerm": t.get("targetTerm", ""),
                        "comment": comment,
                        "source": src_text,
                        "target": tgt_text,
                        "terminology": t.get("terminologyName", ""),
                        "glossaryFile": (t.get("targetAttributes") or {}).get("Source File", ""),
                        "batch": (t.get("sourceAttributes") or {}).get("Batch", ""),
                    })
    return rows


def apply_filters(rows: list, filters: dict) -> list:
    result = rows
    if filters.get("statuses"):
        result = [r for r in result if r["status"] in filters["statuses"]]
    if filters.get("glossaries"):
        result = [r for r in result if r["terminology"] in filters["glossaries"]]
    if filters.get("batches"):
        result = [r for r in result if r["batch"] in filters["batches"]]
    if filters.get("score_min") is not None or filters.get("score_max") is not None:
        lo = filters.get("score_min", 0)
        hi = filters.get("score_max", len(SCORE_ORDER) - 1)
        result = [r for r in result if lo <= score_rank(r["score"]) <= hi or score_rank(r["score"]) == -1]
    return result


def to_csv_bytes(rows: list) -> bytes:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=[
        "file", "segmentId", "score", "status", "stateQualifier",
        "srcMatchType", "tgtMatchType", "sourceTerm", "expectedTargetTerm",
        "comment", "source", "target", "terminology", "glossaryFile", "batch",
    ])
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().encode("utf-8-sig")


def to_txt_bytes(rows: list, match_types: set) -> bytes:
    label = ", ".join(sorted(match_types))
    lines = [f"Terminology errors ({label}): {len(rows)}\n", "=" * 80 + "\n\n"]
    for r in rows:
        lines.append(
            f"[{r['segmentId']}] ({r['score']}) [{r['srcMatchType']}→{r['tgtMatchType']}] "
            f"{r['sourceTerm']} -> expected: {r['expectedTargetTerm']}\n"
            f"  Status  : {r['status']} / {r['stateQualifier']}\n"
            f"  Comment : {r['comment']}\n"
            f"  Source  : {r['source']}\n"
            f"  Target  : {r['target']}\n"
            f"  Glossary: {r['glossaryFile']} ({r['batch']})\n"
            f"  File    : {r['file']}\n\n"
        )
    return "".join(lines).encode("utf-8")


def build_context(rows: list) -> str:
    lines = []
    for i, r in enumerate(rows, 1):
        lines.append(
            f"Entry {i} [{r['srcMatchType']}] segmentId={r['segmentId']} score={r['score']} status={r['status']}\n"
            f"  Source term      : {r['sourceTerm']}\n"
            f"  Expected target  : {r['expectedTargetTerm']}\n"
            f"  Source text      : {r['source']}\n"
            f"  Target text      : {r['target']}\n"
            f"  Comment          : {r['comment']}\n"
            f"  Batch            : {r['batch']}\n"
        )
    return "\n".join(lines)


def claude_call(api_key: str, model: str, context: str, user_prompt: str) -> str:
    import anthropic
    system = (
        "You are an expert translation QA specialist. "
        "Analyze the provided terminology error entries from a translation quality check report "
        "and answer the user's question clearly and concisely."
    )
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=2048,
        system=system,
        messages=[{"role": "user", "content": f"{context}\n\nUser question: {user_prompt}"}],
    )
    return message.content[0].text


st.set_page_config(page_title="TranscheckReport Filter", layout="wide")
st.title("TranscheckReport Filter")

with st.sidebar:
    st.header("Claude AI Settings")
    api_key = st.text_input("API Key", type="password", placeholder="sk-ant-...")
    model = st.selectbox("Model", options=CLAUDE_MODELS, index=0)
    st.caption("Required only for AI analysis.")

    st.divider()
    st.header("App Updates")
    if all(UPDATE_URLS.values()):
        if st.button("Update App"):
            errors = []
            for filename, url in UPDATE_URLS.items():
                try:
                    with urllib.request.urlopen(url, timeout=10) as r:
                        content = r.read()
                    Path(filename).write_bytes(content)
                except Exception as e:
                    errors.append(f"{filename}: {e}")
            if errors:
                st.error("Update failed:\n" + "\n".join(errors))
            else:
                st.success("Updated. Please restart the app.")
    else:
        st.caption("Update URLs not configured.")

uploaded = st.file_uploader("Upload TranscheckReport HTML", type=["html", "htm"])

selected_types = st.multiselect(
    "Match types to filter",
    options=sorted(VALID_MATCH_TYPES),
    default=["exactNoMatch"],
)

run = st.button("Run Filter", disabled=not uploaded or not selected_types)

if run and uploaded and selected_types:
    with st.spinner("Parsing report..."):
        raw = uploaded.read().decode("utf-8", errors="replace")
        try:
            rows = parse_report(raw, set(selected_types))
            st.session_state["rows"] = rows
            st.session_state["selected_types"] = set(selected_types)
            st.session_state["stem"] = uploaded.name.rsplit(".", 1)[0]
        except ValueError as e:
            st.error(str(e))
            st.stop()

if "rows" in st.session_state and st.session_state["rows"] is not None:
    rows = st.session_state["rows"]
    sel_types = st.session_state["selected_types"]
    stem = st.session_state["stem"]
    suffix = "_".join(sorted(sel_types))

    if not rows:
        st.info("No entries matched the selected filter(s).")
        st.stop()

    st.success(f"Found {len(rows)} entries across {len({r['file'] for r in rows})} files")

    with st.expander("Per-file error summary", expanded=True):
        file_counts = Counter(r["file"] for r in rows)
        summary_df = pd.DataFrame(
            sorted(file_counts.items(), key=lambda x: -x[1]),
            columns=["File", "Error count"],
        )
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Filters")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        all_statuses = sorted({r["status"] for r in rows if r["status"]})
        sel_statuses = st.multiselect("Status", options=all_statuses, default=all_statuses)
    with col_b:
        all_glossaries = sorted({r["terminology"] for r in rows if r["terminology"]})
        sel_glossaries = st.multiselect("Glossary", options=all_glossaries, default=all_glossaries)
    with col_c:
        all_batches = sorted({r["batch"] for r in rows if r["batch"]})
        sel_batches = st.multiselect("Batch", options=all_batches, default=all_batches)

    ranked_scores = sorted(
        {r["score"] for r in rows if score_rank(r["score"]) >= 0},
        key=score_rank,
    )
    unranked = {r["score"] for r in rows if score_rank(r["score"]) == -1}

    if ranked_scores:
        score_lo, score_hi = st.select_slider(
            "Score range",
            options=ranked_scores,
            value=(ranked_scores[0], ranked_scores[-1]),
        )
        lo_idx = score_rank(score_lo)
        hi_idx = score_rank(score_hi)
    else:
        lo_idx, hi_idx = None, None

    filtered = apply_filters(rows, {
        "statuses": set(sel_statuses),
        "glossaries": set(sel_glossaries),
        "batches": set(sel_batches),
        "score_min": lo_idx,
        "score_max": hi_idx,
    })

    st.caption(f"Showing {len(filtered)} of {len(rows)} entries after filters")

    if filtered:
        display_cols = [
            "file", "segmentId", "score", "status", "srcMatchType", "tgtMatchType",
            "sourceTerm", "expectedTargetTerm", "source", "target", "terminology", "batch",
        ]
        st.dataframe(pd.DataFrame(filtered)[display_cols], use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Download CSV",
                data=to_csv_bytes(filtered),
                file_name=f"{stem}_{suffix}_filtered.csv",
                mime="text/csv",
            )
        with col2:
            st.download_button(
                "Download TXT",
                data=to_txt_bytes(filtered, sel_types),
                file_name=f"{stem}_{suffix}_filtered.txt",
                mime="text/plain",
            )

        st.divider()
        st.subheader("AI Analysis with Claude")

        analysis_mode = st.radio(
            "Analysis mode",
            options=["All filtered entries", "Per file"],
            horizontal=True,
        )

        user_prompt = st.text_area(
            "Your question or instruction",
            placeholder="e.g. Summarize the most common errors and suggest fixes.",
            height=100,
        )

        if analysis_mode == "All filtered entries":
            analyze_count = st.slider(
                "Number of entries to send",
                min_value=1,
                max_value=len(filtered),
                value=min(30, len(filtered)),
            )
            if st.button("Analyze with Claude", disabled=not user_prompt):
                if not api_key:
                    st.warning("Enter your Claude API key in the sidebar first.")
                else:
                    with st.spinner(f"Sending {analyze_count} entries to {model}..."):
                        try:
                            context = build_context(filtered[:analyze_count])
                            st.markdown(claude_call(api_key, model, context, user_prompt))
                        except Exception as e:
                            st.error(f"Claude API error: {e}")

        else:
            files_with_errors = sorted({r["file"] for r in filtered})
            sel_files = st.multiselect(
                "Select files to analyze",
                options=files_with_errors,
                default=files_with_errors[:3],
            )
            if st.button("Analyze per file", disabled=not user_prompt or not sel_files):
                if not api_key:
                    st.warning("Enter your Claude API key in the sidebar first.")
                else:
                    for fname in sel_files:
                        file_rows = [r for r in filtered if r["file"] == fname]
                        with st.spinner(f"{fname} ({len(file_rows)} entries)..."):
                            try:
                                context = build_context(file_rows)
                                result = claude_call(api_key, model, context, user_prompt)
                                with st.expander(f"{fname} — {len(file_rows)} errors", expanded=True):
                                    st.markdown(result)
                            except Exception as e:
                                st.error(f"{fname}: {e}")
    else:
        st.info("No entries match the current filters.")
