import csv as csv_mod
import html as html_module
import io
import json
import re
import urllib.request
from pathlib import Path

import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup

from filter_exact_no_match import extract_text, SKIP_SCORES, VALID_MATCH_TYPES

VERSION = "1.1.0"

UPDATE_URLS = {
    "app.py": "https://raw.githubusercontent.com/AhmedAbdLMoaty/QA-shortcut/main/app.py",
    "filter_exact_no_match.py": "https://raw.githubusercontent.com/AhmedAbdLMoaty/QA-shortcut/main/filter_exact_no_match.py",
}

SCORE_100_PLUS = {"100%", "100+%"}

CLAUDE_MODELS = ["claude-opus-4-7", "claude-sonnet-4-6", "claude-haiku-4-5"]

TRIAGE_SYSTEM = """You are an expert translation QA specialist.

You will receive a list of terminology error entries from a translation quality check report.
For each entry, classify it as exactly one of:
- Real error: the wrong term was genuinely used and must be fixed
- False positive: the flag is overzealous; the translation is contextually acceptable
- Duplicate: same term error already appears in another entry in this batch
- Ambiguous: cannot determine without more context; needs human review

Reply ONLY with a JSON array. Each element must have these keys:
  "entry": the entry number (integer)
  "segmentId": the segmentId string
  "verdict": one of: Real error, False positive, Duplicate, Ambiguous
  "reason": one short sentence explaining the verdict

Do not include any text outside the JSON array."""

FIX_SYSTEM = """You are an expert translation post-editor.

You will receive a list of flagged segments. Each has:
- source text (original language)
- current (wrong) target translation
- the wrong term that was used
- the correct approved term that must be used instead

For each entry, rewrite the target translation so that:
1. The correct approved term replaces the wrong one
2. The rest of the sentence remains natural and fluent
3. Grammar, punctuation and style are preserved

Reply ONLY with a JSON array. Each element must have these keys:
  "entry": the entry number (integer)
  "segmentId": the segmentId string
  "fixedTarget": the corrected target translation (full sentence)
  "note": one short sentence describing what was changed (or "No change needed" if the fix was not applicable)

Do not include any text outside the JSON array."""


@st.cache_data(ttl=10)
def check_remote_version() -> str | None:
    try:
        with urllib.request.urlopen(UPDATE_URLS["app.py"], timeout=5) as r:
            for line in r.read().decode("utf-8", errors="replace").splitlines():
                if line.startswith("VERSION"):
                    return line.split('"')[1]
    except Exception:
        pass
    return None


def get_static_html(raw: str) -> str:
    marker = '"gs4trTranscheckReportData"'
    pos = raw.find(marker)
    if pos < 0:
        raise ValueError("Could not find gs4trTranscheckReportData in file.")
    brace = raw.find("{", pos + len(marker))
    data, _ = json.JSONDecoder().raw_decode(raw, brace)
    return html_module.unescape(data.get("staticReportContent", ""))


def parse_terminology(raw: str, match_types: set) -> list:
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
                        "matchType": sh.get("matchType", ""),
                        "sourceTerm": t.get("sourceTerm", ""),
                        "expectedTargetTerm": t.get("targetTerm", ""),
                        "comment": comment,
                        "source": src_text,
                        "target": tgt_text,
                        "terminology": t.get("terminologyName", ""),
                        "sourceFile": (t.get("targetAttributes") or {}).get("Source File", ""),
                    })
    return rows


def strip_inline(cell_soup) -> str:
    for tag in cell_soup.find_all(["del", "ins"]):
        tag.decompose()
    return cell_soup.get_text(" ", strip=True)


def parse_unedited_fuzzy(static_html: str) -> list:
    soup = BeautifulSoup(static_html, "lxml")
    rows = []
    for fieldset in soup.find_all("fieldset", id=re.compile(r"CHECKUNEDITEDFUZZY")):
        filename = ""
        h3 = fieldset.find_previous("h3")
        if h3:
            filename = h3.get_text(strip=True)
        table = fieldset.find("table", class_="checkTable")
        if not table:
            continue
        for tr in table.find("tbody").find_all("tr"):
            tds = tr.find_all("td")
            if len(tds) < 6:
                continue
            score = tds[4].get_text(strip=True)
            rows.append({
                "file": filename,
                "segmentId": tds[1].get_text(strip=True),
                "score": score,
                "source": tds[2].get_text(" ", strip=True),
                "target": tds[3].get_text(" ", strip=True),
                "comment": tds[5].get_text(strip=True),
                "is100": score in SCORE_100_PLUS,
            })
    return rows


def parse_consistency(static_html: str, anchor_name: str) -> list:
    soup = BeautifulSoup(static_html, "lxml")
    rows = []
    anchor = soup.find("a", attrs={"name": anchor_name})
    if not anchor:
        return rows
    table = anchor.find_next("table", class_="consistencyCheckTable")
    if not table:
        return rows
    tbody = table.find("tbody")
    if not tbody:
        return rows
    for tr in tbody.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 6:
            continue
        score = tds[5].get_text(strip=True)
        rows.append({
            "file": tds[0].get_text(strip=True),
            "segmentId": tds[2].get_text(strip=True),
            "score": score,
            "source": tds[3].get_text(" ", strip=True),
            "target": strip_inline(tds[4]),
            "is100": score in SCORE_100_PLUS,
        })
    return rows


def to_csv_bytes_term(rows: list) -> bytes:
    buf = io.StringIO()
    w = csv_mod.DictWriter(buf, fieldnames=[
        "file", "segmentId", "score", "matchType", "sourceTerm", "expectedTargetTerm",
        "comment", "source", "target", "terminology", "sourceFile",
    ])
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().encode("utf-8-sig")


def to_txt_bytes(rows: list, match_types: set) -> bytes:
    label = ", ".join(sorted(match_types))
    lines = [f"Terminology errors ({label}): {len(rows)}\n", "=" * 80 + "\n\n"]
    for r in rows:
        lines.append(
            f"[{r['segmentId']}] ({r['score']}) [{r['matchType']}] {r['sourceTerm']} -> expected: {r['expectedTargetTerm']}\n"
            f"  Comment : {r['comment']}\n"
            f"  Source  : {r['source']}\n"
            f"  Target  : {r['target']}\n"
            f"  Glossary: {r['sourceFile']}\n"
            f"  File    : {r['file']}\n\n"
        )
    return "".join(lines).encode("utf-8")


def to_csv_bytes(rows: list, fields: list) -> bytes:
    buf = io.StringIO()
    w = csv_mod.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue().encode("utf-8-sig")


def build_entry_context(rows: list) -> str:
    lines = []
    for i, r in enumerate(rows, 1):
        lines.append(
            f"Entry {i} | segmentId={r['segmentId']} | matchType={r['matchType']} | score={r['score']}\n"
            f"  Source term     : {r['sourceTerm']}\n"
            f"  Expected target : {r['expectedTargetTerm']}\n"
            f"  Source text     : {r['source']}\n"
            f"  Target text     : {r['target']}\n"
            f"  Comment         : {r['comment']}\n"
        )
    return "\n".join(lines)


def triage_with_claude(api_key: str, model: str, rows: list) -> list:
    import anthropic
    context = build_entry_context(rows)
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=4096,
        system=TRIAGE_SYSTEM,
        messages=[{"role": "user", "content": context}],
    )
    return json.loads(message.content[0].text)


def fix_with_claude(api_key: str, model: str, rows: list) -> list:
    import anthropic
    lines = []
    for i, r in enumerate(rows, 1):
        lines.append(
            f"Entry {i} | segmentId={r['segmentId']}\n"
            f"  Source text      : {r['source']}\n"
            f"  Current target   : {r['target']}\n"
            f"  Wrong term used  : {r['sourceTerm']} -> {r['target']}\n"
            f"  Correct term     : {r['sourceTerm']} -> {r['expectedTargetTerm']}\n"
        )
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=4096,
        system=FIX_SYSTEM,
        messages=[{"role": "user", "content": "\n".join(lines)}],
    )
    return json.loads(message.content[0].text)


def analyze_with_claude(api_key: str, model: str, rows: list, user_prompt: str) -> str:
    import anthropic
    context = build_entry_context(rows)
    system = "You are an expert translation QA specialist. Analyze the provided terminology error entries from a translation quality check report and answer the user's question clearly and concisely."
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=2048,
        system=system,
        messages=[{"role": "user", "content": f"{context}\n\nUser question: {user_prompt}"}],
    )
    return message.content[0].text


st.set_page_config(page_title="QA Report", layout="wide")
st.title("QA Report")

with st.sidebar:
    st.header("Claude AI Settings")
    api_key = st.text_input("API Key", type="password", placeholder="sk-ant-...")
    model = st.selectbox("Model", options=CLAUDE_MODELS, index=0)
    st.caption("Required only for AI analysis.")

    st.divider()
    st.header("App Updates")
    remote_version = check_remote_version()
    if remote_version and remote_version != VERSION:
        st.warning(f"Update available: v{remote_version}")
    else:
        st.caption(f"v{VERSION} — up to date")
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

uploaded = st.file_uploader("Upload TranscheckReport HTML", type=["html", "htm"])

if not uploaded:
    st.stop()

raw = uploaded.read().decode("utf-8", errors="replace")
stem = uploaded.name.rsplit(".", 1)[0]

try:
    static_html = get_static_html(raw)
except ValueError as e:
    st.error(str(e))
    st.stop()

fuzzy_rows = parse_unedited_fuzzy(static_html)
target_rows = parse_consistency(static_html, "targetconsistency")
source_rows = parse_consistency(static_html, "sourceconsistency")

tab_term, tab_fuzzy, tab_target, tab_source = st.tabs([
    "Terminology",
    f"Unedited Fuzzy ({len(fuzzy_rows)})",
    f"Target Consistency ({len(target_rows)})",
    f"Source Consistency ({len(source_rows)})",
])

with tab_term:
    selected_types = st.multiselect(
        "Match types to filter",
        options=sorted(VALID_MATCH_TYPES),
        default=["exactNoMatch"],
    )
    run = st.button("Run Filter", disabled=not selected_types)

    if run and selected_types:
        with st.spinner("Parsing report..."):
            try:
                rows = parse_terminology(raw, set(selected_types))
                st.session_state["rows"] = rows
                st.session_state["selected_types"] = set(selected_types)
                st.session_state["stem"] = stem
                st.session_state.pop("triage", None)
            except ValueError as e:
                st.error(str(e))
                st.stop()

    if "rows" in st.session_state and st.session_state["rows"] is not None:
        rows = st.session_state["rows"]
        sel_types = st.session_state["selected_types"]
        suffix = "_".join(sorted(sel_types))

        st.success(f"Found {len(rows)} entries")

        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "Download CSV",
                    data=to_csv_bytes_term(rows),
                    file_name=f"{stem}_{suffix}_filtered.csv",
                    mime="text/csv",
                )
            with col2:
                st.download_button(
                    "Download TXT",
                    data=to_txt_bytes(rows, sel_types),
                    file_name=f"{stem}_{suffix}_filtered.txt",
                    mime="text/plain",
                )

            st.divider()
            st.subheader("Claude AI")
            ai_tab_triage, ai_tab_fix, ai_tab_ask = st.tabs(["Triage entries", "Fix translations", "Ask a question"])

            with ai_tab_triage:
                st.caption("Claude classifies each entry as Real error, False positive, Duplicate, or Ambiguous.")
                triage_count = st.slider("Entries to triage", 1, len(rows), min(30, len(rows)))
                if st.button("Run Triage"):
                    if not api_key:
                        st.warning("Enter your Claude API key in the sidebar first.")
                    else:
                        with st.spinner(f"Triaging {triage_count} entries..."):
                            try:
                                results = triage_with_claude(api_key, model, rows[:triage_count])
                                st.session_state["triage"] = results
                            except Exception as e:
                                st.error(f"Claude API error: {e}")

                if "triage" in st.session_state:
                    triage_df = pd.DataFrame(st.session_state["triage"])
                    verdict_order = ["Real error", "Duplicate", "Ambiguous", "False positive"]
                    triage_df["_order"] = triage_df["verdict"].apply(
                        lambda v: verdict_order.index(v) if v in verdict_order else 99
                    )
                    triage_df = triage_df.sort_values("_order").drop(columns="_order").reset_index(drop=True)
                    verdict_counts = triage_df["verdict"].value_counts()
                    cols = st.columns(len(verdict_counts))
                    for col, (verdict, count) in zip(cols, verdict_counts.items()):
                        col.metric(verdict, count)
                    st.dataframe(triage_df, use_container_width=True, hide_index=True)
                    st.download_button(
                        "Download Triage CSV",
                        data=triage_df.to_csv(index=False).encode("utf-8-sig"),
                        file_name=f"{stem}_{suffix}_triage.csv",
                        mime="text/csv",
                    )

            with ai_tab_fix:
                st.caption("Claude rewrites each flagged segment using the correct glossary term.")
                fix_count = st.slider("Entries to fix", 1, len(rows), min(30, len(rows)), key="fix_count")
                if st.button("Run Fix"):
                    if not api_key:
                        st.warning("Enter your Claude API key in the sidebar first.")
                    else:
                        with st.spinner(f"Fixing {fix_count} entries..."):
                            try:
                                fix_results = fix_with_claude(api_key, model, rows[:fix_count])
                                fix_df = pd.DataFrame(fix_results)
                                rows_df = pd.DataFrame(rows[:fix_count])[["segmentId", "source", "target", "sourceTerm", "expectedTargetTerm"]]
                                fix_df = fix_df.merge(rows_df, on="segmentId", how="left")
                                fix_df = fix_df[["entry", "segmentId", "source", "target", "fixedTarget", "sourceTerm", "expectedTargetTerm", "note"]]
                                st.session_state["fix_df"] = fix_df
                            except Exception as e:
                                st.error(f"Claude API error: {e}")

                if "fix_df" in st.session_state:
                    fix_df = st.session_state["fix_df"]
                    st.dataframe(fix_df, use_container_width=True, hide_index=True)
                    st.download_button(
                        "Download Fixed CSV",
                        data=fix_df.to_csv(index=False).encode("utf-8-sig"),
                        file_name=f"{stem}_{suffix}_fixed.csv",
                        mime="text/csv",
                    )

            with ai_tab_ask:
                user_prompt = st.text_area("Your question", placeholder="e.g. Which errors are most critical to fix first?", height=120)
                analyze_count = st.slider("Entries to send", 1, len(rows), min(30, len(rows)))
                if st.button("Ask Claude", disabled=not user_prompt):
                    if not api_key:
                        st.warning("Enter your Claude API key in the sidebar first.")
                    else:
                        with st.spinner("Asking Claude..."):
                            try:
                                result = analyze_with_claude(api_key, model, rows[:analyze_count], user_prompt)
                                st.markdown(result)
                            except Exception as e:
                                st.error(f"Claude API error: {e}")
        else:
            st.info("No entries matched the selected filter(s).")

FUZZY_FIELDS = ["file", "segmentId", "score", "source", "target", "comment"]
CONSISTENCY_FIELDS = ["file", "segmentId", "score", "source", "target"]

with tab_fuzzy:
    if not fuzzy_rows:
        st.info("No Unedited Fuzzy errors found in this report.")
    else:
        scores_available = sorted(set(r["score"] for r in fuzzy_rows))
        col1, col2 = st.columns(2)
        with col1:
            sel_scores = st.multiselect("Filter by score", options=scores_available, default=scores_available, key="fuzzy_scores")
        with col2:
            hide_100 = st.checkbox("Hide 100% / 100+% matches", value=False, key="fuzzy_hide100")
        filtered_fuzzy = [r for r in fuzzy_rows if r["score"] in sel_scores and (not hide_100 or not r["is100"])]
        st.caption(f"Showing {len(filtered_fuzzy)} of {len(fuzzy_rows)} entries")
        if filtered_fuzzy:
            st.dataframe(pd.DataFrame(filtered_fuzzy)[FUZZY_FIELDS], use_container_width=True, hide_index=True)
            st.download_button(
                "Download CSV",
                data=to_csv_bytes(filtered_fuzzy, FUZZY_FIELDS),
                file_name=f"{stem}_unedited_fuzzy.csv",
                mime="text/csv",
                key="dl_fuzzy",
            )

with tab_target:
    if not target_rows:
        st.info("No Target Consistency errors found in this report.")
    else:
        scores_available = sorted(set(r["score"] for r in target_rows))
        col1, col2 = st.columns(2)
        with col1:
            sel_scores = st.multiselect("Filter by score", options=scores_available, default=scores_available, key="target_scores")
        with col2:
            hide_100 = st.checkbox("Hide 100% / 100+% matches", value=False, key="target_hide100")
        filtered_target = [r for r in target_rows if r["score"] in sel_scores and (not hide_100 or not r["is100"])]
        st.caption(f"Showing {len(filtered_target)} of {len(target_rows)} entries")
        if filtered_target:
            st.dataframe(pd.DataFrame(filtered_target)[CONSISTENCY_FIELDS], use_container_width=True, hide_index=True)
            st.download_button(
                "Download CSV",
                data=to_csv_bytes(filtered_target, CONSISTENCY_FIELDS),
                file_name=f"{stem}_target_consistency.csv",
                mime="text/csv",
                key="dl_target",
            )

with tab_source:
    if not source_rows:
        st.info("No Source Consistency errors found in this report.")
    else:
        scores_available = sorted(set(r["score"] for r in source_rows))
        col1, col2 = st.columns(2)
        with col1:
            sel_scores = st.multiselect("Filter by score", options=scores_available, default=scores_available, key="source_scores")
        with col2:
            hide_100 = st.checkbox("Hide 100% / 100+% matches", value=False, key="source_hide100")
        filtered_source = [r for r in source_rows if r["score"] in sel_scores and (not hide_100 or not r["is100"])]
        st.caption(f"Showing {len(filtered_source)} of {len(source_rows)} entries")
        if filtered_source:
            st.dataframe(pd.DataFrame(filtered_source)[CONSISTENCY_FIELDS], use_container_width=True, hide_index=True)
            st.download_button(
                "Download CSV",
                data=to_csv_bytes(filtered_source, CONSISTENCY_FIELDS),
                file_name=f"{stem}_source_consistency.csv",
                mime="text/csv",
                key="dl_source",
            )
