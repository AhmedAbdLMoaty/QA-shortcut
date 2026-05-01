import io
import json
import re
import xml.etree.ElementTree as ET
from copy import deepcopy

import pandas as pd
import streamlit as st

XLIFF_NS = "urn:oasis:names:tc:xliff:document:1.2"
GS4TR_NS = "http://www.gs4tr.org/schema/xliff-ext"

ET.register_namespace("", XLIFF_NS)
ET.register_namespace("gs4tr", GS4TR_NS)

CLAUDE_MODELS = ["claude-opus-4-7", "claude-sonnet-4-6", "claude-haiku-4-5"]

STATE_QUALIFIER_LABELS = {
    "x-no-match": "No match (0%)",
    "fuzzy-match": "Fuzzy match",
    "exact-match": "Exact match (100%)",
    "x-in-context-match": "In-context match (100%+)",
}


def xl(tag):
    return f"{{{XLIFF_NS}}}{tag}"


def gs(tag):
    return f"{{{GS4TR_NS}}}{tag}"


def extract_text_with_placeholders(elem) -> tuple:
    ph_map = {}
    parts = []
    if elem.text:
        parts.append(elem.text)
    for child in elem:
        local = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if local == "ph":
            ph_id = child.get("id", "?")
            placeholder = f"{{ph:{ph_id}}}"
            ph_map[placeholder] = deepcopy(child)
            parts.append(placeholder)
        if child.tail:
            parts.append(child.tail)
    return "".join(parts), ph_map


def extract_target_text(elem) -> str:
    if elem is None:
        return ""
    parts = []
    if elem.text:
        parts.append(elem.text)
    for child in elem:
        local = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if local == "ph":
            ph_id = child.get("id", "?")
            parts.append(f"{{ph:{ph_id}}}")
        if child.tail:
            parts.append(child.tail)
    return "".join(parts)


def set_target_content(target_elem, translated: str, ph_map: dict):
    for child in list(target_elem):
        target_elem.remove(child)
    target_elem.text = None

    parts = re.split(r"(\{ph:\d+\})", translated)
    last_elem = None
    for part in parts:
        m = re.match(r"\{ph:(\d+)\}", part)
        if m:
            original_ph = ph_map.get(part)
            if original_ph is not None:
                new_ph = deepcopy(original_ph)
                new_ph.tail = None
                target_elem.append(new_ph)
                last_elem = new_ph
            else:
                if last_elem is None:
                    target_elem.text = (target_elem.text or "") + part
                else:
                    last_elem.tail = (last_elem.tail or "") + part
        else:
            if last_elem is None:
                target_elem.text = (target_elem.text or "") + part
            else:
                last_elem.tail = (last_elem.tail or "") + part


def parse_txlf(content: bytes):
    tree = ET.parse(io.BytesIO(content))
    root = tree.getroot()

    file_elem = root.find(xl("file"))
    src_lang = file_elem.get("source-language", "de-DE") if file_elem is not None else "de-DE"
    tgt_lang = file_elem.get("target-language", "pl-PL") if file_elem is not None else "pl-PL"

    segments = []
    for tu in root.iter(xl("trans-unit")):
        seg_id = tu.get("id", "")
        is_rep = tu.get(gs("repetition"), "false") == "true"

        source_elem = tu.find(xl("source"))
        target_elem = tu.find(xl("target"))

        source_text, ph_map = extract_text_with_placeholders(source_elem) if source_elem is not None else ("", {})
        target_text = extract_target_text(target_elem)

        state = target_elem.get("state", "") if target_elem is not None else ""
        state_qualifier = target_elem.get("state-qualifier", "") if target_elem is not None else ""
        score_raw = target_elem.get(gs("score"), "0") if target_elem is not None else "0"
        try:
            score = int(score_raw)
        except ValueError:
            score = 0

        note_elem = tu.find(gs("note"))
        note = note_elem.text if note_elem is not None else ""

        alt_trans_elem = tu.find(xl("alt-trans"))
        alt_target = ""
        if alt_trans_elem is not None:
            alt_target_elem = alt_trans_elem.find(xl("target"))
            alt_target = extract_target_text(alt_target_elem) if alt_target_elem is not None else ""

        segments.append({
            "id": seg_id,
            "repetition": is_rep,
            "source": source_text,
            "ph_map": ph_map,
            "target": target_text,
            "altTarget": alt_target,
            "state": state,
            "stateQualifier": state_qualifier,
            "score": score,
            "note": note or "",
            "translated": False,
        })

    return tree, root, src_lang, tgt_lang, segments


def translate_batch(api_key: str, model: str, segments: list, src_lang: str, tgt_lang: str) -> list:
    import anthropic

    system = f"""You are a professional translator specializing in automotive UI strings.
Translate from {src_lang} to {tgt_lang}.

Rules:
- Preserve ALL {{ph:N}} placeholders EXACTLY as they appear. They represent inline codes and must not be changed, moved, or removed.
- Keep translations concise and natural for a car infotainment UI.
- Do not add explanations or annotations.

Reply ONLY with a JSON array. Each element must have exactly:
  "id": the segment id string
  "translation": the full translated text with {{ph:N}} placeholders preserved

Do not include any text outside the JSON array."""

    lines = []
    for seg in segments:
        lines.append(f'id={seg["id"]}\nsource: {seg["source"]}')

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": "\n\n".join(lines)}],
    )
    return json.loads(message.content[0].text)


def apply_translations_to_tree(root, translations: dict, segments: list):
    ph_map_by_id = {s["id"]: s["ph_map"] for s in segments}
    for tu in root.iter(xl("trans-unit")):
        seg_id = tu.get("id", "")
        if seg_id not in translations:
            continue
        target_elem = tu.find(xl("target"))
        if target_elem is None:
            target_elem = ET.SubElement(tu, xl("target"))
        target_elem.set("state", "needs-review-translation")
        set_target_content(target_elem, translations[seg_id], ph_map_by_id.get(seg_id, {}))


def tree_to_bytes(tree) -> bytes:
    buf = io.BytesIO()
    tree.write(buf, encoding="UTF-8", xml_declaration=True)
    return buf.getvalue()


def check_translations(segments: list) -> list:
    issues = []
    for seg in segments:
        if not seg.get("translated"):
            continue
        seg_id = seg["id"]
        source = seg["source"]
        target = seg["target"]

        source_phs = set(re.findall(r"\{ph:\d+\}", source))
        target_phs = set(re.findall(r"\{ph:\d+\}", target))
        missing_phs = source_phs - target_phs
        extra_phs = target_phs - source_phs
        if missing_phs:
            issues.append({"id": seg_id, "severity": "Critical", "check": "Missing placeholder", "detail": f"Lost: {', '.join(sorted(missing_phs))}", "source": source, "target": target})
        if extra_phs:
            issues.append({"id": seg_id, "severity": "Critical", "check": "Extra placeholder", "detail": f"Added: {', '.join(sorted(extra_phs))}", "source": source, "target": target})

        if not target.strip():
            issues.append({"id": seg_id, "severity": "Critical", "check": "Empty translation", "detail": "Translation is blank", "source": source, "target": target})
            continue

        if source.strip() and target.strip() == source.strip():
            issues.append({"id": seg_id, "severity": "Warning", "check": "Untranslated", "detail": "Target identical to source", "source": source, "target": target})

        src_len = len(source.strip())
        tgt_len = len(target.strip())
        if src_len > 10:
            ratio = tgt_len / src_len
            if ratio < 0.25:
                issues.append({"id": seg_id, "severity": "Warning", "check": "Too short", "detail": f"Length ratio {ratio:.2f} (target {tgt_len} chars vs source {src_len})", "source": source, "target": target})
            elif ratio > 4.0:
                issues.append({"id": seg_id, "severity": "Warning", "check": "Too long", "detail": f"Length ratio {ratio:.2f} (target {tgt_len} chars vs source {src_len})", "source": source, "target": target})

        src_numbers = set(re.findall(r"\b\d+\b", source))
        tgt_numbers = set(re.findall(r"\b\d+\b", target))
        lost_numbers = src_numbers - tgt_numbers
        if lost_numbers:
            issues.append({"id": seg_id, "severity": "Warning", "check": "Lost numbers", "detail": f"Numbers in source not in target: {', '.join(sorted(lost_numbers))}", "source": source, "target": target})

        words = target.split()
        if len(words) >= 6:
            for i in range(len(words) - 4):
                chunk = words[i:i+3]
                if words[i+3:i+6] == chunk:
                    issues.append({"id": seg_id, "severity": "Critical", "check": "Repetition loop", "detail": f"Repeated phrase detected: '{' '.join(chunk)}'", "source": source, "target": target})
                    break

    return issues


st.set_page_config(page_title="TXLF Editor", layout="wide")
st.title("TXLF Segment Editor & Translator")

with st.sidebar:
    st.header("Claude AI Settings")
    api_key = st.text_input("API Key", type="password", placeholder="sk-ant-...")
    model = st.selectbox("Model", CLAUDE_MODELS, index=0)
    st.caption("Required for translation.")

uploaded = st.file_uploader("Upload .txlf file", type=["txlf", "xliff"])

if uploaded:
    content = uploaded.read()
    try:
        tree, root, src_lang, tgt_lang, segments = parse_txlf(content)
        st.session_state["txlf_tree"] = tree
        st.session_state["txlf_root"] = root
        st.session_state["txlf_src"] = src_lang
        st.session_state["txlf_tgt"] = tgt_lang
        st.session_state["txlf_segments"] = segments
        st.session_state["txlf_stem"] = uploaded.name.rsplit(".", 1)[0]
        st.session_state.pop("txlf_translated_ids", None)
    except Exception as e:
        st.error(f"Failed to parse file: {e}")
        st.stop()

if "txlf_segments" not in st.session_state:
    st.stop()

segments = st.session_state["txlf_segments"]
src_lang = st.session_state["txlf_src"]
tgt_lang = st.session_state["txlf_tgt"]
stem = st.session_state["txlf_stem"]

st.info(f"{src_lang} → {tgt_lang}  |  {len(segments)} total segments")

st.subheader("Filters")
col1, col2, col3 = st.columns(3)

all_qualifiers = sorted(set(s["stateQualifier"] for s in segments if s["stateQualifier"]))
qualifier_labels = [STATE_QUALIFIER_LABELS.get(q, q) for q in all_qualifiers]

with col1:
    sel_qualifier_labels = st.multiselect(
        "State qualifier",
        options=qualifier_labels,
        default=qualifier_labels,
    )
    sel_qualifiers = {q for q, l in zip(all_qualifiers, qualifier_labels) if l in sel_qualifier_labels}

with col2:
    show_reps = st.checkbox("Include repetitions", value=True)
    only_empty = st.checkbox("Only segments with empty target", value=False)

with col3:
    score_min, score_max = st.slider("Score range", 0, 100, (0, 100))

filtered = [
    s for s in segments
    if s["stateQualifier"] in sel_qualifiers
    and (show_reps or not s["repetition"])
    and score_min <= s["score"] <= score_max
    and (not only_empty or not s["target"].strip())
]

translated_ids = st.session_state.get("txlf_translated_ids", set())

st.caption(f"{len(filtered)} segments match filters")

st.subheader("Segments")

df = pd.DataFrame([{
    "Select": False,
    "id": s["id"],
    "score": s["score"],
    "qualifier": STATE_QUALIFIER_LABELS.get(s["stateQualifier"], s["stateQualifier"]),
    "rep": s["repetition"],
    "source": s["source"],
    "target": s["target"],
    "note": s["note"],
    "done": s["id"] in translated_ids,
} for s in filtered])

edited_df = st.data_editor(
    df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Select": st.column_config.CheckboxColumn("Select", default=False, width="small"),
        "id": st.column_config.TextColumn("ID", width="small"),
        "score": st.column_config.NumberColumn("Score", width="small"),
        "qualifier": st.column_config.TextColumn("Qualifier", width="medium"),
        "rep": st.column_config.CheckboxColumn("Rep", width="small"),
        "source": st.column_config.TextColumn("Source", width="large"),
        "target": st.column_config.TextColumn("Current target", width="large"),
        "note": st.column_config.TextColumn("Note", width="medium"),
        "done": st.column_config.CheckboxColumn("Translated", width="small"),
    },
    disabled=["id", "score", "qualifier", "rep", "source", "target", "note", "done"],
)

selected_ids = set(edited_df[edited_df["Select"]]["id"].tolist())
selected_segs = [s for s in filtered if s["id"] in selected_ids]

col_a, col_b = st.columns(2)
with col_a:
    btn_selected = st.button(
        f"Translate Selected ({len(selected_segs)})",
        disabled=len(selected_segs) == 0,
    )
with col_b:
    btn_all = st.button(f"Translate All Filtered ({len(filtered)})")

to_translate = None
if btn_selected:
    to_translate = selected_segs
elif btn_all:
    to_translate = filtered

if to_translate:
    if not api_key:
        st.warning("Enter your Claude API key in the sidebar first.")
    else:
        BATCH_SIZE = 40
        all_results = []
        batches = [to_translate[i:i + BATCH_SIZE] for i in range(0, len(to_translate), BATCH_SIZE)]
        progress = st.progress(0, text=f"Translating batch 1 of {len(batches)}...")
        try:
            for i, batch in enumerate(batches):
                progress.progress((i) / len(batches), text=f"Translating batch {i+1} of {len(batches)}...")
                results = translate_batch(api_key, model, batch, src_lang, tgt_lang)
                all_results.extend(results)
            progress.progress(1.0, text="Done.")

            translations = {r["id"]: r["translation"] for r in all_results}

            for seg in st.session_state["txlf_segments"]:
                if seg["id"] in translations:
                    seg["target"] = translations[seg["id"]]
                    seg["translated"] = True

            apply_translations_to_tree(
                st.session_state["txlf_root"],
                translations,
                st.session_state["txlf_segments"],
            )

            existing = st.session_state.get("txlf_translated_ids", set())
            st.session_state["txlf_translated_ids"] = existing | set(translations.keys())

            translated_segs = [s for s in st.session_state["txlf_segments"] if s["id"] in translations]
            qa_issues = check_translations(translated_segs)
            st.session_state["qa_issues"] = qa_issues
            st.rerun()

        except Exception as e:
            st.error(f"Translation error: {e}")

if st.session_state.get("txlf_translated_ids"):
    done_count = len(st.session_state["txlf_translated_ids"])
    st.success(f"{done_count} segment(s) translated. Download the updated file below.")
    txlf_bytes = tree_to_bytes(st.session_state["txlf_tree"])
    st.download_button(
        "Download Translated .txlf",
        data=txlf_bytes,
        file_name=f"{stem}_translated.txlf",
        mime="application/xml",
    )

    qa_issues = st.session_state.get("qa_issues", [])
    st.divider()
    st.subheader(f"Translation QA — {len(qa_issues)} issue(s) found")
    if not qa_issues:
        st.success("No issues detected. All placeholders preserved, lengths look normal, no repetition loops.")
    else:
        critical = [i for i in qa_issues if i["severity"] == "Critical"]
        warnings = [i for i in qa_issues if i["severity"] == "Warning"]
        col1, col2 = st.columns(2)
        col1.metric("Critical", len(critical))
        col2.metric("Warning", len(warnings))
        issues_df = pd.DataFrame(qa_issues)[["severity", "id", "check", "detail", "source", "target"]]
        st.dataframe(
            issues_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "severity": st.column_config.TextColumn("Severity", width="small"),
                "id": st.column_config.TextColumn("ID", width="small"),
                "check": st.column_config.TextColumn("Check", width="medium"),
                "detail": st.column_config.TextColumn("Detail", width="large"),
                "source": st.column_config.TextColumn("Source", width="large"),
                "target": st.column_config.TextColumn("Translation", width="large"),
            },
        )
        st.download_button(
            "Download QA Issues CSV",
            data=issues_df.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{stem}_qa_issues.csv",
            mime="text/csv",
        )
