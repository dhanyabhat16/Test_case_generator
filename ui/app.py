"""
app.py
------
Streamlit UI for the Automated Test Case Generator.

Features:
  - File uploader (.py, .java, .pdf, .docx, .yaml, .json)
  - Auto-detects input type and shows a badge
  - "Generate Tests" button triggers the full pipeline
  - Shows generated test cases in syntax-highlighted code blocks
  - Shows traceability info (which chunk each test came from)
  - Download button for every generated file
  - Sidebar shows evaluation metrics (PR%, EX%, CV%) for the run

Run:
    streamlit run ui/app.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

import streamlit as st

# Make sure project root is on path so we can import pipeline etc.
_UI_DIR       = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_UI_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TestGen — AI Test Case Generator",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4f8ef7, #a259ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #888;
        margin-bottom: 1.5rem;
        font-size: 1rem;
    }
    .type-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .badge-code { background: #d4f5d4; color: #1a7a1a; }
    .badge-api  { background: #d4e8ff; color: #1a4a9a; }
    .badge-req  { background: #fff3d4; color: #8a5a00; }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-ok    { color: #2ecc71; }
    .metric-warn  { color: #f39c12; }
    .metric-error { color: #e74c3c; }
    .trace-block {
        background: #f4f4f8;
        border-left: 3px solid #a259ff;
        padding: 0.7rem 1rem;
        border-radius: 4px;
        font-size: 0.8rem;
        color: #555;
        margin-top: 0.5rem;
    }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _detect_type(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext in (".py", ".java"):
        return "code"
    elif ext in (".pdf", ".docx"):
        return "requirements"
    elif ext in (".yaml", ".yml", ".json"):
        return "api"
    return "unknown"


def _type_badge(input_type: str) -> str:
    badges = {
        "code"        : ('<span class="type-badge badge-code">🐍 Source Code</span>', "code"),
        "requirements": ('<span class="type-badge badge-req">📄 Requirements Doc</span>', "req"),
        "api"         : ('<span class="type-badge badge-api">🔌 API Specification</span>', "api"),
    }
    return badges.get(input_type, ('<span class="type-badge">❓ Unknown</span>', "unknown"))


def _read_file_content(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception:
        return ""


def _extract_traceability(content: str) -> tuple[str, str]:
    """Split file content into code part and traceability comment block."""
    lines       = content.splitlines()
    code_lines  = []
    trace_lines = []
    in_trace    = False

    for line in lines:
        if "TRACEABILITY REPORT" in line:
            in_trace = True
        if in_trace:
            trace_lines.append(line)
        else:
            code_lines.append(line)

    return "\n".join(code_lines), "\n".join(trace_lines)


def _metric_color(value: float) -> str:
    if value >= 80:
        return "metric-ok"
    elif value >= 50:
        return "metric-warn"
    return "metric-error"


# ── Sidebar ────────────────────────────────────────────────────────────────────

def render_sidebar(metrics_list: Optional[List[Dict[str, Any]]] = None) -> None:
    with st.sidebar:
        st.markdown("## 🧪 TestGen")
        st.markdown("AI-powered test case generator using RAG + LLM.")
        st.divider()

        st.markdown("### 📊 Evaluation Metrics")

        if not metrics_list:
            st.info("Metrics will appear here after generation.")
            st.markdown("**What each metric means:**")
            st.markdown("- **PR%** — Parse Rate: % of files that are valid Python syntax")
            st.markdown("- **EX%** — Exec Rate: % of files pytest can collect without errors")
            st.markdown("- **CV%** — Combined Coverage: % of source lines covered by ALL tests together")
            return

        n      = len(metrics_list)
        avg_pr = sum(r.get("parse_rate", 0) for r in metrics_list) / n * 100
        avg_ex = sum(r.get("exec_rate",  0) for r in metrics_list) / n * 100
        ok_cnt = sum(1 for r in metrics_list if r.get("status") == "ok")

        # ── Use combined coverage if available, else fall back to per-file avg ─
        combined = metrics_list[0].get("combined_coverage") if metrics_list else None
        if combined is not None:
            avg_cv = combined
        else:
            cvs    = [r["coverage"] for r in metrics_list if r.get("coverage") is not None]
            avg_cv = sum(cvs) / len(cvs) if cvs else None

        pr_cls = _metric_color(avg_pr)
        ex_cls = _metric_color(avg_ex)
        cv_cls = _metric_color(avg_cv) if avg_cv is not None else "metric-warn"
        cv_str = f"{avg_cv:.0f}%" if avg_cv is not None else "N/A"

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value {pr_cls}">{avg_pr:.0f}%</div>
            <div class="metric-label">Parse Rate (PR)</div>
        </div>
        <div class="metric-card">
            <div class="metric-value {ex_cls}">{avg_ex:.0f}%</div>
            <div class="metric-label">Exec Rate (EX)</div>
        </div>
        <div class="metric-card">
            <div class="metric-value {cv_cls}">{cv_str}</div>
            <div class="metric-label">Combined Coverage (CV)</div>
        </div>
        <div class="metric-card">
            <div class="metric-value metric-ok">{ok_cnt}/{n}</div>
            <div class="metric-label">Files Generated OK</div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        st.markdown("**Per-file breakdown:**")
        for r in metrics_list:
            fname = os.path.basename(r.get("test_file", "?"))
            pr_ic = "✅" if r.get("parse_rate", 0) == 1.0 else "❌"
            ex_ic = "✅" if r.get("exec_rate",  0) == 1.0 else "❌"
            cv_v  = f"{r['coverage']:.0f}%" if r.get("coverage") is not None else "—"
            st.markdown(f"`{fname[:30]}`  PR{pr_ic} EX{ex_ic} CV={cv_v}")

        st.divider()
        st.caption("💡 PR=Parse Rate · EX=Exec Rate · CV=Combined Coverage")


# ── Main page ──────────────────────────────────────────────────────────────────

def main() -> None:
    # Session state init
    if "results"      not in st.session_state: st.session_state.results      = []
    if "eval_results" not in st.session_state: st.session_state.eval_results = []
    if "output_dir"   not in st.session_state: st.session_state.output_dir   = None

    # Sidebar
    render_sidebar(st.session_state.eval_results or None)

    # Header
    st.markdown('<div class="main-header">🧪 TestGen</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Automated Test Case Generation using RAG + LLM</div>',
                unsafe_allow_html=True)

    # ── Upload section ─────────────────────────────────────────────────────────
    st.markdown("### 📁 Upload Input File")

    uploaded = st.file_uploader(
        "Drag & drop or click to upload",
        type=["py", "java", "pdf", "docx", "yaml", "yml", "json"],
        help="Supported: Python/Java source files, PDF/DOCX requirement docs, YAML/JSON API specs",
    )

    if uploaded is None:
        st.markdown("""
        > **Supported file types:**
        > - 🐍 `.py` / `.java` — Source code → generates **pytest unit tests**
        > - 📄 `.pdf` / `.docx` — Requirements doc → generates **BDD Gherkin tests**
        > - 🔌 `.yaml` / `.json` — OpenAPI spec → generates **API integration tests**
        """)
        render_sidebar(None)
        return

    # Detect type and show badge
    input_type            = _detect_type(uploaded.name)
    badge_html, badge_cls = _type_badge(input_type)
    st.markdown(badge_html, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        st.success(f"**{uploaded.name}** uploaded successfully ({uploaded.size:,} bytes)")
    with col2:
        top_k = st.number_input(
            "Top-K retrieval", min_value=0, max_value=20, value=5,
            help="Number of similar chunks to retrieve (0 = zero-shot, no RAG)",
        )

    # ── Generate button ────────────────────────────────────────────────────────
    st.markdown("---")
    generate_btn = st.button("⚡ Generate Tests", type="primary", use_container_width=True)

    if generate_btn:
        _run_generation(uploaded, input_type, top_k)

    # ── Show results ───────────────────────────────────────────────────────────
    if st.session_state.results:
        _render_results(st.session_state.results, st.session_state.output_dir)

    # Re-render sidebar with metrics
    render_sidebar(st.session_state.eval_results or None)


# ── Generation logic ───────────────────────────────────────────────────────────

def _run_generation(uploaded, input_type: str, top_k: int) -> None:
    """Save the uploaded file, run pipeline, run evaluator, store in session state."""
    from pipeline import run_pipeline
    from evaluation.evaluator import evaluate_all
    import shutil

    # Save uploaded file to a temp directory
    tmp_dir    = tempfile.mkdtemp(prefix="testgen_input_")
    input_path = os.path.join(tmp_dir, uploaded.name)
    with open(input_path, "wb") as f:
        f.write(uploaded.getbuffer())

    # Create outputs dir and copy source file into it so pytest can import it
    output_dir = os.path.join(tmp_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    st.session_state.output_dir = output_dir
    shutil.copy(input_path, os.path.join(output_dir, uploaded.name))

    # ── Run pipeline with live progress ───────────────────────────────────────
    progress_bar = st.progress(0, text="Starting pipeline...")
    status_box   = st.empty()

    try:
        status_box.info("📂 Step 1/5 — Parsing input file...")
        progress_bar.progress(10, text="Parsing...")
        time.sleep(0.3)

        status_box.info("🧮 Step 2/5 — Embedding chunks into FAISS store...")
        progress_bar.progress(25, text="Embedding...")

        status_box.info("🔍 Step 3/5 — Loading vector store...")
        progress_bar.progress(40, text="Loading store...")

        status_box.info("🤖 Step 4-5/5 — Retrieving context & generating tests with LLM...")
        progress_bar.progress(60, text="Generating tests (this may take 15–60s)...")

        results = run_pipeline(
            input_file=input_path,
            output_dir=output_dir,
            top_k=top_k,
        )

        progress_bar.progress(85, text="Evaluating generated tests...")
        status_box.info("📊 Evaluating generated tests (computing combined coverage)...")

        source_for_cov = input_path if input_type == "code" else None
        eval_results   = evaluate_all(output_dir=output_dir, source_file=source_for_cov)

        progress_bar.progress(100, text="Done!")
        status_box.success(
            f"✅ Done! Generated {sum(1 for r in results if r['status'] == 'ok')} test file(s)."
        )

        st.session_state.results      = results
        st.session_state.eval_results = eval_results

    except Exception as e:
        status_box.error(f"❌ Pipeline error: {e}")
        progress_bar.empty()
        st.exception(e)


# ── Result display ─────────────────────────────────────────────────────────────

def _render_results(results: List[Dict[str, Any]], output_dir: str) -> None:
    """Render generated test files with syntax highlighting, traceability, and download."""
    ok_results  = [r for r in results if r["status"] == "ok"]
    err_results = [r for r in results if r["status"] != "ok"]

    st.markdown("---")
    st.markdown(f"### 📋 Generated Test Cases ({len(ok_results)} file(s))")

    if not ok_results:
        st.warning("No test files were generated successfully.")

    for i, result in enumerate(ok_results):
        source_name = result.get("source", "unknown")
        out_file    = result.get("output_file", "")
        file_label  = os.path.basename(out_file) if out_file else f"Result {i+1}"

        with st.expander(f"📄 {file_label}  —  `{source_name}`", expanded=(i == 0)):
            if out_file and os.path.exists(out_file):
                content              = _read_file_content(out_file)
                code_part, trace_part = _extract_traceability(content)

                lang = "python" if out_file.endswith(".py") else "gherkin"
                st.code(code_part.strip() or content, language=lang)

                # Traceability shown inline (no nested expander)
                if trace_part.strip():
                    st.markdown("**🔗 Traceability — source chunks used:**")
                    st.markdown(
                        f'<div class="trace-block"><pre>{trace_part}</pre></div>',
                        unsafe_allow_html=True,
                    )

                # Download button
                with open(out_file, "rb") as f:
                    file_bytes = f.read()

                st.download_button(
                    label=f"⬇️  Download {file_label}",
                    data=file_bytes,
                    file_name=file_label,
                    mime="text/plain",
                    key=f"dl_{i}",
                )
            else:
                st.warning("Output file not found.")

    # Show errors if any
    if err_results:
        st.markdown("---")
        st.markdown("### ❌ Failed Chunks")
        for r in err_results:
            st.error(f"**{r.get('source', '?')}** — {r.get('error', 'unknown error')}")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()