from __future__ import annotations

import json
import re
import zipfile
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pandas as pd
import streamlit as st

# -----------------------------
# Import compiled LangGraph app
# -----------------------------
from bwa_backend import app


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="BlogCraft AI | LangGraph Writer",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------
# Professional styling
# -----------------------------
st.markdown(
    """
    <style>
        :root {
            --bg: #f7f8fb;
            --card: #ffffff;
            --text: #111827;
            --muted: #6b7280;
            --border: #e5e7eb;
            --primary: #4f46e5;
            --primary-dark: #3730a3;
            --soft-primary: #eef2ff;
            --success: #059669;
            --warning: #d97706;
            --danger: #dc2626;
            --shadow: 0 16px 40px rgba(17, 24, 39, 0.08);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(79, 70, 229, 0.14), transparent 32%),
                radial-gradient(circle at top right, rgba(14, 165, 233, 0.12), transparent 26%),
                var(--bg);
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #111827 0%, #1f2937 100%);
            border-right: 1px solid rgba(255,255,255,0.08);
        }

        section[data-testid="stSidebar"] * {
            color: #f9fafb !important;
        }

        section[data-testid="stSidebar"] .stTextArea textarea,
        section[data-testid="stSidebar"] input,
        section[data-testid="stSidebar"] textarea {
            background: rgba(255,255,255,0.08) !important;
            border: 1px solid rgba(255,255,255,0.14) !important;
            border-radius: 14px !important;
            color: #ffffff !important;
        }

        section[data-testid="stSidebar"] .stDateInput input {
            background: rgba(255,255,255,0.08) !important;
            border-radius: 14px !important;
        }

        div[data-testid="stButton"] button,
        div[data-testid="stDownloadButton"] button {
            border-radius: 14px !important;
            border: 0 !important;
            font-weight: 700 !important;
            box-shadow: 0 10px 24px rgba(79,70,229,0.18);
        }

        div[data-testid="stButton"] button[kind="primary"] {
            background: linear-gradient(135deg, #6366f1, #2563eb) !important;
            color: white !important;
        }

        .hero {
            padding: 34px 36px;
            border-radius: 28px;
            background: linear-gradient(135deg, #111827 0%, #312e81 52%, #1d4ed8 100%);
            color: white;
            box-shadow: var(--shadow);
            margin-bottom: 24px;
            position: relative;
            overflow: hidden;
        }

        .hero:after {
            content: "";
            position: absolute;
            width: 260px;
            height: 260px;
            border-radius: 50%;
            background: rgba(255,255,255,0.10);
            right: -70px;
            top: -90px;
        }

        .hero-eyebrow {
            letter-spacing: 0.16em;
            text-transform: uppercase;
            font-size: 0.78rem;
            color: #c7d2fe;
            font-weight: 800;
            margin-bottom: 10px;
        }

        .hero-title {
            font-size: 2.35rem;
            line-height: 1.12;
            font-weight: 900;
            margin: 0 0 10px 0;
        }

        .hero-subtitle {
            color: #dbeafe;
            font-size: 1.02rem;
            max-width: 760px;
            margin: 0;
        }

        .metric-card {
            background: rgba(255,255,255,0.86);
            border: 1px solid rgba(229,231,235,0.9);
            border-radius: 22px;
            padding: 20px 22px;
            box-shadow: var(--shadow);
            min-height: 108px;
        }

        .metric-label {
            font-size: 0.80rem;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 800;
            margin-bottom: 8px;
        }

        .metric-value {
            color: var(--text);
            font-size: 1.75rem;
            font-weight: 900;
            margin-bottom: 4px;
        }

        .metric-help {
            color: var(--muted);
            font-size: 0.88rem;
        }

        .section-card {
            background: rgba(255,255,255,0.90);
            border: 1px solid rgba(229,231,235,0.92);
            border-radius: 24px;
            padding: 24px;
            box-shadow: var(--shadow);
            margin-bottom: 18px;
        }

        .small-muted {
            color: var(--muted);
            font-size: 0.9rem;
        }

        .pill {
            display: inline-block;
            padding: 6px 11px;
            border-radius: 999px;
            background: #eef2ff;
            color: #3730a3;
            font-size: 0.78rem;
            font-weight: 800;
            margin-right: 6px;
            margin-bottom: 6px;
        }

        .sidebar-brand {
            padding: 18px 0 8px 0;
        }

        .sidebar-brand-title {
            font-size: 1.28rem;
            font-weight: 900;
            color: #ffffff;
            margin-bottom: 4px;
        }

        .sidebar-brand-subtitle {
            font-size: 0.86rem;
            color: #cbd5e1 !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
            background: rgba(255,255,255,0.66);
            padding: 8px;
            border-radius: 18px;
            border: 1px solid rgba(229,231,235,0.85);
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 14px;
            padding: 10px 16px;
            font-weight: 800;
        }

        .stTabs [aria-selected="true"] {
            background: #eef2ff;
            color: #3730a3;
        }

        div[data-testid="stDataFrame"] {
            border-radius: 18px;
            overflow: hidden;
            border: 1px solid #e5e7eb;
        }

        .empty-state {
            text-align: center;
            padding: 72px 32px;
            border: 1px dashed #cbd5e1;
            border-radius: 26px;
            background: rgba(255,255,255,0.72);
        }

        .empty-icon {
            font-size: 2.8rem;
            margin-bottom: 12px;
        }

        .empty-title {
            font-size: 1.4rem;
            font-weight: 900;
            color: #111827;
            margin-bottom: 8px;
        }

        .empty-text {
            color: #6b7280;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Helpers
# -----------------------------
def safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s or "blog"


def normalize_obj(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, list):
        return [normalize_obj(x) for x in obj]
    if isinstance(obj, dict):
        return {k: normalize_obj(v) for k, v in obj.items()}
    return obj


def bundle_zip(md_text: str, md_filename: str, images_dir: Path) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr(md_filename, md_text.encode("utf-8"))
        if images_dir.exists() and images_dir.is_dir():
            for p in images_dir.rglob("*"):
                if p.is_file():
                    z.write(p, arcname=str(p))
    return buf.getvalue()


def images_zip(images_dir: Path) -> Optional[bytes]:
    if not images_dir.exists() or not images_dir.is_dir():
        return None
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in images_dir.rglob("*"):
            if p.is_file():
                z.write(p, arcname=str(p))
    return buf.getvalue()


def try_stream(graph_app, inputs: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    try:
        for step in graph_app.stream(inputs, stream_mode="updates"):
            yield ("updates", step)
        out = graph_app.invoke(inputs)
        yield ("final", out)
        return
    except Exception as exc:
        yield ("log", f"updates stream unavailable: {exc}")

    try:
        for step in graph_app.stream(inputs, stream_mode="values"):
            yield ("values", step)
        out = graph_app.invoke(inputs)
        yield ("final", out)
        return
    except Exception as exc:
        yield ("log", f"values stream unavailable: {exc}")

    out = graph_app.invoke(inputs)
    yield ("final", out)


def extract_latest_state(current_state: Dict[str, Any], step_payload: Any) -> Dict[str, Any]:
    if isinstance(step_payload, dict):
        if len(step_payload) == 1 and isinstance(next(iter(step_payload.values())), dict):
            current_state.update(next(iter(step_payload.values())))
        else:
            current_state.update(step_payload)
    return current_state


_MD_IMG_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")
_CAPTION_LINE_RE = re.compile(r"^\*(?P<cap>.+)\*$")


def _resolve_image_path(src: str) -> Path:
    return Path(src.strip().lstrip("./")).resolve()


def render_markdown_with_local_images(md: str):
    matches = list(_MD_IMG_RE.finditer(md))
    if not matches:
        st.markdown(md, unsafe_allow_html=False)
        return

    parts: List[Tuple[str, str]] = []
    last = 0
    for m in matches:
        before = md[last : m.start()]
        if before:
            parts.append(("md", before))
        parts.append(("img", f"{(m.group('alt') or '').strip()}|||{(m.group('src') or '').strip()}"))
        last = m.end()
    if md[last:]:
        parts.append(("md", md[last:]))

    i = 0
    while i < len(parts):
        kind, payload = parts[i]
        if kind == "md":
            st.markdown(payload, unsafe_allow_html=False)
            i += 1
            continue

        alt, src = payload.split("|||", 1)
        caption = None
        if i + 1 < len(parts) and parts[i + 1][0] == "md":
            nxt = parts[i + 1][1].lstrip()
            if nxt.strip():
                first_line = nxt.splitlines()[0].strip()
                mcap = _CAPTION_LINE_RE.match(first_line)
                if mcap:
                    caption = mcap.group("cap").strip()
                    parts[i + 1] = ("md", "\n".join(nxt.splitlines()[1:]))

        if src.startswith(("http://", "https://")):
            st.image(src, caption=caption or (alt or None), use_container_width=True)
        else:
            img_path = _resolve_image_path(src)
            if img_path.exists():
                st.image(str(img_path), caption=caption or (alt or None), use_container_width=True)
            else:
                st.warning(f"Image not found: `{src}`")
        i += 1


def list_past_blogs() -> List[Path]:
    files = [p for p in Path(".").glob("*.md") if p.is_file()]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files


def read_md_file(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def extract_title_from_md(md: str, fallback: str) -> str:
    for line in md.splitlines():
        if line.startswith("# "):
            return line[2:].strip() or fallback
    return fallback


def get_plan_dict(out: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    plan = out.get("plan")
    if plan is None:
        return None
    return normalize_obj(plan)


def get_blog_title(out: Dict[str, Any]) -> str:
    final_md = out.get("final") or ""
    plan = get_plan_dict(out)
    if plan and plan.get("blog_title"):
        return str(plan["blog_title"])
    return extract_title_from_md(final_md, "blog")


def render_metric(label: str, value: Any, help_text: str = ""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-help">{help_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state():
    st.markdown(
        """
        <div class="empty-state">
            <div class="empty-icon">✨</div>
            <div class="empty-title">Ready to generate a polished technical blog</div>
            <div class="empty-text">Enter a topic in the sidebar, choose the as-of date, and run your LangGraph writing agent.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# Session state
# -----------------------------
if "last_out" not in st.session_state:
    st.session_state["last_out"] = None
if "logs" not in st.session_state:
    st.session_state["logs"] = []
if "loaded_blog_name" not in st.session_state:
    st.session_state["loaded_blog_name"] = None


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-title">✍️ BlogCraft AI</div>
            <div class="sidebar-brand-subtitle">LangGraph-powered technical writing workspace</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.subheader("Generate blog")
    topic = st.text_area(
        "Topic",
        height=130,
        placeholder="Example: State of Multimodal LLMs in 2026",
        help="Give a clear topic. The router decides whether fresh research is needed.",
    )
    as_of = st.date_input("As-of date", value=date.today(), help="Used for recency-sensitive topics.")

    run_btn = st.button("🚀 Generate Blog", type="primary", use_container_width=True)

    st.divider()
    st.subheader("Past blogs")
    past_files = list_past_blogs()
    if not past_files:
        st.caption("No saved .md blogs found in this folder.")
    else:
        labels: List[str] = []
        file_by_label: Dict[str, Path] = {}
        for p in past_files[:50]:
            try:
                title = extract_title_from_md(read_md_file(p), p.stem)
            except Exception:
                title = p.stem
            label = f"{title} · {p.name}"
            labels.append(label)
            file_by_label[label] = p

        selected_label = st.selectbox("Load saved blog", labels, label_visibility="collapsed")
        if st.button("📂 Load selected", use_container_width=True):
            selected_path = file_by_label[selected_label]
            md_text = read_md_file(selected_path)
            st.session_state["last_out"] = {
                "plan": None,
                "evidence": [],
                "image_specs": [],
                "final": md_text,
            }
            st.session_state["loaded_blog_name"] = selected_path.name
            st.success("Blog loaded.")

    st.divider()
    st.caption("Tip: Keep your API keys in `.env`; never push them to GitHub.")


# -----------------------------
# Hero
# -----------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-eyebrow">AI Writing Agent</div>
        <div class="hero-title">Generate research-aware technical blogs with a clean professional workflow.</div>
        <p class="hero-subtitle">Plan, evidence, markdown preview, images, downloads, and logs are organized into a focused dashboard for your LangGraph blog writer.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------
# Run graph
# -----------------------------
page_logs: List[str] = []


def log(msg: str):
    page_logs.append(msg)


if run_btn:
    if not topic.strip():
        st.warning("Please enter a topic first.")
        st.stop()

    inputs: Dict[str, Any] = {
        "topic": topic.strip(),
        "mode": "",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "as_of": as_of.isoformat(),
        "recency_days": 7,
        "sections": [],
        "merged_md": "",
        "md_with_placeholders": "",
        "image_specs": [],
        "final": "",
    }

    status = st.status("Running LangGraph workflow…", expanded=True)
    progress_area = st.empty()
    current_state: Dict[str, Any] = {}
    last_node = None

    for kind, payload in try_stream(app, inputs):
        if kind == "log":
            log(str(payload))
            continue

        if kind in ("updates", "values"):
            node_name = None
            if isinstance(payload, dict) and len(payload) == 1 and isinstance(next(iter(payload.values())), dict):
                node_name = next(iter(payload.keys()))

            if node_name and node_name != last_node:
                status.write(f"➡️ `{node_name}` completed")
                last_node = node_name

            current_state = extract_latest_state(current_state, payload)
            plan_obj = current_state.get("plan")
            plan_dict = normalize_obj(plan_obj) if plan_obj else {}
            summary = {
                "mode": current_state.get("mode"),
                "needs_research": current_state.get("needs_research"),
                "queries": current_state.get("queries", [])[:3] if isinstance(current_state.get("queries"), list) else [],
                "evidence": len(current_state.get("evidence", []) or []),
                "tasks": len(plan_dict.get("tasks", []) or []) if isinstance(plan_dict, dict) else None,
                "sections_done": len(current_state.get("sections", []) or []),
                "images": len(current_state.get("image_specs", []) or []),
            }
            progress_area.json(summary)
            log(f"[{kind}] {json.dumps(normalize_obj(payload), default=str)[:1200]}")

        elif kind == "final":
            st.session_state["last_out"] = payload
            st.session_state["loaded_blog_name"] = None
            status.update(label="✅ Blog generated successfully", state="complete", expanded=False)
            log("[final] received final state")

if page_logs:
    st.session_state["logs"].extend(page_logs)


# -----------------------------
# Main dashboard
# -----------------------------
out = st.session_state.get("last_out")

if not out:
    render_empty_state()
    st.stop()

out = normalize_obj(out)
final_md = out.get("final") or ""
plan_dict = get_plan_dict(out) or {}
evidence = normalize_obj(out.get("evidence") or [])
image_specs = normalize_obj(out.get("image_specs") or [])
blog_title = get_blog_title(out)

if st.session_state.get("loaded_blog_name"):
    st.info(f"Loaded saved blog: `{st.session_state['loaded_blog_name']}`")

metric_cols = st.columns(4)
with metric_cols[0]:
    render_metric("Mode", out.get("mode") or "saved", "Research routing result")
with metric_cols[1]:
    render_metric("Sections", len(out.get("sections") or plan_dict.get("tasks", []) or []), "Generated / planned sections")
with metric_cols[2]:
    render_metric("Evidence", len(evidence), "Sources collected")
with metric_cols[3]:
    render_metric("Images", len(image_specs), "Planned/generated visuals")

st.markdown("<br/>", unsafe_allow_html=True)

tab_overview, tab_plan, tab_evidence, tab_preview, tab_images, tab_logs = st.tabs(
    ["🏠 Overview", "🧩 Plan", "🔎 Evidence", "📝 Preview", "🖼️ Images", "🧾 Logs"]
)

with tab_overview:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader(blog_title)
    st.caption("Generated markdown output summary")

    c1, c2 = st.columns([2, 1])
    with c1:
        if final_md:
            preview_text = re.sub(r"\s+", " ", final_md.replace("#", "")).strip()[:650]
            st.write(preview_text + ("..." if len(final_md) > 650 else ""))
        else:
            st.warning("No final markdown found.")
    with c2:
        st.markdown("**Downloads**")
        md_filename = f"{safe_slug(blog_title)}.md"
        st.download_button(
            "⬇️ Markdown",
            data=final_md.encode("utf-8"),
            file_name=md_filename,
            mime="text/markdown",
            use_container_width=True,
        )
        st.download_button(
            "📦 Bundle",
            data=bundle_zip(final_md, md_filename, Path("images")),
            file_name=f"{safe_slug(blog_title)}_bundle.zip",
            mime="application/zip",
            use_container_width=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

with tab_plan:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Content plan")
    if not plan_dict:
        st.info("No plan found. Saved blogs usually only contain markdown output.")
    else:
        st.markdown(
            f"""
            <span class="pill">Audience: {plan_dict.get('audience')}</span>
            <span class="pill">Tone: {plan_dict.get('tone')}</span>
            <span class="pill">Kind: {plan_dict.get('blog_kind')}</span>
            """,
            unsafe_allow_html=True,
        )
        tasks = plan_dict.get("tasks", []) or []
        if tasks:
            df = pd.DataFrame(
                [
                    {
                        "ID": t.get("id"),
                        "Section": t.get("title"),
                        "Words": t.get("target_words"),
                        "Research": t.get("requires_research"),
                        "Citations": t.get("requires_citations"),
                        "Code": t.get("requires_code"),
                        "Tags": ", ".join(t.get("tags") or []),
                    }
                    for t in tasks
                ]
            ).sort_values("ID")
            st.dataframe(df, use_container_width=True, hide_index=True)
            with st.expander("View full task details"):
                st.json(tasks)
    st.markdown('</div>', unsafe_allow_html=True)

with tab_evidence:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Evidence sources")
    if not evidence:
        st.info("No evidence returned. This can happen for closed-book topics or when Tavily has no results/API key.")
    else:
        rows = []
        for e in evidence:
            rows.append(
                {
                    "Title": e.get("title"),
                    "Published": e.get("published_at"),
                    "Source": e.get("source"),
                    "URL": e.get("url"),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.markdown('</div>', unsafe_allow_html=True)

with tab_preview:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Markdown preview")
    if not final_md:
        st.warning("No final markdown found.")
    else:
        render_markdown_with_local_images(final_md)
    st.markdown('</div>', unsafe_allow_html=True)

with tab_images:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Images")
    images_dir = Path("images")
    if image_specs:
        with st.expander("Image plan"):
            st.json(image_specs)

    if images_dir.exists():
        files = sorted([p for p in images_dir.iterdir() if p.is_file()])
        if files:
            cols = st.columns(2)
            for idx, p in enumerate(files):
                with cols[idx % 2]:
                    st.image(str(p), caption=p.name, use_container_width=True)
            z = images_zip(images_dir)
            if z:
                st.download_button(
                    "⬇️ Download Images ZIP",
                    data=z,
                    file_name="images.zip",
                    mime="application/zip",
                    use_container_width=True,
                )
        else:
            st.info("images/ exists but is empty.")
    else:
        st.info("No images generated for this blog.")
    st.markdown('</div>', unsafe_allow_html=True)

with tab_logs:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Execution logs")
    st.text_area("Latest events", value="\n\n".join(st.session_state["logs"][-80:]), height=520)
    if st.button("Clear logs"):
        st.session_state["logs"] = []
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
