"""
PromptGuard — Interactive Dashboard
Visualises evaluation results. Run: ``streamlit run dashboard/app.py``
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from attacks.templates import ATTACK_TEMPLATES
from pipelines.openai_pipeline import build_mock_pipeline
from evaluation.runner import evaluate_pipeline, generate_report
from detector.guard import build_guarded_pipeline

st.set_page_config(
    page_title="PromptGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .metric-card { background: #1e1e2e; border-radius: 12px; padding: 20px; border: 1px solid #313244; }
    .stMetric > div { background: #1e1e2e; border-radius: 8px; padding: 12px; }
    header { background: transparent !important; }
</style>
""",
    unsafe_allow_html=True,
)

st.sidebar.title("Configuration")
pipeline_option = st.sidebar.selectbox(
    "Target Pipeline",
    [
        "Mock (Low Vulnerability)",
        "Mock (High Vulnerability)",
        "OpenAI GPT-4o-mini (requires key)",
    ],
)

sensitivity = st.sidebar.select_slider(
    "Guard Sensitivity",
    options=["low", "medium", "high"],
    value="medium",
)

run_mutation = st.sidebar.checkbox("Enable Mutation Engine", value=False)

st.sidebar.markdown("---")
attack_types = sorted(set(a["type"] for a in ATTACK_TEMPLATES))
selected_types = st.sidebar.multiselect(
    "Filter Attack Types",
    options=attack_types,
    default=attack_types,
)

uploaded = st.sidebar.file_uploader("Load JSON report (optional)", type=["json"])
default_report_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "promptguard_report.json"
)
if st.sidebar.button("Load default report file"):
    if os.path.isfile(default_report_path):
        with open(default_report_path, encoding="utf-8") as f:
            st.session_state.report = json.load(f)
        st.sidebar.success(f"Loaded {os.path.basename(default_report_path)}")
    else:
        st.sidebar.warning("File not found. Run main.py first.")

run_btn = st.sidebar.button("Run Evaluation", type="primary", use_container_width=True)

st.title("PromptGuard")
st.markdown("**LLM prompt injection detection and evaluation**")
st.markdown("---")

if "report" not in st.session_state:
    st.session_state.report = None


def _blocked_adversarial(results: list[dict]) -> int:
    return sum(
        1
        for r in results
        if r.get("blocked_by_guard") and r.get("attack_type") != "benign"
    )


def _render_report(report: dict) -> None:
    bm = report["baseline"]["metrics"]
    gm = report["with_guard"]["metrics"]
    imp = report.get("improvement", {})
    g_results = report["with_guard"]["results"]

    bench_models = report.get("benchmark", {}).get("models")
    if bench_models:
        st.subheader("Multi-model benchmark")
        rows = []
        for name, payload in bench_models.items():
            if "baseline" not in payload:
                continue
            b = payload["baseline"]["metrics"]
            g = payload.get("with_guard", {}).get("metrics", {})
            rows.append(
                {
                    "Target": name,
                    "Baseline ASR": f"{b.get('asr', 0)*100:.1f}%",
                    "Guarded ASR": f"{g.get('asr', 0)*100:.1f}%",
                    "p95 latency (ms)": b.get("latency_summary_ms", {}).get("p95_ms", ""),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.subheader("Evaluation Summary")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("Baseline ASR", f"{bm['asr']*100:.1f}%")
    with col2:
        st.metric(
            "Guarded ASR",
            f"{gm['asr']*100:.1f}%",
            delta=f"-{imp.get('asr_reduction_percent', 0)}%",
            delta_color="inverse",
        )
    with col3:
        st.metric("Adv. blocked by guard", _blocked_adversarial(g_results))
    with col4:
        st.metric("Benign FPR (guard)", f"{gm.get('fpr_guard_benign_block', 0)*100:.1f}%")
    with col5:
        st.metric("Total adversarial", bm["total_attacks"])
    with col6:
        st.metric("Avg latency", f"{bm['avg_latency_ms']}ms")

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Baseline vs guarded — ASR by type")
        types = sorted(bm["per_type"].keys())
        baseline_asrs = [bm["per_type"][t]["asr"] * 100 for t in types]
        guard_asrs = [gm["per_type"].get(t, {}).get("asr", 0) * 100 for t in types]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(name="Baseline", x=types, y=baseline_asrs, marker_color="#f38ba8")
        )
        fig.add_trace(
            go.Bar(name="Guarded", x=types, y=guard_asrs, marker_color="#a6e3a1")
        )
        fig.update_layout(
            barmode="group",
            plot_bgcolor="#1e1e2e",
            paper_bgcolor="#1e1e2e",
            font_color="#cdd6f4",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            yaxis_title="ASR (%)",
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Guarded ASR gauge")
        fig2 = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=gm["asr"] * 100,
                delta={"reference": bm["asr"] * 100, "valueformat": ".1f", "suffix": "%"},
                title={"text": "Guarded ASR (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#a6e3a1"},
                    "steps": [
                        {"range": [0, 20], "color": "#313244"},
                        {"range": [20, 50], "color": "#45475a"},
                        {"range": [50, 100], "color": "#585b70"},
                    ],
                    "threshold": {
                        "line": {"color": "#f38ba8", "width": 4},
                        "thickness": 0.75,
                        "value": bm["asr"] * 100,
                    },
                },
            )
        )
        fig2.update_layout(
            plot_bgcolor="#1e1e2e",
            paper_bgcolor="#1e1e2e",
            font_color="#cdd6f4",
            height=350,
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Attack success heatmap (baseline)")
    heatmap_data: dict = {}
    for result in report["baseline"]["results"]:
        atype = result["attack_type"]
        sev = result["severity"]
        heatmap_data.setdefault(atype, {}).setdefault(sev, {"success": 0, "total": 0})
        heatmap_data[atype][sev]["total"] += 1
        if result["success"]:
            heatmap_data[atype][sev]["success"] += 1

    severities_order = ["high", "medium", "low", "none"]
    atypes = [t for t in heatmap_data if t != "benign"]
    sevs = [s for s in severities_order if any(s in heatmap_data.get(t, {}) for t in atypes)]

    z_vals = []
    for sev in sevs:
        row = []
        for atype in atypes:
            d = heatmap_data.get(atype, {}).get(sev, {"success": 0, "total": 1})
            row.append(round(d["success"] / max(d["total"], 1) * 100, 1))
        z_vals.append(row)

    fig3 = go.Figure(
        go.Heatmap(
            z=z_vals,
            x=atypes,
            y=sevs,
            colorscale="RdYlGn_r",
            zmin=0,
            zmax=100,
            text=[[f"{v}%" for v in row] for row in z_vals],
            texttemplate="%{text}",
            colorbar=dict(title="ASR (%)"),
        )
    )
    fig3.update_layout(
        plot_bgcolor="#1e1e2e",
        paper_bgcolor="#1e1e2e",
        font_color="#cdd6f4",
        height=280,
        xaxis_title="Attack type",
        yaxis_title="Severity",
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Detailed results")
    tab1, tab2 = st.tabs(["Baseline", "Guarded"])
    for tab, key, _label in [
        (tab1, "baseline", "Baseline"),
        (tab2, "with_guard", "Guarded"),
    ]:
        with tab:
            rows = []
            for r in report[key]["results"]:
                rows.append(
                    {
                        "Attack": r["attack_name"],
                        "Type": r["attack_type"],
                        "Severity": r["severity"],
                        "Result": "Compromised" if r["success"] else "Defended",
                        "Blocked": r.get("blocked_by_guard", False),
                        "Latency (ms)": r["latency_ms"],
                        "Output preview": (r.get("output", "")[:120] + "...")
                        if len(r.get("output", "")) > 120
                        else r.get("output", ""),
                    }
                )
            st.dataframe(pd.DataFrame(rows), use_container_width=True, height=380)

    st.markdown("---")
    st.download_button(
        label="Download JSON report",
        data=json.dumps(report, indent=2),
        file_name=f"promptguard_report_{report['report_id']}.json",
        mime="application/json",
    )


if uploaded is not None and not run_btn:
    try:
        st.session_state.report = json.loads(uploaded.getvalue().decode("utf-8"))
        st.sidebar.success("Uploaded report applied.")
    except Exception as e:
        st.sidebar.error(f"Invalid JSON: {e}")

if run_btn:
    vuln_level = 0.2 if "Low" in pipeline_option else 0.65
    if "OpenAI" in pipeline_option:
        try:
            from pipelines.openai_pipeline import build_openai_pipeline

            pipeline = build_openai_pipeline(
                system_prompt="You are a helpful assistant for a financial services company.",
                model="gpt-4o-mini",
            )
        except Exception as e:
            st.error(f"OpenAI pipeline failed ({e}). Falling back to mock.")
            pipeline = build_mock_pipeline(vulnerability_level=vuln_level)
    else:
        pipeline = build_mock_pipeline(vulnerability_level=vuln_level)

    attacks = [a for a in ATTACK_TEMPLATES if a["type"] in selected_types]

    if run_mutation:
        from attacks.generator import run_mutation_loop
        from attacks.templates import get_adversarial_attacks

        seeds = get_adversarial_attacks()[:3]

        def _eval(atks):
            return evaluate_pipeline(pipeline, atks, verbose=False)

        extra = run_mutation_loop(
            seed_attacks=seeds,
            evaluate_fn=_eval,
            llm_client=None,
            n_iterations=1,
            n_variants_per_seed=2,
        )
        seen = {a["prompt"] for a in attacks}
        for a in extra:
            if a.get("prompt") and a["prompt"] not in seen:
                seen.add(a["prompt"])
                attacks.append(a)

    with st.spinner("Running baseline evaluation..."):
        baseline_results = evaluate_pipeline(pipeline, attacks, verbose=False)
    with st.spinner(f"Running guarded evaluation (sensitivity={sensitivity})..."):
        guarded_pipeline = build_guarded_pipeline(pipeline, sensitivity=sensitivity)
        guard_results = evaluate_pipeline(guarded_pipeline, attacks, verbose=False)

    report = generate_report(
        results=baseline_results,
        pipeline_name=getattr(pipeline, "__name__", "pipeline"),
        guard_results=guard_results,
        output_path=None,
        sensitivity=sensitivity,
        mutation_enabled=run_mutation,
    )
    st.session_state.report = report
    st.success("Evaluation complete.")

if st.session_state.report:
    report = st.session_state.report
    if "with_guard" not in report:
        st.error("This report has no `with_guard` section. Re-run `python main.py` to regenerate.")
        st.json(report)
    else:
        _render_report(report)
else:
    st.info("Upload a report or run an evaluation from the sidebar.")
