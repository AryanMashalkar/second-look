"""Second Look: a second-opinion desk for loan applicants with no credit bureau record."""

import json
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from secondlook import config, ui
from secondlook.features import add_ratios
from secondlook.model import Baseline, SecondLook

st.set_page_config(page_title="Second Look", page_icon=":material/account_balance:", layout="wide")
st.markdown(ui.CSS, unsafe_allow_html=True)
ui.register_plotly_theme()


@st.cache_resource
def load_models():
    return SecondLook.load(), Baseline.load()


@st.cache_data
def load_artifacts():
    a = config.ARTIFACTS
    return (
        json.loads((a / "metrics.json").read_text()),
        pd.read_csv(a / "demo_profiles.csv"),
        pd.read_csv(a / "policy_curves.csv"),
        pd.read_csv(a / "fairness.csv"),
    )


model, baseline = load_models()
metrics, profiles, curves, fairness = load_artifacts()
nb = metrics["no_bureau"]
policy = nb["policy_holdout"]["second_look"]  # cut-offs fixed on validation, outcomes measured on test
first = metrics["first_time"]


def first_time_note(row) -> str:
    """Second Look adds nothing measurable for applicants with no history at this lender, so say so on the case."""
    if row["PREV_COUNT"][0] > 0:
        return ""
    fa = first["auc"]
    return (f'<p class="sl-note"><b>First-time borrower:</b> no history with this lender either. On such applicants Second Look '
            f'ranks no better than the external score (AUC {fa["second_look"]["auc"]:.2f} vs {fa["baseline"]["auc"]:.2f}), '
            "so treat this as a standard review.</p>")
MODEL_NAMES = {
    "baseline": "External credit score",
    "application_only": "Second Look without lender history",
    "second_look": "Second Look",
    "combined": "Second Look + external score",
}
QUEUE_GROUPS = {
    "Second look: approve": "Score declines, Second Look approves",
    "Second look: caution": "Score approves, Second Look cautions",
    "Both approve": "Both approve",
    "Both decline": "Both decline",
}
if "decisions" not in st.session_state:
    st.session_state.decisions = []


def fmt(feature, value):
    """Human-readable value for a feature."""
    if pd.isna(value):
        return "not provided"
    if feature.startswith("AMT_"):
        return f"{value:,.0f}"
    if feature in ("DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "DAYS_LAST_PHONE_CHANGE", "PREV_DAYS_LAST_DECISION"):
        return f"{-value / 365.25:.1f} years"
    if feature.endswith("_SHARE"):
        return f"{value:.0%}"
    if feature == "TERM_MONTHS":
        return f"{value:.0f} months"
    if feature in ("CREDIT_TO_INCOME", "GOODS_TO_CREDIT", "PREV_APPLIED_TO_GRANTED"):
        return f"{value:.2f}x"
    if feature == "ANNUITY_TO_INCOME":
        return f"{value:.0%} of income"
    if feature == "INST_MEAN_DAYS_LATE":
        return f"{value:.1f} days"
    if feature.startswith("FLAG_") or feature in ("EMPLOYED_YEARS_MISSING", "REG_CITY_NOT_WORK_CITY"):
        return "yes" if value in (1, "Y") else "no"
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def score(row):
    rb, rs = float(baseline.risk(row)[0]), float(model.risk(row)[0])
    return rb, rb <= baseline.threshold, rs, rs <= model.threshold


def case_facts(row):
    r = row.iloc[0]
    job = "not employed or pensioner" if r["EMPLOYED_YEARS_MISSING"] == 1 else fmt("DAYS_EMPLOYED", r["DAYS_EMPLOYED"])
    return [
        ("Loan requested", fmt("AMT_CREDIT", r["AMT_CREDIT"])),
        ("Declared income", fmt("AMT_INCOME_TOTAL", r["AMT_INCOME_TOTAL"])),
        ("Monthly repayment", fmt("AMT_ANNUITY", r["AMT_ANNUITY"])),
        ("Repayment burden", fmt("ANNUITY_TO_INCOME", r["ANNUITY_TO_INCOME"])),
        ("Term", fmt("TERM_MONTHS", r["TERM_MONTHS"])),
        ("Time in job", job),
        ("Occupation", fmt("OCCUPATION_TYPE", r["OCCUPATION_TYPE"])),
        ("Education", fmt("NAME_EDUCATION_TYPE", r["NAME_EDUCATION_TYPE"])),
        ("Instalments paid to this lender", f"{r['INST_COUNT']:.0f}"),
        ("Of those, paid late", fmt("INST_LATE_SHARE", r["INST_LATE_SHARE"])),
    ]


def reasons_chart(row, key):
    reasons = model.reasons(row)
    reasons["shown"] = [f"{l}: {fmt(f, v)}" for l, f, v in zip(reasons["label"], reasons["feature"], reasons["value"])]
    reasons = reasons.sort_values("contribution")
    fig = go.Figure(go.Bar(
        x=reasons["contribution"], y=reasons["shown"], orientation="h",
        marker_color=[ui.APPROVE if c < 0 else ui.DECLINE for c in reasons["contribution"]],
        hovertemplate="%{y}<br>effect %{x:+.2f}<extra></extra>",
    ))
    fig.update_layout(height=330, xaxis_title="Pushes risk down ←   → pushes risk up", showlegend=False,
                      margin=dict(l=10, r=10, t=10, b=40))
    st.plotly_chart(fig, key=key)


def offer_slip(row):
    rc = model.recourse(row)
    if rc is None:
        st.markdown('<div class="sl-slip clear">Within Second Look\'s cut-off as applied for. No counter-offer needed.</div>',
                    unsafe_allow_html=True)
    elif rc["found"]:
        parts = []
        if rc["amount_factor"] < 1:
            parts.append(f"a loan of <b>{rc['new_credit']:,.0f}</b> instead of {float(row['AMT_CREDIT'].iloc[0]):,.0f}")
        if rc["term_factor"] != 1:
            parts.append(f"a <b>{rc['new_term_months']:.0f}-month</b> term (repayment {rc['new_annuity']:,.0f} a month)")
        st.markdown(
            f'<div class="sl-slip"><b>Counter-offer:</b> {" and ".join(parts)} brings risk to <b>{rc["new_risk"]:.1%}</b>, '
            "inside the cut-off.<br><span class='sl-note'>Only loan terms change, never who the applicant is. "
            "This describes what the model responds to, not a promise of repayment.</span></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="sl-slip none">No loan amount or term within ±50% brings this applicant inside the cut-off.</div>',
                    unsafe_allow_html=True)


# ---------------------------------------------------------------- masthead
st.markdown(
    f"""
<div class="sl-mast">
  <div>
    <h1>Second Look</h1>
    <p>One in seven applicants has no credit bureau record, and the external credit score judges them worst.
    Second Look re-reads their file from the application and the lender's own repayment records,
    explains itself, and leaves the decision to a loan officer.</p>
  </div>
  {ui.ledger([
      ("Ranking quality (AUC), vs external score", f"{nb['auc']['second_look']['auc']:.3f} vs {nb['auc']['baseline']['auc']:.3f}"),
      ("Extra approvals per 10,000, no higher default rate", f"+{policy['extra_approved_per_10k']:,.0f}"),
      ("Of whom actually repaid", f"+{policy['extra_repayers_per_10k']:,.0f}"),
      ("Real test applicants without a bureau record", f"{metrics['split']['test_no_bureau']:,}"),
  ])}
</div>
""",
    unsafe_allow_html=True,
)

desk, trial, evidence, fair_tab, how = st.tabs(["Case desk", "Try an applicant", "Evidence", "Fairness", "How it works"])

# ---------------------------------------------------------------- case desk
with desk:
    st.markdown(
        '<p class="sl-note">Cases are <b>synthetic</b>: each is stitched from parts of different training applicants and '
        "jittered, because the Home Credit data may not be redistributed. Every figure in Evidence and Fairness comes from real "
        "held-out applicants.</p>",
        unsafe_allow_html=True,
    )
    left, right = st.columns([4, 7], gap="large")
    with left:
        st.subheader("Review queue")
        groups = st.pills("Show", list(QUEUE_GROUPS), default=list(QUEUE_GROUPS)[:2], selection_mode="multi",
                          format_func=QUEUE_GROUPS.get, label_visibility="collapsed")
        view = profiles[profiles["queue_reason"].isin(groups or [])].copy()
        view["_order"] = view["queue_reason"].map(list(QUEUE_GROUPS).index)
        view = view.sort_values(["_order", "profile_id"])
        if view.empty:
            st.info("Pick at least one group above to fill the queue.")
            st.stop()
        short = {"Second look: approve": "flips to approve", "Second look: caution": "flips to caution",
                 "Both approve": "both approve", "Both decline": "both decline"}
        labels = {r.profile_id: f"**{r.profile_id}**  loan {r.AMT_CREDIT:,.0f}, {short[r.queue_reason]}" for r in view.itertuples()}
        with st.container(height=560, border=True):
            pid = st.radio("Open case", list(labels), format_func=labels.get, label_visibility="collapsed")
        done = {d["profile"] for d in st.session_state.decisions}
        st.caption(f"{len(done)} of {len(view)} cases decided this session.")

    with right:
        row = profiles[profiles["profile_id"] == pid].reset_index(drop=True)
        rb, ok_b, rs, ok_s = score(row)
        st.markdown(
            f"""<div class="sl-case"><h3>Case {ui.esc(pid)}</h3>
<div class="sub">{ui.esc(QUEUE_GROUPS[row['queue_reason'][0]])}. No credit bureau record.</div>
{ui.facts(case_facts(row))}
{ui.stamp_pair(rb, ok_b, rs, ok_s, baseline.threshold, model.threshold, fresh=True)}
{first_time_note(row)}</div>""",
            unsafe_allow_html=True,
        )
        file_tab, terms_tab, decide_tab = st.tabs(["Why", "Counter-offer and what-if", "Decide"])
        with file_tab:
            st.markdown('<p class="sl-note">Each bar is the exact contribution of one factor to Second Look\'s score '
                        "(TreeSHAP). Green lowers the risk, red raises it.</p>", unsafe_allow_html=True)
            reasons_chart(row, key=f"reasons_{pid}")
            with st.expander("Every field in this case"):
                st.table(pd.DataFrame([(config.LABELS.get(f, f), fmt(f, row[f][0])) for f in config.SECOND_LOOK_FEATURES],
                                      columns=["Field", "Value"]).set_index("Field"))
        with terms_tab:
            offer_slip(row)
            st.markdown("**Try other terms**")
            c1, c2 = st.columns(2)
            amount = c1.slider("Loan amount, % of requested", 40, 120, 100, 5, key=f"amt_{pid}")
            term = c2.slider("Term, % of requested", 50, 200, 100, 5, key=f"term_{pid}")
            what_if = row.copy()
            what_if["AMT_CREDIT"] = row["AMT_CREDIT"] * amount / 100
            what_if["AMT_ANNUITY"] = row["AMT_ANNUITY"] * (amount / 100) / (term / 100)
            add_ratios(what_if)
            r_new = float(model.risk(what_if)[0])
            ok_new = r_new <= model.threshold
            st.markdown(
                f"""<div class="sl-case" style="border-left-color:{ui.APPROVE if ok_new else ui.DECLINE}">
{ui.facts([("Loan", f"{float(what_if['AMT_CREDIT'][0]):,.0f}"), ("Term", f"{float(what_if['TERM_MONTHS'][0]):.0f} months"),
           ("Monthly repayment", f"{float(what_if['AMT_ANNUITY'][0]):,.0f}"), ("Second Look risk", f"{r_new:.1%}")])}
<div class="sl-stamps"><div class="sl-stampbox"><div class="who">Second Look on these terms</div>{ui.stamp(ok_new)}</div></div></div>""",
                unsafe_allow_html=True,
            )
        with decide_tab:
            with st.form(f"decide_{pid}", clear_on_submit=True):
                choice = st.radio("Decision", ["Approve", "Approve with counter-offer", "Request documents", "Decline"], horizontal=True)
                note = st.text_input("Reason (required, shared with the applicant if declined)")
                if st.form_submit_button("Record decision", type="primary"):
                    if not note.strip():
                        st.error("Write a reason before recording a decision.")
                    else:
                        st.session_state.decisions.append({
                            "time": datetime.now().strftime("%H:%M:%S"), "case": pid, "profile": pid, "decision": choice,
                            "reason": note.strip(), "external_risk": round(rb, 4), "second_look_risk": round(rs, 4),
                        })
                        st.success(f"Recorded: {choice} for {pid}.")
            if st.session_state.decisions:
                log = pd.DataFrame(st.session_state.decisions).drop(columns="profile")
                st.table(log.set_index(log.columns[0]))
                st.download_button("Download decision log (CSV)", log.to_csv(index=False), "decision_log.csv", "text/csv")
            else:
                st.caption("Decisions you record appear here and can be downloaded as a CSV log.")

# ---------------------------------------------------------------- try an applicant
with trial:
    st.subheader("Score an applicant of your own")
    st.markdown('<p class="sl-note">Fill in the form and both decisions update as you type. Fields not shown take '
                "typical values for no-bureau applicants.</p>", unsafe_allow_html=True)
    template = profiles.iloc[[0]].copy().reset_index(drop=True)
    for col in config.SECOND_LOOK_FEATURES + config.EXTERNAL:
        if col in config.APP_CATEGORICAL:
            template[col] = profiles[col].mode()[0]
        else:
            template[col] = profiles[col].median()

    form_col, out_col = st.columns([5, 6], gap="large")
    with form_col:
        st.markdown("**The loan**")
        a1, a2 = st.columns(2)
        loan = a1.number_input("Loan amount", 25_000, 4_000_000, 450_000, 25_000)
        months = a2.slider("Term (months)", 6, 60, 20)
        income = a1.number_input("Declared income (same period as repayments)", 10_000, 2_000_000, 150_000, 10_000)
        contract = a2.selectbox("Loan type", model.categories["NAME_CONTRACT_TYPE"])
        st.markdown("**The person**")
        b1, b2 = st.columns(2)
        no_job = b2.checkbox("Pensioner or not employed")
        job_years = b1.slider("Years in current job", 0.0, 30.0, 4.0, 0.5, disabled=no_job)
        education = b1.selectbox("Education", model.categories["NAME_EDUCATION_TYPE"],
                                 index=model.categories["NAME_EDUCATION_TYPE"].index("Secondary / secondary special"))
        occupation = b2.selectbox("Occupation", model.categories["OCCUPATION_TYPE"],
                                  index=model.categories["OCCUPATION_TYPE"].index("Laborers"))
        st.markdown("**History with this lender**")
        c1, c2 = st.columns(2)
        paid = c1.slider("Instalments paid on earlier loans", 0, 150, 30)
        late = c2.slider("Share of them paid late (%)", 0, 100, 5, disabled=paid == 0)
        refused = c1.slider("Earlier applications refused (%)", 0, 100, 0)
        ext2 = c2.slider("External credit score (0 = worst, 1 = best)", 0.0, 1.0, 0.5, 0.01)

    app = template.copy()
    app["AMT_CREDIT"], app["AMT_INCOME_TOTAL"] = float(loan), float(income)
    app["AMT_ANNUITY"], app["AMT_GOODS_PRICE"] = loan / months, loan * 0.9
    app["NAME_CONTRACT_TYPE"], app["NAME_EDUCATION_TYPE"], app["OCCUPATION_TYPE"] = contract, education, occupation
    app["EMPLOYED_YEARS_MISSING"] = int(no_job)
    app["DAYS_EMPLOYED"] = np.nan if no_job else -job_years * 365.25
    app["NAME_INCOME_TYPE"] = "Pensioner" if no_job else template["NAME_INCOME_TYPE"][0]
    app["INST_COUNT"] = paid
    app["INST_LATE_SHARE"] = late / 100 if paid else np.nan
    app["INST_MEAN_DAYS_LATE"] = (late / 100) * 8 if paid else np.nan
    app["INST_UNDERPAID_SHARE"] = 0.0 if paid else np.nan
    app["PREV_COUNT"] = max(1, round(paid / 12)) if paid or refused else 0
    app["PREV_REFUSED_SHARE"] = refused / 100 if app["PREV_COUNT"][0] else np.nan
    app["PREV_REFUSED"] = round(app["PREV_COUNT"][0] * refused / 100)
    app["PREV_APPROVED"] = app["PREV_COUNT"] - app["PREV_REFUSED"]
    app["EXT_SOURCE_1"], app["EXT_SOURCE_2"], app["EXT_SOURCE_3"] = np.nan, ext2, np.nan
    add_ratios(app)

    with out_col:
        rb, ok_b, rs, ok_s = score(app)
        st.markdown(
            f"""<div class="sl-case"><h3>Your applicant</h3>
<div class="sub">Loan {loan:,.0f} over {months} months, repayment {loan / months:,.0f} a month
({loan / months / income:.0%} of declared income).</div>
{ui.stamp_pair(rb, ok_b, rs, ok_s, baseline.threshold, model.threshold)}
{first_time_note(app)}</div>""",
            unsafe_allow_html=True,
        )
        reasons_chart(app, key="reasons_trial")
        offer_slip(app)

# ---------------------------------------------------------------- evidence
with evidence:
    a = metrics["audit"]
    st.markdown(
        f"Trained and tested on **Home Credit Default Risk** ({a['applicants']:,} real loans). "
        f"**{a['no_bureau_applicants']:,} applicants ({a['no_bureau_share']:.1%}) have no credit bureau record** and default more often "
        f"({a['no_bureau_default_rate']:.1%} vs {a['bureau_default_rate']:.1%}). Everything here is measured on a 20% test split "
        f"({metrics['split']['test']:,} applicants, {metrics['split']['test_no_bureau']:,} without a bureau record) that no model saw in training."
    )
    st.subheader("Set your own lending policy")
    rate = st.slider("Approve this share of no-bureau applicants", 50, 99, 80, 1, format="%d%%")
    c = curves[curves["population"] == "no_bureau"]
    pick = c.iloc[(c["approval_rate"] - rate / 100).abs().groupby(c["model"]).idxmin()].set_index("model")
    n_test = metrics["split"]["test_no_bureau"]
    b_bad, s_bad = pick.loc["baseline", "bad_rate"], pick.loc["second_look", "bad_rate"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Defaults among approved, external score", f"{b_bad:.1%}")
    m2.metric("Defaults among approved, Second Look", f"{s_bad:.1%}", f"{(s_bad - b_bad) * 100:+.1f} points", delta_color="inverse")
    m3.metric("Fewer defaulted loans per 10,000 applicants", f"{(b_bad - s_bad) * rate / 100 * 10_000:,.0f}")
    cc = c[c["model"].isin(["baseline", "second_look"])].copy()
    cc["model"] = cc["model"].map(MODEL_NAMES)
    fig = px.line(cc, x="approval_rate", y="bad_rate", color="model",
                  color_discrete_map={MODEL_NAMES["baseline"]: ui.BASELINE, MODEL_NAMES["second_look"]: ui.APPROVE},
                  labels={"approval_rate": "Share of applicants approved", "bad_rate": "Default rate among approved", "model": ""})
    fig.add_vline(x=rate / 100, line_dash="dot", line_color=ui.INK)
    fig.update_traces(line_width=3)
    fig.update_layout(height=380, yaxis_tickformat=".0%", xaxis_tickformat=".0%")
    st.plotly_chart(fig, key="policy")
    st.markdown(
        "The headline cut-offs are chosen on a separate validation split, never on these test applicants: the external score "
        "approves 80%, and Second Look the largest share with no higher default rate. Applied to the test applicants, the external "
        f"score approves {policy['baseline_approval_rate']:.1%} (defaults {policy['baseline_bad_rate']:.2%}) and Second Look "
        f"**{policy['model_approval_rate']:.1%}** (defaults {policy['model_bad_rate']:.2%}): "
        f"**{policy['extra_approved_per_10k']:,.0f} more people per 10,000** "
        f"(95% CI {policy['extra_approved_ci'][0]:,.0f} to {policy['extra_approved_ci'][1]:,.0f}), "
        f"of whom **{policy['extra_repayers_per_10k']:,.0f} repaid** ({policy['extra_repayers_ci'][0]:,.0f} to {policy['extra_repayers_ci'][1]:,.0f})."
    )

    st.subheader("Ranking quality")
    rows = []
    for key, label in MODEL_NAMES.items():
        for pop, pop_label in (("no_bureau", "No bureau record"), ("all", "Everyone")):
            v = metrics[pop]["auc"][key]
            d = v.get("delta_vs_baseline")
            rows.append({"Model": label, "Applicants": pop_label, "AUC": v["auc"], "low": v["ci_low"], "high": v["ci_high"],
                         "Gain vs external score (95% CI)": f"{d['mean']:+.3f} ({d['ci_low']:+.3f} to {d['ci_high']:+.3f})" if d else "reference"})
    auc = pd.DataFrame(rows)
    fig = go.Figure()
    for pop_label, color in (("No bureau record", ui.APPROVE), ("Everyone", ui.BASELINE)):
        part = auc[auc["Applicants"] == pop_label]
        fig.add_trace(go.Scatter(x=part["AUC"], y=part["Model"], mode="markers", name=pop_label, marker=dict(size=13, color=color),
                                 error_x=dict(type="data", symmetric=False, array=part["high"] - part["AUC"],
                                              arrayminus=part["AUC"] - part["low"], color=color, thickness=2)))
    fig.update_layout(height=300, xaxis_title="AUC with 95% bootstrap interval (higher ranks risk better)")
    st.plotly_chart(fig, key="auc")
    st.table(auc[["Model", "Applicants", "AUC", "Gain vs external score (95% CI)"]].set_index("Model").style.format({"AUC": "{:.3f}"}))
    st.caption(
        f"The ablation shows where the gain comes from: without the lender's own history the gain falls from "
        f"{nb['auc']['second_look']['delta_vs_baseline']['mean']:+.3f} to {nb['auc']['application_only']['delta_vs_baseline']['mean']:+.3f}. "
        f"{a['no_bureau_with_lender_history']:.0%} of no-bureau applicants had applied to this lender before."
    )

    st.subheader("Who it helps, and who it doesn't")
    fa, fp = first["auc"], first["policy_holdout"]
    st.markdown(
        f"The gain comes from the lender's own records. **{a['no_bureau_with_lender_history']:.0%} of no-bureau applicants** had "
        "applied here before, and for them Second Look reads repayment behaviour the external score never sees. "
        f"For the **{first['n']:,} true first-time borrowers** in the test set (no bureau record *and* no history with this lender), "
        f"it ranks no better: AUC {fa['second_look']['auc']:.3f} vs {fa['baseline']['auc']:.3f} "
        f"(difference {fa['second_look']['delta_vs_baseline']['mean']:+.3f}, 95% CI {fa['second_look']['delta_vs_baseline']['ci_low']:+.3f} "
        f"to {fa['second_look']['delta_vs_baseline']['ci_high']:+.3f}), and its cut-off approves more of them "
        f"({fp['model_approval_rate']:.0%} vs {fp['baseline_approval_rate']:.0%}) at a higher default rate "
        f"({fp['model_bad_rate']:.1%} vs {fp['baseline_bad_rate']:.1%}). "
        "So the app flags first-time borrowers on the case card and tells the officer to treat them as a standard review."
    )

    st.subheader("Who changes decision, and how they actually did")
    ch = metrics["decision_changes_no_bureau"]
    order = [("both_approve", "Both approve"), ("second_look_only", "Only Second Look approves"),
             ("baseline_only", "Only the external score approves"), ("both_decline", "Both decline")]
    dc = pd.DataFrame([{"Group": lbl, "Applicants": ch[k]["n"], "Defaulted": ch[k]["default_rate"]} for k, lbl in order])
    fig = px.bar(dc, x="Defaulted", y="Group", orientation="h", text=dc["Defaulted"].map("{:.1%}".format),
                 color="Group", color_discrete_sequence=[ui.INK, ui.APPROVE, ui.BASELINE, ui.DECLINE],
                 hover_data={"Applicants": True})
    fig.update_layout(height=260, showlegend=False, xaxis_tickformat=".0%", xaxis_title="Share who defaulted", yaxis_title="")
    st.plotly_chart(fig, key="changes")
    st.caption("The people Second Look adds default less often than the people it drops.")

    st.subheader("What drives the model")
    imp = pd.DataFrame(metrics["importance"]).sort_values("mean_abs_contribution")
    fig = px.bar(imp, x="mean_abs_contribution", y="label", orientation="h", color_discrete_sequence=[ui.INK],
                 labels={"mean_abs_contribution": "Average effect on risk", "label": ""})
    fig.update_layout(height=440)
    st.plotly_chart(fig, key="importance")

# ---------------------------------------------------------------- fairness
with fair_tab:
    st.subheader("Approval rates by group")
    st.markdown(
        "Gender, age and family status are **not** model inputs, but proxies can leak in, so every group is audited at each model's cut-off "
        "on real no-bureau test applicants."
    )
    measure = st.segmented_control(
        "Measure", ["Approval rate", "Approval rate among people who repaid"], default="Approval rate", label_visibility="collapsed")
    col = "repayer_approval_rate" if measure and measure.startswith("Approval rate among") else "approval_rate"
    f = fairness.copy()
    f["model"] = f["model"].map(MODEL_NAMES)
    f["attribute"] = f["attribute"].map({"CODE_GENDER": "Gender", "AGE_BAND": "Age"})
    f["group"] = f["group"].replace({"F": "Women", "M": "Men", "<30": "Under 30", "30-44": "30 to 44", "45-59": "45 to 59", "60+": "60 and over"})
    group_order = ["Women", "Men", "Under 30", "30 to 44", "45 to 59", "60 and over"]
    fig = px.bar(f, x="group", y=col, color="model", barmode="group", text=f[col].map("{:.0%}".format),
                 category_orders={"group": group_order},
                 color_discrete_map={MODEL_NAMES["baseline"]: ui.BASELINE, MODEL_NAMES["second_look"]: ui.APPROVE},
                 labels={"group": "", col: measure or "Approval rate", "model": ""})
    fig.update_layout(height=400, yaxis_tickformat=".0%", yaxis_range=[0, 1.08])
    st.plotly_chart(fig, key="fairness")
    gaps = pd.DataFrame(metrics["fairness_gaps"])
    g1, g2 = st.columns(2)
    for colm, attr, label in ((g1, "AGE_BAND", "Age"), (g2, "CODE_GENDER", "Gender")):
        b = gaps[(gaps["model"] == "baseline") & (gaps["attribute"] == attr)]["approval_gap"].iloc[0]
        s = gaps[(gaps["model"] == "second_look") & (gaps["attribute"] == attr)]["approval_gap"].iloc[0]
        colm.metric(f"Largest {label.lower()} gap in approval rate, Second Look", f"{s:.1%}",
                    f"{(s - b) * 100:+.1f} points vs external score", delta_color="inverse")
    st.warning(
        "Every group is approved more often under Second Look, but the **age gap widens**, mostly because almost all over-60s "
        "are approved. The gender gap widens by under one point. Below is what closing the age gap would cost."
    )

    st.subheader("What closing the age gap costs")
    opts = pd.DataFrame(metrics["fairness_options"])
    by_age = pd.DataFrame(list(opts["approval_by_age"]))
    table = pd.DataFrame({
        "Policy": opts["option"],
        "Approved": opts["approval_rate"].map("{:.1%}".format),
        "Defaults among approved": opts["bad_rate"].map("{:.2%}".format),
        "Extra approvals per 10,000": opts["extra_approved_per_10k"].map("{:+,.0f}".format),
        "Age gap": (opts["age_gap"] * 100).map("{:.1f} pts".format),
        "Under 30": by_age["<30"].map("{:.0%}".format),
        "60 and over": by_age["60+"].map("{:.0%}".format),
    })
    st.table(table.set_index("Policy"))  # static HTML: st.dataframe's grid script sometimes fails to load on Streamlit Cloud
    sl_opt, np_opt, par_opt = (opts.set_index("option").loc[k] for k in
                               ("Second Look", "Second Look without pension signals", "Second Look, equal approval rate by age"))
    st.markdown(
        f"- **Removing the age proxies does not help.** Retrained without income type and employment fields, the gap is still "
        f"{np_opt['age_gap'] * 100:.1f} points: older applicants in this data genuinely default less, so any accurate model favours them.\n"
        f"- **Equal approval rates by age close it** to {par_opt['age_gap'] * 100:.1f} points at the same default rate, but Second Look then adds "
        f"{par_opt['extra_approved_per_10k']:,.0f} approvals per 10,000 instead of {sl_opt['extra_approved_per_10k']:,.0f}. "
        "Each age band gets its own cut-off, chosen on validation data.\n"
        "- **The choice belongs to the lender.** Using age at the decision step is regulated in many places and needs legal review. "
        "Second Look shows the trade-off; it does not make it."
    )

# ---------------------------------------------------------------- how it works
with how:
    st.subheader("How a case gets its second look")
    st.markdown(
        """
1. **Data.** Home Credit Default Risk (Kaggle): the application form, earlier applications with the lender, and instalment repayment history.
   Applicants with no rows in the credit bureau table form the *no-bureau* group.
2. **Baseline.** Logistic regression on the three external scores (`EXT_SOURCE_1/2/3`), standing in for the score a lender buys.
3. **Second Look.** LightGBM on application fields, affordability ratios and the lender's own history.
   It never sees the external scores, bureau enquiries, gender, age or family status.
   Monotone constraints keep it sensible: more late payments can never lower risk.
4. **Cut-offs.** Chosen on a validation split, never on test. The external score approves 80% of no-bureau applicants; Second Look's cut-off is the most generous one with no higher default rate.
5. **Explanations.** Exact per-factor contributions from the tree model (TreeSHAP via LightGBM `pred_contrib`), written in plain English.
6. **Counter-offers.** A search over loan amount (50 to 100%) and term (75 to 150%) for the smallest change that reaches the cut-off.
7. **People decide.** The officer records a decision and a reason for every case, in an exportable log.
"""
    )
    st.subheader("Limits, stated plainly")
    st.markdown(
        """
- **Selection bias.** Home Credit only knows outcomes for people it lent to, so claims are about *approved borrowers without a bureau record*, not every unbanked person.
- **Not Indian data.** The lender and market are not disclosed; results need retraining on local data before they transfer.
- **Lender history matters.** 94% of no-bureau applicants had applied to this lender before. For true first-time borrowers there is no measurable gain, and the app says so on the case.
- **Age gap.** Second Look widens it; closing it costs about a third of the extra approvals (see Fairness).
- **Counter-offers are correlational.** They describe what the model responds to, not a guarantee of repayment.
- **Thin files only.** For applicants with a bureau record, Second Look only matches the external score.

Second Look recommends; it does not decide. It is a hackathon prototype, not a validated credit model, and is not for real lending decisions.
"""
    )
