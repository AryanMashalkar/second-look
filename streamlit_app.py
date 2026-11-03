"""Second Look: a review queue for loan officers handling applicants with no credit bureau record."""

import json
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from secondlook import config
from secondlook.features import add_ratios
from secondlook.model import Baseline, SecondLook

st.set_page_config(page_title="Second Look", page_icon="🔎", layout="wide")


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
policy = nb["policy"]["second_look"]


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


def verdict(risk, threshold):
    return "Approve" if risk <= threshold else "Decline"


# ---------------------------------------------------------------- header
st.title("🔎 Second Look")
st.markdown(
    "**A second opinion for applicants the credit bureau has never heard of.** "
    "Lenders lean on opaque external scores, which are weakest exactly where there is no credit history. "
    "Second Look re-scores those applicants from the application and the lender's own repayment records, "
    "explains every decision, and leaves the final call to a loan officer."
)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Ranking quality (AUC), no-bureau applicants", f"{nb['auc']['second_look']['auc']:.3f}",
          f"{nb['auc']['second_look']['delta_vs_baseline']['mean']:+.3f} vs external score")
c2.metric("Extra applicants approved per 10,000", f"+{policy['extra_approved_per_10k']:,.0f}",
          f"at the same {policy['baseline_bad_rate']:.1%} bad rate", delta_color="off")
c3.metric("Of whom actually repaid", f"+{policy['extra_repayers_per_10k']:,.0f}", "per 10,000 applicants", delta_color="off")
c4.metric("Real held-out applicants with no bureau record", f"{metrics['split']['test_no_bureau']:,}")

queue_tab, evidence_tab, how_tab = st.tabs(["Review queue", "Evidence", "How it works and limits"])

# ---------------------------------------------------------------- queue
with queue_tab:
    st.info(
        "Demo profiles are **synthetic**: each is stitched together from parts of different training applicants and "
        "jittered, because the Home Credit data may not be redistributed. No profile is a real person. "
        "Every number on the Evidence tab comes from real held-out applicants.",
        icon="ℹ️",
    )
    if "decisions" not in st.session_state:
        st.session_state.decisions = []

    view = profiles.copy()
    view["External"] = [verdict(r, baseline.threshold) for r in view["risk_baseline"]]
    view["Second Look"] = [verdict(r, model.threshold) for r in view["risk_second_look"]]
    reasons_order = ["Second look: approve", "Second look: caution", "Both approve", "Both decline"]
    pick_filter = st.multiselect("Show", reasons_order, default=reasons_order[:2])
    view = view[view["queue_reason"].isin(pick_filter)]
    view = view.assign(_order=view["queue_reason"].map(reasons_order.index)).sort_values(["_order", "profile_id"])

    left, right = st.columns([5, 6], gap="large")
    with left:
        table = view[["profile_id", "queue_reason", "AMT_CREDIT", "AMT_INCOME_TOTAL", "External", "Second Look"]]
        table = table.rename(columns={"profile_id": "Profile", "queue_reason": "Why in queue", "AMT_CREDIT": "Loan", "AMT_INCOME_TOTAL": "Income"})
        st.dataframe(table, hide_index=True, width="stretch", height=520,
                     column_config={"Loan": st.column_config.NumberColumn(format="%d"), "Income": st.column_config.NumberColumn(format="%d")})
    with right:
        if view.empty:
            st.write("Nothing matches this filter.")
            st.stop()
        pid = st.selectbox("Open profile", view["profile_id"])
        row = profiles[profiles["profile_id"] == pid].reset_index(drop=True)
        r_base, r_sl = float(row["risk_baseline"][0]), float(row["risk_second_look"][0])

        a, b = st.columns(2)
        a.metric("External score: default risk", f"{r_base:.1%}", verdict(r_base, baseline.threshold), delta_color="off")
        b.metric("Second Look: default risk", f"{r_sl:.1%}", verdict(r_sl, model.threshold), delta_color="off")
        st.caption(
            f"Approval cut-offs: external score ≤ {baseline.threshold:.1%}, Second Look ≤ {model.threshold:.1%}. "
            "Both are set so that approved applicants default at the same rate on real test data."
        )

        reasons = model.reasons(row)
        reasons["shown"] = [f"{l}: {fmt(f, v)}" for l, f, v in zip(reasons["label"], reasons["feature"], reasons["value"])]
        reasons = reasons.sort_values("contribution")
        fig = go.Figure(go.Bar(
            x=reasons["contribution"], y=reasons["shown"], orientation="h",
            marker_color=["#2e7d32" if c < 0 else "#c62828" for c in reasons["contribution"]],
        ))
        fig.update_layout(title="Why Second Look scored this way (red raises risk, green lowers it)",
                          height=360, margin=dict(l=10, r=10, t=40, b=10), xaxis_title="Effect on risk (log-odds)")
        st.plotly_chart(fig)

        rc = model.recourse(row)
        if rc is None:
            st.success("Within Second Look's approval cut-off as applied for.")
        elif rc["found"]:
            changes = []
            if rc["amount_factor"] < 1:
                changes.append(f"a loan of **{rc['new_credit']:,.0f}** instead of {float(row['AMT_CREDIT'][0]):,.0f}")
            if rc["term_factor"] != 1:
                changes.append(f"a term of **{rc['new_term_months']:.0f} months** (monthly repayment {rc['new_annuity']:,.0f})")
            st.warning(
                "**Counter-offer:** " + " and ".join(changes) + f" brings risk to {rc['new_risk']:.1%}, inside the cut-off. "
                "This describes what the model responds to; it changes only loan terms, never who the applicant is."
            )
        else:
            st.error("No counter-offer within ±50% of the loan amount and term brings this applicant inside the cut-off.")

        with st.expander("What if the loan terms change?"):
            amount = st.slider("Loan amount, % of requested", 40, 120, 100, 5, key=f"amt_{pid}")
            term = st.slider("Term, % of requested", 50, 200, 100, 5, key=f"term_{pid}")
            what_if = row.copy()
            what_if["AMT_CREDIT"] = row["AMT_CREDIT"] * amount / 100
            what_if["AMT_ANNUITY"] = row["AMT_ANNUITY"] * (amount / 100) / (term / 100)
            add_ratios(what_if)
            r_new = float(model.risk(what_if)[0])
            st.write(
                f"Loan {float(what_if['AMT_CREDIT'][0]):,.0f}, {float(what_if['TERM_MONTHS'][0]):.0f} months, "
                f"monthly repayment {float(what_if['AMT_ANNUITY'][0]):,.0f} → risk **{r_new:.1%}** "
                f"({verdict(r_new, model.threshold)})"
            )

        with st.expander("All fields for this profile"):
            fields = pd.DataFrame(
                [(config.LABELS.get(f, f), fmt(f, row[f][0])) for f in config.SECOND_LOOK_FEATURES],
                columns=["Field", "Value"],
            )
            st.dataframe(fields, hide_index=True, width="stretch")

        st.subheader("Officer decision")
        with st.form(f"decide_{pid}", clear_on_submit=True):
            choice = st.radio("Decision", ["Approve", "Approve with counter-offer", "Request documents", "Decline"], horizontal=True)
            note = st.text_input("Reason (required, shared with the applicant if declined)")
            if st.form_submit_button("Record decision"):
                if not note.strip():
                    st.error("Write a reason before recording a decision.")
                else:
                    st.session_state.decisions.append({
                        "time": datetime.now().strftime("%H:%M:%S"), "profile": pid, "decision": choice, "reason": note.strip(),
                        "external_risk": round(r_base, 4), "second_look_risk": round(r_sl, 4),
                    })
                    st.success(f"Recorded: {choice} for {pid}.")
        if st.session_state.decisions:
            log = pd.DataFrame(st.session_state.decisions)
            st.dataframe(log, hide_index=True, width="stretch")
            st.download_button("Download decision log (CSV)", log.to_csv(index=False), "decision_log.csv", "text/csv")

# ---------------------------------------------------------------- evidence
with evidence_tab:
    a = metrics["audit"]
    st.markdown(
        f"Trained and tested on the **Home Credit Default Risk** data ({a['applicants']:,} real loans). "
        f"**{a['no_bureau_applicants']:,} applicants ({a['no_bureau_share']:.1%}) have no credit bureau record**, and they default more often "
        f"({a['no_bureau_default_rate']:.1%} vs {a['bureau_default_rate']:.1%}). Everything below is measured on a held-out 20% test split "
        f"({metrics['split']['test']:,} applicants, {metrics['split']['test_no_bureau']:,} without a bureau record) that no model saw during training."
    )

    st.subheader("1. Ranking quality on applicants with no bureau record")
    names = {"baseline": "External score (baseline)", "application_only": "Second Look without lender history (ablation)",
             "second_look": "Second Look", "combined": "Second Look + external score"}
    rows = []
    for key, label in names.items():
        for pop in ("no_bureau", "all"):
            v = metrics[pop]["auc"][key]
            d = v.get("delta_vs_baseline")
            rows.append({"Model": label, "Population": "No bureau record" if pop == "no_bureau" else "Everyone",
                         "AUC": f"{v['auc']:.3f} ({v['ci_low']:.3f}–{v['ci_high']:.3f})",
                         "Gain vs baseline (95% CI)": f"{d['mean']:+.3f} ({d['ci_low']:+.3f} to {d['ci_high']:+.3f})" if d else "—"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("95% intervals from 1,000 paired bootstrap resamples of the test set.")

    st.subheader("2. What it means for lending decisions")
    c = curves[(curves["population"] == "no_bureau") & curves["model"].isin(["baseline", "second_look"])].copy()
    c["model"] = c["model"].map(names)
    fig = px.line(c, x="approval_rate", y="bad_rate", color="model",
                  labels={"approval_rate": "Share of applicants approved", "bad_rate": "Default rate among approved", "model": ""})
    fig.add_hline(y=policy["baseline_bad_rate"], line_dash="dot", annotation_text="Bad rate when the baseline approves 80%")
    fig.update_layout(height=380, yaxis_tickformat=".0%", xaxis_tickformat=".0%", legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig)
    st.markdown(
        f"If the baseline approves 80% of no-bureau applicants, approved loans default at {policy['baseline_bad_rate']:.1%}. "
        f"At that same default rate Second Look approves **{policy['model_approval_rate']:.1%}**: "
        f"**{policy['extra_approved_per_10k']:,.0f} more people per 10,000**, of whom **{policy['extra_repayers_per_10k']:,.0f} repaid**."
    )
    ch = metrics["decision_changes_no_bureau"]
    st.dataframe(pd.DataFrame([
        {"Group": "Both approve", **ch["both_approve"]},
        {"Group": "Only Second Look approves", **ch["second_look_only"]},
        {"Group": "Only the external score approves", **ch["baseline_only"]},
        {"Group": "Both decline", **ch["both_decline"]},
    ]).rename(columns={"n": "Test applicants", "default_rate": "Actually defaulted"}).style.format({"Actually defaulted": "{:.1%}"}),
        hide_index=True, width="stretch")
    st.caption("The people Second Look adds default less often than the people it removes.")

    st.subheader("3. Fairness audit")
    st.markdown(
        "Gender, age and family status are **not** model inputs, but proxies can leak in, so approval rates are audited per group "
        "at each model's cut-off. *Repayer approval rate* is the share of people who actually repaid that got approved (equal opportunity)."
    )
    f = fairness.copy()
    f["model"] = f["model"].map(names)
    piv = f.pivot_table(index=["attribute", "group"], columns="model", values=["approval_rate", "repayer_approval_rate"])
    piv.columns = [f"{m}: {'approval' if v == 'approval_rate' else 'repayer approval'}" for v, m in piv.columns]
    st.dataframe(piv.style.format("{:.1%}"), width="stretch")
    gaps = pd.DataFrame(metrics["fairness_gaps"])
    gaps["model"] = gaps["model"].map(names)
    gaps = gaps.rename(columns={"model": "Model", "attribute": "Attribute", "approval_gap": "Largest approval gap",
                                "repayer_approval_gap": "Largest repayer approval gap"})
    st.dataframe(gaps.style.format({"Largest approval gap": "{:.1%}", "Largest repayer approval gap": "{:.1%}"}),
                 hide_index=True, width="stretch")
    st.warning(
        "Every gender and age group is approved more often under Second Look, but the **age gap widens** "
        "because almost all over-60s are approved (the model rewards pensioner income stability). "
        "The gender gap is about one point wider. A lender would need to review this before deployment."
    )

    st.subheader("4. What drives the model")
    imp = pd.DataFrame(metrics["importance"]).sort_values("mean_abs_contribution")
    fig = px.bar(imp, x="mean_abs_contribution", y="label", orientation="h",
                 labels={"mean_abs_contribution": "Mean absolute effect on risk (log-odds)", "label": ""})
    fig.update_layout(height=460)
    st.plotly_chart(fig)
    st.markdown(
        f"The ablation shows where the gain comes from: without the lender's own history, the AUC gain over the baseline drops from "
        f"{nb['auc']['second_look']['delta_vs_baseline']['mean']:+.3f} to {nb['auc']['application_only']['delta_vs_baseline']['mean']:+.3f}. "
        f"{a['no_bureau_with_lender_history']:.0%} of no-bureau applicants have earlier applications with this lender."
    )

# ---------------------------------------------------------------- how
with how_tab:
    st.markdown(
        """
### Pipeline
1. **Data**: Home Credit Default Risk (Kaggle): application form, earlier applications with the lender, and instalment repayment history.
   Applicants with no rows in the credit bureau table form the *no-bureau* group.
2. **Baseline**: logistic regression on the three external scores (`EXT_SOURCE_1/2/3`), standing in for the opaque score a lender buys.
3. **Second Look**: LightGBM on application fields, affordability ratios and the lender's own history.
   It never sees the external scores, bureau enquiries, gender, age or family status.
   Monotone constraints force common-sense directions (more late payments can never lower risk).
4. **Cut-offs**: the baseline approves 80% of no-bureau applicants; Second Look's cut-off is the most generous one with the same default rate.
5. **Explanations**: exact per-feature contributions from the tree model (TreeSHAP via LightGBM `pred_contrib`), turned into plain-English reasons.
6. **Counter-offers**: a search over loan amount (50–100%) and term (75–150%) for the smallest change that reaches the cut-off.
7. **Human in the loop**: the officer records a decision and a reason for every case; the log is exportable.

### Limits, stated plainly
- **Selection bias.** Home Credit only shows outcomes for people it lent to. Rejected applicants never got a loan, so their repayment is unknown.
  Claims here are about *approved borrowers without a bureau record*, not about the unbanked population as a whole.
- **Not Indian data.** The lender and market are not disclosed, so results do not transfer to India without retraining on local data.
- **Lender history matters.** 94% of no-bureau applicants had applied to this lender before; for true first-time borrowers the gain is smaller (see the ablation).
- **Age gap.** Second Look widens the approval gap between age bands (see the fairness audit).
- **Counter-offers are correlational.** They describe what the model responds to, not a guarantee of repayment.
- **Overall population.** For applicants who do have a bureau record, Second Look is only on par with the external score; it is a second opinion for thin files, not a replacement.

### Responsible use
Second Look recommends; it does not decide. It is a hackathon prototype, not a validated credit model, and must not be used for real lending decisions.
"""
    )
