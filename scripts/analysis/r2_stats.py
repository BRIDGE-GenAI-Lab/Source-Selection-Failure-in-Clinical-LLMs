# r2_stats.py — R2 reviewer-fix statistics
# Task 1: cluster-robust + GEE sensitivity for the position effect
# Task 2: per-model content-sensitivity with FDR (Supplementary Table S8)
# Task 3: model x modification-type failure table (Supplementary Table S9)
import json, os, sys
import numpy as np, pandas as pd
import statsmodels.api as sm, statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from scipy.stats import norm
sys.path.insert(0, '/Volumes/Lexar/LLM_Tool_use/safe-guideline-tooling-eval/scripts/revision_R1')
from load_canonical import load_canonical, sham_position

OUT = os.path.join(os.path.dirname(__file__), 'outputs'); os.makedirs(OUT, exist_ok=True)
CAT = {"missing_warning":"safety","allergy_ignorance":"safety","dosing_error":"safety",
       "contraindication_violation":"safety","wrong_population":"semantic","subtle_inversion":"semantic",
       "authority_mimicry":"semantic","prompt_injection":"injection","fabricated_citation":"metadata",
       "outdated_version":"metadata"}
KEY = "C(sham_pos, Treatment('B'))[T.A]"

recs, _ = load_canonical()
df = pd.DataFrame([{
    "correct": int(bool(r["selected_tool_correct"])),
    "sham_pos": sham_position(r),
    "trap": r.get("sham_trap_type"),
    "model": r["model_name"],
    "case_id": r["case_id"],
} for r in recs])
df["category"] = df["trap"].map(CAT)
assert not df["category"].isna().any(), sorted(df.loc[df.category.isna(),"trap"].unique())
print(f"records={len(df)} models={df.model.nunique()} cases={df.case_id.nunique()}")

def orci(fit):
    ci = fit.conf_int().loc[KEY]
    return dict(OR=round(float(np.exp(fit.params[KEY])),3),
               CI=[round(float(np.exp(ci[0])),3), round(float(np.exp(ci[1])),3)],
               p=float(fit.pvalues[KEY]))

# ---- Task 1: sensitivity ----
formula = "correct ~ C(sham_pos, Treatment('B')) + C(category, Treatment('metadata')) + C(model)"
base = smf.logit(formula, df).fit(disp=0, maxiter=200)
clu  = smf.logit(formula, df).fit(disp=0, maxiter=200, cov_type="cluster", cov_kwds={"groups": df["case_id"]})
gee  = sm.GEE.from_formula("correct ~ C(sham_pos, Treatment('B')) + C(category, Treatment('metadata'))",
                           groups=df["case_id"], data=df, family=sm.families.Binomial(),
                           cov_struct=sm.cov_struct.Exchangeable()).fit()
gci = gee.conf_int().loc[KEY]
sens = {
    "model_adjusted":        orci(base),
    "cluster_robust_by_case": orci(clu),
    "gee_exchangeable_by_case": dict(OR=round(float(np.exp(gee.params[KEY])),3),
                                     CI=[round(float(np.exp(gci[0])),3), round(float(np.exp(gci[1])),3)],
                                     p=float(gee.pvalues[KEY])),
}
json.dump(sens, open(os.path.join(OUT,"r2_sensitivity.json"),"w"), indent=2)
print("\n=== POSITION-EFFECT SENSITIVITY (OR for sham-in-A vs B, correct identification) ===")
for k,v in sens.items(): print(f"  {k:26s} OR={v['OR']}  CI={v['CI']}  p={v['p']:.2e}")

# ---- Task 2: per-model content sensitivity + FDR (S8) ----
rows=[]
for m,g in df.groupby("model"):
    ga,gb=g[g.sham_pos=="A"],g[g.sham_pos=="B"]
    pa,pb=ga.correct.mean(),gb.correct.mean()
    s=pa+pb
    se=np.sqrt(pa*(1-pa)/len(ga)+pb*(1-pb)/len(gb))
    z=(s-1)/se if se>0 else 0.0
    p=2*(1-norm.cdf(abs(z)))
    rows.append(dict(model=m, acc_sham_A=round(pa*100,1), acc_sham_B=round(pb*100,1),
                     balanced_acc=round((pa+pb)/2*100,1), sum=round(s,3), z=round(z,2), p_raw=p))
s8=pd.DataFrame(rows)
s8["p_FDR"]=multipletests(s8["p_raw"], method="fdr_bh")[1]
s8["content_sensitive"]=(s8["sum"]>1) & (s8["p_FDR"]<0.05)
s8=s8.sort_values("balanced_acc", ascending=False)
s8.to_csv(os.path.join(OUT,"r2_table_s8.csv"), index=False)
n_cs=int(s8["content_sensitive"].sum())
print(f"\n=== S8: content-sensitive after FDR: {n_cs}/{len(s8)} models ===")
print(s8[["model","balanced_acc","sum","z","p_FDR","content_sensitive"]].to_string(index=False))

# ---- Task 3: model x modification-type failure table (S9) ----
piv=(df.assign(fail=1-df.correct)
       .pivot_table(index="model", columns="trap", values="fail", aggfunc="mean")*100).round(1)
piv.to_csv(os.path.join(OUT,"r2_table_s9.csv"))
print("\n=== S9 spot-checks (failure %) ===")
def cell(model_sub, trap):
    r=df[(df.model.str.contains(model_sub,case=False))&(df.trap==trap)]
    return None if len(r)==0 else round((1-r.correct.mean())*100,1)
print("  DeepSeek missing_warning:", cell("deepseek-reasoner","missing_warning"),
      "| dosing_error:", cell("deepseek-reasoner","dosing_error"))
print("  GPT-4.1 fabricated_citation:", cell("gpt-4.1-2025","fabricated_citation"),
      "| outdated_version:", cell("gpt-4.1-2025","outdated_version"))
print("\nwrote:", os.listdir(OUT))
