import json, pandas as pd, os
out={"l1": pd.read_csv("results/l1_results.csv").to_dict("records")}
if os.path.exists("results/l2_results.csv"):
    d=pd.read_csv("results/l2_results.csv"); out["l2"]=d.to_dict("records")
    lo=d[d.protocol=="lopo"].set_index("model"); ra=d[d.protocol=="random"].set_index("model")
    best=lo.macroF1.idxmax()
    out["l2note"]=("On the 358 PROMISE quality requirements with an attribute, every model loses accuracy on unseen projects: "
      f"B1 macro-F1 {ra.macroF1['B1']:.2f} → {lo.macroF1['B1']:.2f}, B3 {ra.macroF1['B3']:.2f} → {lo.macroF1['B3']:.2f}. "
      f"Security and Usability are recognised best (B3 {lo.F1_Security['B3']:.2f}, {lo.F1_Usability['B3']:.2f}); Maintainability ({lo.F1_Maintainability['B3']:.2f}) and Reliability ({lo.F1_Reliability['B3']:.2f}) are weakest, "
      f"and TF-IDF nearly fails on Legal ({lo.F1_Legal['B1']:.2f}). External attribute labels must still be annotated before Level 2 can be tested outside PROMISE.")
json.dump(out,open("report/results.json","w"),default=float)
print(list(out))
