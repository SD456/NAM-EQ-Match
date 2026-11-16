"""Post-result tables and timing diagnostics; never change the frozen decision."""
import csv
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.stage2_benchmark import read_json,write_json,sha


def main():
    folder = ROOT/"artifacts/stage6/calibration"
    result = read_json(folder/"results.json")
    assert read_json(folder/"locked-plan.json") == result["plan"]
    assert sha(folder/"locked-plan.json") == result["lock_sha256"]
    assert sha(folder/"selected-setting.json") == result["selection_sha256"]
    for key in ("assets","protected_baselines","source_hashes"):
        assert all(sha(ROOT/p)==v for p,v in result["plan"][key].items())
    rows,diagnostics = [],{}
    for split in ("calibration","heldout"):
        run = result[split]
        for item in run["grid"]:
            window,threshold = item["pitch_window"],item["threshold"]
            counts = [n for row in run["rows"] for x in row["pitch"]
                      if x["pitch_window"]==window and x["threshold"]==threshold
                      for n in x["metrics"]["per_note"] if n["reference_frames"]]
            a,c,o = [sum(n[k]/n["reference_frames"] for n in counts)
                     for k in ("accepted_frames","correct_frames","octave_frames")]
            metrics = item["metrics"]
            np.testing.assert_allclose([a,c,o],[metrics[k] for k in
                ("accepted_weight","correct_weight","octave_weight")],rtol=0,atol=1e-12)
            assert len(counts)==metrics["n_notes"]==78
            rows.append({"split":split,"pitch_window":window,"threshold":threshold,
                **{k:metrics[k] for k in ("n_notes","total_reference_frames","coverage","conditional_exact",
                                        "full_correct","conditional_octave","n_notes_with_any_accept")}})
        diagnostics[split] = {"zero_answer_notes":{},"octave_error_notes":{},"onset_extra_timing":{}}
        for window in (1024,2048):
            items = [(row,next(x["metrics"] for x in row["pitch"]
                      if x["pitch_window"]==window and x["threshold"]==.8)) for row in run["rows"]]
            diagnostics[split]["zero_answer_notes"][window] = [
                {"id":row["id"],"pitch":row["event"]["pitch"]} for row,m in items if not m["n_notes_with_any_accept"]]
            diagnostics[split]["octave_error_notes"][window] = [
                {"id":row["id"],"pitch":row["event"]["pitch"],"counts":m["per_note"]}
                for row,m in items if m["octave_weight"]]
            extra = {"before_label_minus_50ms":0,"duplicate_within_label_50ms":0,
                     "after_label_50_to_120ms":0,"after_label_120_to_400ms":0,"after_label_at_least_400ms":0}
            for row in run["rows"]:
                truth = row["event"]["onset"]
                pred = row["predicted_onsets"][str(window)]
                matched = next((i for i,p in enumerate(pred) if truth-.05<=p<=truth+.05),None)
                for i,p in enumerate(pred):
                    if i==matched:
                        continue
                    age = p-truth
                    key = ("before_label_minus_50ms" if age<-.05 else
                           "duplicate_within_label_50ms" if age<=.05 else
                           "after_label_50_to_120ms" if age<.12 else
                           "after_label_120_to_400ms" if age<.4 else "after_label_at_least_400ms")
                    extra[key] += 1
            reference = next(x["metrics"] for x in run["onsets"] if x["window"]==window and x["tolerance"]==.05)
            assert sum(extra.values()) == reference["fp"]
            diagnostics[split]["onset_extra_timing"][window] = extra
    with (folder/"pitch-grid.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(ROOT/"artifacts/stage6/analysis.json",{
        "status":"post_result_diagnostics_no_reselection","result_sha256":sha(folder/"results.json"),
        "script_sha256":sha(__file__),"selection":result["selection"],"gate":result["gate"],
        "grid":rows,"diagnostics":diagnostics,
        "confidence_bins":{s:{str(w):next(x["metrics"]["confidence_bins"] for x in result[s]["grid"]
            if x["pitch_window"]==w and x["threshold"]==0) for w in (1024,2048)} for s in ("calibration","heldout")}})
    fig,axes = plt.subplots(1,2,figsize=(11.5,4.3),layout="constrained")
    for split,style in (("calibration","-"),("heldout","--")):
        for window,color in ((1024,"#cf4b28"),(2048,"#176b99")):
            group = [x for x in rows if x["split"]==split and x["pitch_window"]==window]
            axes[0].plot([x["threshold"] for x in group],[100*x["coverage"] for x in group],
                         style,marker="o",color=color,label=f'{window//16} ms, {split}')
    axes[0].set(xlabel="pYIN probability cutoff",ylabel="Coverage of labeled core (%)",ylim=(65,102),
                title="Longer pitch window preserves coverage")
    axes[0].grid(alpha=.2)
    axes[0].legend(fontsize=8,loc="lower left")
    for i,split in enumerate(("calibration","heldout")):
        group = [next(x["metrics"] for x in result[split]["onsets"]
                      if x["window"]==w and x["tolerance"]==.05) for w in (1024,2048)]
        x = np.arange(2)+(i-.5)*.32
        axes[1].bar(x,[g["f1"] for g in group],width=.30,label=split,color=("#729fae","#304d63")[i])
        for position,g in zip(x,group):
            axes[1].text(position,g["f1"]+.015,f'{g["f1"]:.3f}\nFP {g["fp"]}',ha="center",fontsize=8)
    axes[1].axhline(.8,color="#a83d35",linestyle=":",label="Frozen primary gate 0.80")
    axes[1].set(xticks=[0,1],xticklabels=["64 ms (primary)","128 ms (diagnostic)"],
                ylabel="Onset F1 (50 ms tolerance)",ylim=(0,1.06),
                title="Shorter onset window produces more extra events")
    axes[1].legend(fontsize=8,loc="lower right")
    fig.suptitle("Clean single-note DI: 78 Ibanez calibration / 78 Fender held-out notes")
    fig.savefig(ROOT/"artifacts/stage6/frontend-calibration.png",dpi=160)
    plt.close(fig)
    print("Verified frozen hashes and 24 equal-note summaries; exported tables, plot and post-result diagnostics")


if __name__=="__main__":
    main()
