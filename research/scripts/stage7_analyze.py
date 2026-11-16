"""Read-only result analysis and plots; no audio, peak changes or reselection."""
import csv
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.stage2_benchmark import read_json,write_json,sha
from scripts.onset_merging import pair_classes


def timing_support(rows):
    result = {}
    for name in ('fast_change','repeat','fast_repeat'):
        pairs = [(row,a,b) for row in rows for a,b in pair_classes(row['events'])[name]]
        gaps = [round((row['events'][b]['onset']-row['events'][a]['onset'])*1e6) for row,a,b in pairs]
        result[name] = dict(pairs=len(pairs),clips=len({row['id'] for row,_,_ in pairs}),
            distinct_licks=len({row['lick'] for row,_,_ in pairs}),
            min_ioi_ms=min(gaps)/1000 if gaps else None,
            ioi_bins=[sum(g<=80000 for g in gaps),sum(80000<g<=180000 for g in gaps),
                      sum(180000<g<=200000 for g in gaps),sum(g>200000 for g in gaps)])
    return result


def main():
    folder = ROOT/'artifacts/stage7/onsets'
    result = read_json(folder/'results.json')
    assert result['plan']==read_json(folder/'locked-plan.json')
    assert sha(folder/'locked-plan.json')==result['lock_sha256']
    assert read_json(folder/'selected-setting.json')==result['selected']
    assert sha(folder/'calibration.json')==result['selected']['calibration_sha256']
    for key in ('source_hashes','protected_baselines','assets','historical_assets'):
        assert all(sha(ROOT/p)==v for p,v in result['plan'][key].items())
    groups = dict(calibration=result['calibration'],heldout=result['heldout'],
                  historical_calibration=result['historical_replay']['calibration'],
                  historical_heldout=result['historical_replay']['heldout'])
    table,fp_locations,losses = [],{},[]
    for group,run in groups.items():
        fp_locations[group] = {}
        for method,by_tolerance in run['summary'].items():
            for tolerance,score in by_tolerance.items():
                row = dict(group=group,method=method,tolerance=float(tolerance))
                row.update({k:score[k] for k in ('tp','fp','fn','near_fp','other_fp','macro_f1','micro_f1',
                                               'precision','recall','median_signed_ms','median_absolute_ms')})
                for name,p in score['protected'].items():
                    row.update({name+'_'+k:p[k] for k in ('support','merged_later_hits','merged_both_pair_hits',
                        'baseline_recovered_later_lost_count','baseline_both_pair_lost_count')})
                table.append(row)
            locations = dict(before_first_label=0,within_label_envelope=0,after_last_label_offset=0)
            for clip in run['rows']:
                item = clip['methods'][method]
                primary = item['scores']['0.05']
                first,last = clip['events'][0]['onset'],max(e['offset'] for e in clip['events'])
                for index in primary['matches']['fp_indices']:
                    time = item['predicted_onsets'][index]
                    time_us,first_us,last_us = (round(t*1e6) for t in (time,first,last))
                    key = ('before_first_label' if time_us<first_us-50000 else
                           'after_last_label_offset' if time_us>last_us+50000 else 'within_label_envelope')
                    locations[key] += 1
                for loss in primary.get('loss_attribution',[]):
                    event = clip['events'][loss['truth_index']]
                    losses.append(dict(group=group,id=clip['id'],method=method,event=event,**loss))
            assert sum(locations.values())==by_tolerance['0.05']['fp']
            fp_locations[group][method] = locations
    with (folder/'method-grid.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    support = {s:timing_support(result[s]['rows']) for s in ('calibration','heldout')}
    historical_deltas = [v for run in result['historical_replay'].values() for row in run['rows']
                         for method in ('raw64','raw128') for score in row['methods'][method]['scores'].values()
                         for v in score['score_delta'].values()]
    analysis = dict(status='post_result_diagnostics_no_reselection',result_sha256=sha(folder/'results.json'),
        script_sha256=sha(__file__),selected=result['selected'],gate=result['gate'],grid=table,
        label_support=support,unmatched_peak_time_locations=fp_locations,loss_attribution=losses,
        historical_raw_peak_reproduction_verified=True,
        historical_score_delta_max=max(map(abs,historical_deltas)))
    write_json(ROOT/'artifacts/stage7/analysis.json',analysis)

    fig,axes = plt.subplots(2,2,figsize=(12.4,8.5),layout='constrained')
    for axis,split,title in zip(axes[0],('calibration','heldout'),('New calibration (9 clips)','New held-out (9 clips)')):
        scores = result[split]['summary']
        for method,color in (('first','#21759b'),('strongest','#d66730')):
            axis.plot([32,48,64,80],[scores[f'{method}_r{r}']['0.05']['macro_f1'] for r in (2,3,4,5)],
                      marker='o',color=color,label=method)
        axis.axhline(scores['raw64']['0.05']['macro_f1'],color='#777777',linestyle='--',label='raw 64 ms')
        axis.axhline(.8,color='#a33636',linestyle=':',label='held-out F1 target')
        axis.set(title=title,xlabel='Inclusive merging radius (ms)',ylabel='Mean clip F1 (50 ms tolerance)',ylim=(.45,.85))
        axis.legend(fontsize=8)
        axis.grid(alpha=.2)
    names = ['raw64','raw128','first_r5','strongest_r5']
    scores = [result['heldout']['summary'][m]['0.05'] for m in names]
    x = np.arange(len(names))
    axes[1,0].bar(x-.18,[s['fp'] for s in scores],width=.35,color='#cc9c49',label='Extra peaks (FP)')
    axes[1,0].bar(x+.18,[s['fn'] for s in scores],width=.35,color='#ba4d45',label='Unmatched labels (FN)')
    for i,s in enumerate(scores):
        axes[1,0].text(i-.18,s['fp']+2,str(s['fp']),ha='center',fontsize=9)
        axes[1,0].text(i+.18,s['fn']+2,str(s['fn']),ha='center',fontsize=9)
    axes[1,0].set(title='New held-out: fewer peaks can lose events',xticks=x,
                  xticklabels=['Raw 64','Raw 128','First 80','Strongest 80'],ylabel='Count')
    axes[1,0].legend(fontsize=8)
    for j,name in enumerate(('fast_change','repeat')):
        values = [s['protected'][name]['merged_both_pair_rate'] for s in scores]
        axes[1,1].bar(x+(j-.5)*.35,values,width=.34,label=f'{name} (N={scores[0]["protected"][name]["support"]})',
                      color=('#3d8198','#95b4aa')[j])
    axes[1,1].set(title='Both labeled events recovered (small subsets)',xticks=x,
                  xticklabels=['Raw 64','Raw 128','First 80','Strongest 80'],ylabel='Pair recovery fraction',ylim=(0,1.18))
    axes[1,1].legend(fontsize=8,loc='lower left')
    axes[1,1].text(.02,.96,'Fast repeated pitch <=200 ms: N=0',transform=axes[1,1].transAxes,va='top',fontsize=9)
    fig.suptitle('Onset merging: 3 new musical patterns per role, 3 guitar performances per pattern')
    fig.savefig(ROOT/'artifacts/stage7/onset-merging.png',dpi=160)
    plt.close(fig)
    print('Verified frozen hashes; wrote 80 aggregate rows, label support, FP locations, loss provenance and plot')


if __name__=='__main__':
    main()
