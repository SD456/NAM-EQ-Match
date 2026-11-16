"""Synthetic policy checks: no real waveform features are computed here."""
from copy import deepcopy
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from scripts.stage7_benchmark import (CLASSES, FIXED, METHODS, choose_setting, evaluate,
                                      gate, label_seconds, loss_attribution, summarize, validate_config)
from scripts.onset_merging import match_onset_indices, safety_metrics


def summaries():
    protection = {name:dict(support=20,clips=3,baseline_recovered_later_lost_count=0,
                           baseline_both_pair_lost_count=0) for name in CLASSES}
    row = dict(macro_f1=.8,micro_f1=.95,near_fp=4,fp=5,protected=protection)
    return {m:{'0.05':deepcopy(row)} for m in METHODS}


class Stage7PolicyTests(unittest.TestCase):
    def test_selection_uses_macro_and_requires_near_fp_reduction(self):
        s = summaries()
        s['first_r2']['0.05'].update(macro_f1=.79,micro_f1=.99,near_fp=1)
        s['strongest_r3']['0.05'].update(macro_f1=.81,micro_f1=.8,near_fp=2)
        s['first_r4']['0.05'].update(macro_f1=.9,near_fp=4)
        self.assertEqual(choose_setting(s),'strongest_r3')

    def test_net_recall_improvement_does_not_erase_lost_protected_id(self):
        events = [dict(onset=i*.15,pitch=p) for i,p in enumerate([60,60,62,64])]
        raw = match_onset_indices([0,.15],[0,.15,.3,.45],.05)
        merged = match_onset_indices([0,.3,.45],[0,.15,.3,.45],.05)
        safety = safety_metrics(raw,merged,events)
        self.assertGreater(merged['metrics']['recall'],raw['metrics']['recall'])
        self.assertEqual(safety['repeat']['baseline_recovered_later_lost'],[1])
        s = summaries()
        candidate = s['first_r2']['0.05']
        candidate.update(macro_f1=.9,near_fp=1)
        candidate['protected']['repeat']['baseline_recovered_later_lost_count'] = 1
        self.assertIsNone(choose_setting(s))

    def test_tie_order_fp_radius_then_first(self):
        s = summaries()
        for m in METHODS[2:]:
            s[m]['0.05'].update(macro_f1=.9,near_fp=1,fp=2)
        s['first_r2']['0.05']['fp'] = 3
        self.assertEqual(choose_setting(s),'strongest_r2')
        s['first_r2']['0.05']['fp'] = 2
        self.assertEqual(choose_setting(s),'first_r2')

    def test_no_feasible_candidate_retains_baseline(self):
        s = summaries()
        self.assertIsNone(choose_setting(s))
        decision = gate(s,s,None,FIXED)
        self.assertEqual(decision['status'],'retain_baseline')
        self.assertFalse(decision['promote'])

    def test_insufficient_empty_class_cannot_promote(self):
        s = summaries()
        s['first_r2']['0.05'].update(macro_f1=.9,near_fp=1)
        s['raw64']['0.05']['protected']['fast_repeat'].update(support=0,clips=0)
        self.assertEqual(choose_setting(s),'first_r2')
        decision = gate(s,s,'first_r2',FIXED)
        self.assertTrue(decision['performance_passed'])
        self.assertEqual(decision['status'],'insufficient_evidence')
        self.assertFalse(decision['promote'])

    def test_empty_metrics_keep_zero_denominators_explicit(self):
        empty = match_onset_indices([],[],.05)
        safety = safety_metrics(empty,empty,[])
        row = dict(id='silent',methods={m:dict(scores={t:dict(matches=empty,near_fp=0,other_fp=0,safety=safety)
                                                    for t in ('0.05','0.1')}) for m in METHODS})
        summary = summarize([row])['raw64']['0.05']
        self.assertEqual((summary['macro_f1'],summary['micro_f1']), (0,0))
        self.assertIsNone(summary['median_absolute_ms'])
        self.assertIsNone(summary['protected']['fast_repeat']['merged_later_recall'])

    def test_exact_silence_has_no_raw_or_merged_peaks(self):
        with tempfile.TemporaryDirectory() as directory:
            sf.write(Path(directory)/'silence.wav',np.zeros(4410),44100,subtype='PCM_24')
            record = dict(id='silent',split='synthetic',wav='silence.wav',sample_rate=44100,frames=4410,events=[])
            with patch('scripts.stage7_benchmark.ROOT',Path(directory)):
                result = evaluate([record],FIXED)
        row = result['rows'][0]
        self.assertTrue(row['silent'])
        self.assertTrue(all(not method['predicted_onsets'] for method in row['methods'].values()))

    def test_config_and_label_validation(self):
        validate_config(FIXED)
        for key in FIXED:
            bad = dict(FIXED,**{key:None})
            with self.assertRaisesRegex(ValueError,key):
                validate_config(bad)
        self.assertEqual(label_seconds('14.4368'),14.4368)
        for value in ('nan','-1','0.0000001'):
            with self.assertRaises(ValueError):
                label_seconds(value)

    def test_lost_truth_attribution_distinguishes_suppression_and_reassignment(self):
        raw = dict(pairs=[[0,0],[1,1],[2,2]])
        matches = dict(pairs=[[0,2]])
        result = loss_attribution(raw,matches,[10,12,18],dict(kept_indices=[0,2],suppressed_by={1:0}))
        self.assertEqual(result[0]['cause'],'retained_reassigned')
        self.assertEqual(result[0]['retained_peak_matched_truth'],2)
        self.assertEqual(result[1]['cause'],'suppressed')
        self.assertEqual((result[1]['suppressor_frame'],result[1]['distance_frames'],result[1]['suppressor_matched_truth']),
                         (10,2,2))


if __name__ == '__main__':
    unittest.main()
