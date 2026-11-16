import numpy as np

from scripts.stage8_benchmark import attach_quality, grouped_mean, bootstrap_indices


def test_oracle_is_chosen_after_holdout_aggregation():
    # Two H events favour different candidates; there is no per-event free oracle.
    table = np.array([[[0.5, 0.4, 0.6, 0.3], [0.7, 0.6, 0.8, 0.5]]])
    rows = [dict(guitar='G', type='lick', teacher_pos=0, eq=0, gain_mode='unknown', true_gi=0, selected=0)]
    r = attach_quality(rows, {('G', 'lick'): {'sc': table, 'ltas': table}}, 2)[0]
    assert r['oracle'] == 3 and np.isclose(r['regret'], .2)
    rows[0]['gain_mode'] = 'known'
    r = attach_quality(rows, {('G', 'lick'): {'sc': table, 'ltas': table}}, 2)[0]
    assert r['oracle'] == 0 and r['regret'] == 0


def test_macro_weights_parents_types_and_guitars_not_slices():
    common = dict(teacher=0, true_gi=0)
    rows = [dict(**common, guitar='F', type='note', parent='a', value=0)] * 30
    rows += [dict(**common, guitar='F', type='note', parent='b', value=2)]
    rows += [dict(**common, guitar='F', type='lick', parent='c', value=3)]
    rows += [dict(**common, guitar='S', type='excerpt', parent='d', value=0)]
    result = grouped_mean(rows, 'value')
    assert result == {'macro': 1.0, 'folds': {'F': 2.0, 'S': 0.0}}


def test_bootstrap_keeps_parent_slices_together():
    items = [dict(guitar='A', parent='p'), dict(guitar='A', parent='p'),
             dict(guitar='A', parent='q'), dict(guitar='B', parent='r')]
    for seed in range(10):
        sample = bootstrap_indices(items, np.arange(4), np.random.default_rng(seed))
        assert np.sum(sample == 0) == np.sum(sample == 1)
        assert set(sample) <= set(range(4))
