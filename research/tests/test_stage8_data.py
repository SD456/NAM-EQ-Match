from collections import Counter
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import stage8_data


def test_inventory_and_group_splits():
    items = stage8_data.records()
    by_id = {r['id']: r for r in items}
    assert Counter(r['guitar'] for r in items) == {
        'Ibanez': 78, 'Fender': 84, 'Aristides': 6, 'LesPaul': 6, 'Sire': 18}
    assert Counter(r['type'] for r in items) == {'isolated': 156, 'lick': 18, 'excerpt': 18}
    for group, fold in stage8_data.folds(items).items():
        train, query, holdout = (set(fold[k]) for k in ('train', 'query', 'holdout'))
        assert len(train) == 192 - sum(r['guitar'] == group for r in items)
        assert query and holdout and not (train & query or train & holdout or query & holdout)
        assert all(by_id[i]['guitar'] != group for i in train)
        assert all(by_id[i]['guitar'] == group for i in query | holdout)
        assert {by_id[i]['parent'] for i in query}.isdisjoint(by_id[i]['parent'] for i in holdout)
        assert {by_id[i]['parent_hash'] for i in query}.isdisjoint(by_id[i]['parent_hash'] for i in holdout)
        if group != 'Sire':
            assert len(train | query | holdout) == 192
        else:
            assert query == {'stage4:T0', 'stage4:T1'}
            assert holdout == {'stage4:H0', 'stage4:H1'}
    fender = stage8_data.folds(items)['Fender']
    assert {by_id[i]['type'] for i in fender['query']} == {'isolated', 'lick'}
    assert {by_id[i]['type'] for i in fender['holdout']} == {'isolated', 'lick'}
    assert all(r['q_or_h'] == ('Q' if r['fret'] % 2 == 0 else 'H')
               for r in items if r['type'] == 'isolated')
