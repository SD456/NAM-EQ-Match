"""Reproduce one recorded NAM failure for listening, without a new search."""
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.stage2_benchmark import read_json,write_json,sha,paired_quality
from tonematch.audio import db_to_lin,load_audio
from tonematch.nam_backend import NamCapture


def rms(audio):
    return float(np.sqrt(np.mean(np.asarray(audio,dtype=float)**2)))


def main():
    folder = ROOT/'artifacts/listening-current'
    if folder.exists() and any(folder.iterdir()):
        raise FileExistsError('Listening pack exists; refusing overwrite')
    protected = read_json(ROOT/'artifacts/stage7/baseline-sha256.json')
    stage7 = read_json(ROOT/'artifacts/stage7/onsets/results.json')
    protected.update(stage7['plan']['source_hashes'])
    protected['artifacts/stage7/onsets/results.json'] = sha(ROOT/'artifacts/stage7/onsets/results.json')
    assert all(sha(ROOT/p)==v for p,v in protected.items())
    previous = read_json(ROOT/'artifacts/stage4/independent/results.json')
    row = next(x for x in previous['retrieval'] if not x['eq_on'] and not x['top1'])
    assert (row['case'],row['reference'],row['truth'],row['winner'],row['gain_db'])==('6_0_E0','T0',6,3,-7)
    di_meta = read_json(ROOT/'assets/stage4/di-provenance.json')
    clip = next(x for x in di_meta['segments'] if x['id']=='H0')
    library = read_json(ROOT/'assets/stage2/models-ir-provenance.json')
    models,irs = ({x['id']:x for x in library[key]} for key in ('models','irs'))
    source_hashes = {clip['path']:clip['sha256']}
    torch.set_num_threads(4)

    def capture(index):
        rig = previous['plan']['rigs'][index]
        model = models[rig['model_id']]
        source_hashes[model['path']] = model['sha256']
        if rig['cab_id']:
            cab = irs[rig['cab_id']]
            source_hashes[cab['path']] = cab['sha256']
        assert all(sha(ROOT/p)==v for p,v in source_hashes.items())
        return NamCapture(str(ROOT/model['path']),device='cuda',rig_type=model['rig_type'],
                          cab_ir_path=str(ROOT/irs[rig['cab_id']]['path']) if rig['cab_id'] else None)

    dry,sr = load_audio(str(ROOT/clip['path']),48000)
    assert sr==48000 and len(dry)==384000
    calibration_db = previous['plan']['config']['calibration_db']
    calibrated = dry.astype(np.float64)*db_to_lin(calibration_db)
    target = capture(row['truth']).process(calibrated*db_to_lin(row['truth_gain_db']))
    retrieved = capture(row['winner']).process(calibrated*db_to_lin(row['gain_db']))
    quality = paired_quality(retrieved,target)
    expected = row['holdout_bypass']['H0']['bypass']
    for key in ('mrstft_sc','ltas_mae_db'):
        np.testing.assert_allclose(quality[key],expected[key],rtol=0,atol=1e-6)
    # Listening gains act only after NAM; they never alter the model excitation.
    signals = [dry,target,retrieved]
    gains = [db_to_lin(-22)/rms(x[:len(dry)]) for x in signals]
    peak = max(float(np.max(np.abs(x)))*g for x,g in zip(signals,gains))
    common = min(1.0,db_to_lin(-3)/peak)
    length = max(map(len,signals))
    folder.mkdir(parents=True,exist_ok=True)
    files = []
    playback = []
    names = ('01-dry-di.wav','02-known-target-wet.wav','03-retrieved-wet.wav')
    for name,x,g in zip(names,signals,gains):
        y = np.pad(x.astype(float)*g*common,(0,length-len(x)))
        path = folder/name
        sf.write(path,y,sr,subtype='PCM_16')
        check,rate = sf.read(path)
        assert rate==sr and len(check)==length and np.isfinite(check).all()
        assert np.max(np.abs(check))<=db_to_lin(-3)+1/32768
        assert np.max(np.abs(check-y))<=1/32768
        playback.append(check)
        files.append(dict(file=path.relative_to(ROOT).as_posix(),sha256=sha(path),sample_rate=sr,
            channels=1,subtype='PCM_16',frames=length,duration=length/sr,
            original_render_frames=len(x),raw_rms_first_8s=rms(x[:len(dry)]),
            playback_gain_db=float(20*np.log10(g*common)),
            output_rms_first_8s=rms(check[:len(dry)]),peak_dbfs=float(20*np.log10(np.max(np.abs(check))))))
    sequence = np.concatenate([part for x in playback for part in (x,np.zeros(sr//2))])
    sf.write(folder/'00-dry-target-retrieved-sequence.wav',sequence,sr,subtype='PCM_16')

    # Optional label sonification uses saved peaks only, without re-running detection.
    onset_row = next(x for x in stage7['heldout']['rows'] if x['id']=='LP_Lick6_KN')
    original,rate = sf.read(ROOT/onset_row['wav'])
    source_hashes[onset_row['wav']] = sha(ROOT/onset_row['wav'])
    left = original*min(db_to_lin(-22)/rms(original),db_to_lin(-6)/np.max(np.abs(original)))
    click = .25*np.sin(2*np.pi*1500*np.arange(round(.008*rate))/rate)*np.hanning(round(.008*rate))
    labels = {'04-onsets-label.wav':[e['onset'] for e in onset_row['events']],
              '05-onsets-raw.wav':onset_row['methods']['raw64']['predicted_onsets'],
              '06-onsets-first80.wav':onset_row['methods']['first_r5']['predicted_onsets']}
    onset_files = []
    for name,times in labels.items():
        right = np.zeros(len(left))
        for time in times:
            start = round(time*rate)
            end = min(start+len(click),len(right))
            if start<end:
                right[start:end] += click[:end-start]
        y = np.column_stack([left,right])
        path = folder/name
        sf.write(path,y,rate,subtype='PCM_16')
        check,check_rate = sf.read(path)
        assert check_rate==rate and check.shape==y.shape and np.isfinite(check).all()
        assert np.max(np.abs(check))<.71
        onset_files.append(dict(file=path.relative_to(ROOT).as_posix(),sha256=sha(path),
            channels=2,sample_rate=rate,frames=len(y),duration=len(y)/rate,
            click_times_seconds=times,left='same dry guitar',right='diagnostic clicks only'))
    assert all(np.array_equal(sf.read(folder/onset_files[0]['file'].split('/')[-1])[0][:,0],
                              sf.read(ROOT/item['file'])[0][:,0]) for item in onset_files[1:])
    assert all(sha(ROOT/p)==v for p,v in protected.items())
    assert all(sha(ROOT/p)==v for p,v in source_hashes.items())
    write_json(folder/'manifest.json',dict(case=row['case'],reference_used_in_old_search=row['reference'],
        listening_di='H0, directinput_05 0-8 s; already-observed historical holdout',
        selection_rule='first stored stage4 E0 retrieval whose top1 is false; no new ranking',
        target_rig=previous['plan']['rigs'][row['truth']],retrieved_rig=previous['plan']['rigs'][row['winner']],
        calibration_db=calibration_db,target_trim_db=row['truth_gain_db'],retrieved_trim_db=row['gain_db'],
        eq='bypass',real_nam_device='cuda float32 without TF32',
        playback_policy='post-chain first-8s RMS alignment to -22dBFS plus shared attenuation to peak <=-3dBFS; linear only',
        common_attenuation_db=float(20*np.log10(common)),source_hashes=source_hashes,
        source_provenance=['assets/stage2/models-ir-provenance.json','assets/stage4/di-provenance.json','assets/stage7/data-provenance.json'],
        wet_quality_reproduced=quality,files=files,onset_files=onset_files,
        preserved_hash_count=len(protected),script_sha256=sha(__file__),
        scope='Listening illustration of an existing known-target failure, not Free Bird, new wet-to-NAM training, blind listening evidence or new validation. Click tracks add analysis labels only.'))
    print(dict(output=str(folder),quality=quality,files=[x['file'] for x in files],preserved=len(protected)))


if __name__=='__main__':
    main()
