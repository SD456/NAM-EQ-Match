# Stage 6 data discovery: Guitar-TECHS

Discovery only; no waveform, pitch, onset or listening experiment has been performed. The frozen experiment plan must precede such work.

## Source and contents

The [authors' dataset site](https://guitar-techs.github.io/) explicitly licenses **all data** under CC BY 4.0; preserve attribution and the license link when using or redistributing it. The [Zenodo dataset record](https://zenodo.org/records/14963133) requests citation of Pedroza et al., Guitar-TECHS, ICASSP 2025, DOI 10.1109/ICASSP49660.2025.10887996. P1 and P2 are separate players with different guitars and recording setups. Their guitars used a Fishman Triple Play Connect pickup to capture MIDI per string independently of the DI audio path. These are measured performances, not synthetic samples.

The [paper](https://arxiv.org/html/2501.03720) describes 138 string/fret positions per player, four seconds per tone, six-track MIDI at 9600 ticks per quarter note, with note onset and duration; its stated audio specification is 48 kHz 32-bit float WAV. These are publication claims, not verified file headers or parsed MIDI. MIDI pitch and timing originate in an automated pickup, not manually verified ground truth. The archives have one continuous DI WAV and one MIDI per player, according to Zenodo previews: [P1](https://zenodo.org/records/14963133/preview/P1_singlenotes.zip?include_deleted=0), [P2](https://zenodo.org/records/14963133/preview/P2_singlenotes.zip?include_deleted=0). The archive paths are `P{1,2}_singlenotes/audio/directinput/directinput_allsinglenotes.wav` and `P{1,2}_singlenotes/audio/midi/midi_allsinglenotes.mid`. Other signals are not needed for this task.

| Archive | Official download | Bytes | Published MD5 |
| --- | --- | ---: | --- |
| P1 | [P1_singlenotes.zip](https://zenodo.org/records/14963133/files/P1_singlenotes.zip?download=1) | 108626613 | ca0c4674dde3805574685a313f7c39eb |
| P2 | [P2_singlenotes.zip](https://zenodo.org/records/14963133/files/P2_singlenotes.zip?download=1) | 116133457 | 40fbf03d8b04bb2cf42df20f36dc2254 |

## Acceptance and limitations

Conditionally accept for sustained single-note pitch calibration, pending a verified download, inspection of actual MIDI events and audio headers, and a frozen plan. Reserve one player as a held-out player; all windows from each recording and note event must stay together. Two players mean only one independent held-out player per direction; this is not broad population validation. The documentation does not establish repeated takes per string/fret: pitch recurrence on different strings or another player is not the same as repeated independent takes. Do not assume a repeat-based split is available. Label-derived pitch and silence regions need quality checks and exclusion rules; report actual event counts and class coverage once inspected.

**Onset limitation:** the Zenodo record warns of up to 100 ms signal misalignment. Although the paper/site describe synchronization, the record's explicit warning takes precedence for planning. It does not establish whether DI-to-MIDI error is constant, its direction, or exact magnitude. The raw MIDI onset is therefore not a validated sample-accurate reference; a short-window onset threshold or millisecond latency claim cannot be calibrated or validated against it without an independently justified alignment and uncertainty assessment. Do not optimize alignment against the same test labels and claim an independent onset result. Label pitch is nominal MIDI note, not a measured continuous F0 contour. Compare long-window pitch with short-window onset only under these limits; report unsupported precision and omit claims the labels cannot support.

## Acquisition status

Both archive links and published checksums are visible in the official record; the actual archive bytes have **not** been acquired or hashed here. System DNS failed for zenodo.org; a DNS query to 8.8.8.8 returned 0.0.0.0/::, and HTTPS DNS-over-HTTPS to 1.1.1.1 failed certificate validation with SEC_E_UNTRUSTED_ROOT. No TLS bypass was used. Retry on a trusted network, verify TLS normally, verify byte count and published MD5, then inspect archive paths, MIDI format and WAV headers before any experiment. Published MD5 checks accidental corruption but is not a cryptographic signature. No P3 music or Free Bird is included.
