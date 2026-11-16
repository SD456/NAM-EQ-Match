# Stage 7 IDMT DI selection

The frozen selection comprises 18 original dataset2 WAV/XML pairs: Aristides 010 (AR), Fender Stratocaster (FS), and Gibson Les Paul (LP), all six ordinary picked (`PK`) and normal expression (`NO`) licks. Calibration uses lick identities 1–3 and heldout uses 4–6. The same lick is played on three guitars, so the nine files per role are **three musical patterns**, not nine independent musical patterns. The same guitarist played dataset2. These are heldout recordings and score identities, not heldout players or guitars.

The [authors' dataset description](../assets/stage6/source-description.pdf) identifies dataset2 licks, guitar models, recording setup, and annotation semantics: the electric instruments were plugged directly into the interface, while the microphone exception is in dataset4. Dataset1 is isolated notes, dataset4 includes acoustic/microphone recordings and lacks comparable note-pitch labels; neither is selected. Dataset2 licks 7–10 and 12 have near-simultaneous chord onsets, and lick11 uses special expression; initialization and alternate technique recordings are excluded. Dataset3 has only two onset-monophonic recordings in the inspected labels and no adjacent same-pitch repetitions or fast pitch changes, so it was not included. The source's XML offsets can overlap following onsets: the selection excludes simultaneous annotated onset groups (less than 40 ms apart), not natural overlapping note decays.

The [Fraunhofer page](https://www.idmt.fraunhofer.de/en/publications/datasets/guitar.html) states the license is CC BY-NC-ND 4.0. The [Zenodo record](https://zenodo.org/records/7544110) publishes the archive MD5. Its web summary appears to swap dataset1 and dataset2 descriptions; the versioned source PDF and observed ZIP paths/XML/WAV headers determine the selection here. All selected WAVs have mono, 44,100 Hz, 24-bit PCM headers.

## Label-only coverage

Counts are computed from successive original XML events in onset order. A fast pitch change has different MIDI pitches and onset interval at most 200 ms; a same-pitch repeat is any interval, and a fast repeat is at most 200 ms. Bins count all consecutive pairs, not only changes. Boundaries are inclusive at the right end. Each row combines three guitar performances per score; observations are therefore correlated.

| Role | Clips | Events | Fast changes | Same-pitch repeats | Fast repeats | <=80 ms | (80,180] ms | (180,200] ms | >200 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Calibration | 9 | 141 | 16 | 12 | 0 | 0 | 8 | 8 | 116 |
| Heldout | 9 | 96 | 5 | 9 | 0 | 0 | 1 | 4 | 82 |

Neither role meets 20 fast changes and 20 repeats; neither has a repeat within 200 ms or any adjacent onsets within 80 ms. This set supports descriptive diagnostics, not a sufficient demonstration of safe fast-note merging. No waveform-derived onset, pitch, or spectral features were used to select these files.

## Reproduction and integrity

Run `python scripts/prepare_stage7_di.py` from the project root. It requests only the ZIP central directory and the selected entries using TLS-verified HTTP Range requests, and checks each extracted ZIP entry's size and CRC32. It stores original WAV and XML files under `assets/stage7/di` and `assets/stage7/labels` with roles in separate folders. `assets/stage7/data-provenance.json` records source paths, byte ranges, ZIP CRC32, SHA-256, native headers, every XML event, and label-derived counts. The full archive is not downloaded, so the publisher's full archive MD5 is recorded but **not verified**; CRC32 is not cryptographic source authentication.
