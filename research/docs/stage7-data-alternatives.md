# Stage 7: alternative data discovery

Evidence checked 2026-11-15. These candidates have not yet established an accessible, independently precise, held-out continuous monophonic electric DI benchmark with confirmed fast changes and same-pitch repicks. No audio was analyzed for this discovery.

## EGDB-DI: best lead, not yet qualified

[Chen et al.](https://arxiv.org/abs/2202.09907) recorded a single professional player on one Stratocaster-type electric guitar, with a hex pickup and DI box. Its 240 performances total 118 minutes and include solo, arpeggio and comping tabs; they are not all monophonic. The original [author demo](https://ss12f32v.github.io/Guitar-Transcription/) links the [EGDB Drive folder](https://drive.google.com/drive/folders/1h9DrB4dk4QstgjNaHh7lL7IMeKdYw82_?usp=share_link). Availability of individual files, dataset license and reuse permissions remain unverified.

Labels are produced by per-string onset detection matched to expected tab notes and manually checked for all recordings. Offset labels are less reliable. This is not independent proof of onset timing accuracy: the paper does not report a timing error bound, and its use of a 50 ms evaluation tolerance does not establish a 50 ms label guarantee. Same-pitch repick counts, fast-change counts, and qualifying monophonic tracks are unknown. The published 8:1:1 split is not player-independent (one player); any held-out recordings and versions must be frozen before audio experiments and kept distinct from tuning. Recordings with amp rendering are variants of the same performances and must stay in the same split. To qualify this lead, verify access and license, identify untouched DI recordings and monophonic event counts from labels, and independently validate timing and repick completeness before making tight onset claims.

## GOAT: potentially useful, access constrained

The [authors' paper](https://zenodo.org/record/17706552/files/000076.pdf) describes 5.9 hours of real pickup DI by several players on different guitars, comprising 153 standard-tuning and 19 nonstandard-tuning items. It includes fine-aligned and quantized MIDI, tabs, and separate amplifier and synthetic RSE versions. The paper says tablatures are manually checked, then MIDI fine-aligned to performances; it does not establish a timing-error bound, counts of same-pitch repicks, fast changes, or strictly monophonic excerpts. Player identifiers and a player-disjoint split require verification. The [authors' repository](https://github.com/JackJamesLoth/GOAT-Dataset) says request access via [Zenodo](https://doi.org/10.5281/zenodo.15690894); no request was made. The [dataset catalog](https://datasetcatalog.nlm.nih.gov/dataset?q=0002114021) reports CC BY-NC 4.0 and research-only terms. The paper's CC BY license is not the dataset license. Keep DI separate from amp and RSE versions, and group all variants from the same performance together in any split.

## Other candidates

- [EG-IPT](https://nime.org/proc/nime2025_14/index.html) has a real DI channel, but the [archive description](https://explore.openaire.eu/search/dataset?pid=10.5281/zenodo.15205643) says isolated single-note performances by one guitarist: it cannot exercise merging adjacent peaks in a continuous phrase. License and event timing precision were not verified.
- [EG-Solo](https://bryanyu1997.github.io/EG-Solo_demo/) provides 76 YouTube solo clips with backing tracks, MIDI note/technique labels, and 6,833 notes. It is not isolated raw DI. Label precision and audio rights are unverified. The [annotation folder](https://drive.google.com/drive/folders/1iC7Q8imfKtesEfnaIz0hjWJU5680M2hy) is linked by the authors.
- [GuitarSet](https://zenodo.org/records/3371780) is acoustic, despite pickup recordings; not electric DI. [EGDB-PG](https://zenodo.org/records/19789500) is amp-rendered; not raw DI. They do not substitute for the requested signal.

No candidate above supports a verified tight 50 ms independent ground truth yet. In particular, the separate Guitar-TECHS timing caveat (up to 100 ms misalignment) cannot establish such a metric; uncertainty-aware bounds or independently corrected timestamps are needed.
