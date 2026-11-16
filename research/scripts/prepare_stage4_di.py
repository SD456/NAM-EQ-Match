"""Extract CRC-verified Guitar-TECHS recordings and fixed stage 4 segments."""

import hashlib
import json
import struct
import zlib
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets/stage4/di"
FILES = {2: (5211047, 10752690, 0xcc8af592), 3: (3958205, 8436690, 0xffb0d954),
         4: (5158508, 10680690, 0x187c8da7), 5: (7091601, 15840690, 0xaf7d3845)}
RANGES = {2: (114750295, 119961465), 3: (110791741, 114750068),
          4: (89278355, 94436985), 5: (82186404, 89278127)}
SPLIT = {2: ("probe", "P", 3), 3: ("calibration_reference", "C", 2),
         4: ("test_reference", "T", 2), 5: ("holdout", "H", 2)}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    recordings, segments = [], []
    tail = (OUT / "zip-tail.bin").read_bytes()
    central = tail.find(b"PK\x01\x02")
    directory = {}
    while tail[central:central + 4] == b"PK\x01\x02":
        entry = struct.unpack_from("<4s6H3I5H2I", tail, central)
        name_len, extra_len, comment_len = entry[10:13]
        name = tail[central + 46:central + 46 + name_len].decode()
        directory[name] = (entry[8], entry[9], entry[7], entry[-1])
        central += 46 + name_len + extra_len + comment_len
    for number, (compressed_size, size, crc) in FILES.items():
        filename = f"P3_music/audio/directinput/directinput_{number:02}.wav"
        assert directory[filename] == (compressed_size, size, crc, RANGES[number][0])
        data = (OUT / f"range{number:02}.bin").read_bytes()
        assert data[:4] == b"PK\x03\x04"
        header = struct.unpack_from("<4s5H3I2H", data)
        assert header[3] == 8
        offset = 30 + header[-2] + header[-1]
        assert data[30:30 + header[-2]].decode() == filename
        raw = zlib.decompress(data[offset:offset + compressed_size], -15)
        assert len(raw) == size and zlib.crc32(raw) == crc
        raw_path = OUT / f"directinput_{number:02}.raw.wav"
        raw_path.write_bytes(raw)
        audio, rate = sf.read(raw_path, dtype="float32", always_2d=True)
        assert audio.shape[1] == 2 and np.array_equal(audio[:, 0], audio[:, 1])
        role, prefix, count = SPLIT[number]
        assert len(audio) >= count * 8 * rate
        recording_id = f"directinput_{number:02}"
        recordings.append({"id": recording_id, "path": raw_path.relative_to(ROOT).as_posix(),
                           "sha256": sha256(raw_path), "sample_rate": rate,
                           "channels": audio.shape[1], "frames": len(audio),
                           "zip_crc32": f"{crc:08x}", "zip_member": filename,
                           "archive_byte_range_inclusive": RANGES[number]})
        for i in range(count):
            start, end = i * 8 * rate, (i + 1) * 8 * rate
            samples = audio[start:end, 0]
            assert len(samples) == 8 * rate and np.max(np.abs(samples)) > 0
            path = OUT / f"{prefix}{i}.wav"
            sf.write(path, samples, rate, subtype="FLOAT")
            check, check_rate = sf.read(path, dtype="float32")
            assert check_rate == rate and np.array_equal(check, samples)
            segments.append({"id": f"{prefix}{i}", "role": role, "recording_id": recording_id,
                             "path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path),
                             "sample_rate": rate, "channels": 1, "frames": len(samples),
                             "start_frame": start, "end_frame": end,
                             "rms": float(np.sqrt(np.mean(samples ** 2))),
                             "peak": float(np.max(np.abs(samples)))})
    provenance = {"sources": [{"title": "Guitar-TECHS", "url": "https://zenodo.org/records/14963133",
                               "authors": ["Hegel Emmanuel Pedroza Villalobos", "Termeh Taheri", "Wallace Abreu", "Ryan Corey", "Iran R. Roman"],
                               "license": "CC BY 4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/",
                               "archive": "P3_music.zip", "archive_md5_published": "071ba80aecf00f4a31fbd167b3f22198",
                               "archive_url": "https://zenodo.org/records/14963133/files/P3_music.zip",
                               "central_directory_evidence": {"path": "assets/stage4/di/zip-tail.bin", "sha256": sha256(OUT / "zip-tail.bin"), "archive_byte_range_inclusive": [129439553, 129505088]},
                               "retrieval": "Local DNS returned 0.0.0.0 and direct lookup failed. HTTPS range requests used a publicly listed CERN IP via curl --resolve; the local certificate trust chain failed, so TLS verification was disabled. CRC-32 checks detect accidental corruption but do not authenticate the download.",
                               "integrity": "Individual ZIP entries validated against CRC-32 in the official archive central directory; full archive MD5 was not checked."}],
                  "recordings": recordings, "segments": segments,
                  "selection_and_processing": "Fixed before scoring: 02 probe 0-8/8-16/16-24 s, 03 calibration 0-8/8-16 s, 04 test 0-8/8-16 s, 05 holdout 0-8/8-16 s. Sample ranges are half-open. Left channel selected after exact stereo equality check; float32 WAV, no normalization, resampling, fades, gain, denoising or effects. Recording gain and physical plucking strength are unknown; RMS is not a strength label. Player 3 used Sire T7 and UMC202HD as described at https://guitar-techs.github.io/."}
    (ROOT / "assets/stage4/di-provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
