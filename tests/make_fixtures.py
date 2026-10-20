"""Create test audio fixtures in every supported format from one WAV file.

Usage: python tests/make_fixtures.py <source.wav> <out_dir>
"""
import sys
from pathlib import Path

import av
import numpy as np
import soundfile as sf


def encode_av(y: np.ndarray, sr: int, path: Path, container: str, codec: str, out_sr: int):
    with av.open(str(path), "w", format=container) as out:
        st = out.add_stream(codec, rate=out_sr)
        st.layout = "mono"
        pcm = (np.clip(y, -1, 1) * 32767).astype(np.int16)
        frame = av.AudioFrame.from_ndarray(pcm[None, :], format="s16", layout="mono")
        frame.sample_rate = sr
        resampler = av.AudioResampler(format=st.format.name, layout="mono", rate=out_sr)
        for f in resampler.resample(frame):
            for pkt in st.encode(f):
                out.mux(pkt)
        for f in resampler.resample(None):
            for pkt in st.encode(f):
                out.mux(pkt)
        for pkt in st.encode(None):
            out.mux(pkt)


def main(src: str, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    y, sr = sf.read(src, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    sf.write(out / "fx.wav", y, sr)
    sf.write(out / "fx.flac", y, sr)
    sf.write(out / "fx.ogg", y, sr, format="OGG", subtype="VORBIS")
    sf.write(out / "fx.mp3", y, sr, format="MP3", subtype="MPEG_LAYER_III")
    encode_av(y, sr, out / "fx.m4a", "mp4", "aac", 44100)
    encode_av(y, sr, out / "fx.webm", "webm", "libopus", 48000)
    stereo = np.stack([y, y * 0.5], axis=1)
    sf.write(out / "fx_stereo_44k.wav", np.asarray(
        __import__("librosa").resample(stereo.T, orig_sr=sr, target_sr=44100)).T, 44100)
    (out / "fx_corrupt.wav").write_bytes(b"RIFF\x00\x00\x00\x00WAVEgarbage" * 20)
    (out / "fx_empty.wav").write_bytes(b"")
    sf.write(out / "fx_silence.wav", np.zeros(sr * 2, dtype="float32"), sr)
    sf.write(out / "fx_short.wav", y[: int(0.2 * sr)], sr)
    for p in sorted(out.glob("fx*")):
        print(f"{p.name:20s} {p.stat().st_size:>9,d} bytes")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
