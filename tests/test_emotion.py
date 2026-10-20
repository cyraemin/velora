"""Emotion model wrapper: label mapping, window layout and compatibility with older results."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from velora import emotion as E  # noqa: E402


def test_labels_map_to_display_names():
    labels = ["生气/angry", "厌恶/disgusted", "恐惧/fearful", "开心/happy", "中立/neutral", "其他/other",
              "难过/sad", "吃惊/surprised", "<unk>"]
    scores = [0.05, 0.05, 0.05, 0.5, 0.1, 0.05, 0.1, 0.05, 0.05]
    probs = E._to_display(labels, scores)
    assert list(probs) == E.EMOTIONS
    assert probs["Happy"] == 0.5
    assert probs["Other"] == pytest.approx(0.1)   # "other" and "<unk>" are reported together
    assert sum(probs.values()) == pytest.approx(1.0)


def test_classes_of_older_four_class_result():
    old = {"probs": {"Happy": 0.1, "Neutral": 0.7, "Sad": 0.1, "Angry": 0.1}}
    assert E.classes(old) == ["Neutral", "Happy", "Sad", "Angry"]
    assert E.short_name({"model": "superb/wav2vec2-base-superb-er"}) == "wav2vec2 (IEMOCAP)"


def test_margin_and_colors():
    assert E.margin({"probs": {"Happy": 0.6, "Sad": 0.3, "Angry": 0.1}}) == pytest.approx(0.3)
    assert all(E.color(e).startswith("#") for e in E.EMOTIONS)
    assert E.color("not a class") == "#9A96B0"


@pytest.mark.parametrize("duration", [1.0, 4.0, 10.0, 60.0, 180.0])
def test_windows_cover_clip(duration):
    wins = E.windows(duration)
    assert 1 <= len(wins) <= E.MAX_WINDOWS
    assert wins[0][0] == 0.0
    assert wins[-1][1] <= duration + 1e-6 or len(wins) == 1


def test_dominant_at_handles_old_and_new_timelines():
    old = {"t": [1.0], "start": [0.0], "end": [2.0], "speech": [True],
           "Happy": [0.1], "Neutral": [0.7], "Sad": [0.1], "Angry": [0.1]}
    assert E.dominant_at(old, 0.0, 2.0) == "Neutral"
    new = dict(old, **{e: [0.0] for e in E.EMOTIONS})
    new["Fearful"] = [0.9]
    assert E.dominant_at(new, 0.0, 2.0) == "Fearful"
