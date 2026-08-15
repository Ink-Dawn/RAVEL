from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from validate_profile_launch import parse_fingerprint  # noqa: E402


def test_parse_fingerprint_accepts_checked_in_legacy_format():
    _model, version, fields = parse_fingerprint(
        "Model|vllm-0.7.0|dtype=float|max_num_seqs=64"
    )
    assert version == "0.7.0"
    assert fields["max_num_seqs"] == "64"


def test_parse_fingerprint_accepts_current_calibrator_format():
    _model, version, fields = parse_fingerprint(
        "Model|vllm=0.7.0|dtype=float|max_num_seqs=64"
    )
    assert version == "0.7.0"
    assert fields["dtype"] == "float"
