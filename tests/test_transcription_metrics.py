import pytest

from src.transcription_metrics import TranscriptionMetrics, ZindiScore


def test_exact_transcriptions_receive_perfect_proxy_score():
    result = TranscriptionMetrics()(["First line", "Second line"], ["First line", "Second line"])

    assert result["weighted_wer"] == 0.0
    assert result["weighted_cer"] == 0.0
    assert result["zindi_score_proxy"] == 1.0


def test_unicode_is_normalized_before_scoring():
    composed = "Caf\u00e9"
    decomposed = "Cafe\u0301"

    assert TranscriptionMetrics()([composed], [decomposed])["zindi_score_proxy"] == 1.0


def test_score_proxy_uses_the_configured_normalizers():
    score = ZindiScore(word_normalizer=10.0, character_normalizer=20.0)

    assert score(weighted_wer=2.0, weighted_cer=4.0) == pytest.approx(0.8)


def test_mismatched_or_empty_inputs_are_rejected():
    metric = TranscriptionMetrics()

    with pytest.raises(ValueError, match="equal reference/prediction counts"):
        metric(["one"], [])
    with pytest.raises(ValueError, match="empty prediction set"):
        metric([], [])

