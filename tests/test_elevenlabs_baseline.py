import pytest

from scripts.synthesize_elevenlabs_baseline import MODEL_ID, OUTPUT_FORMAT, VOICE_ID, synthesize


def test_fixed_elevenlabs_configuration():
    assert MODEL_ID == "eleven_multilingual_v2"
    assert VOICE_ID == "JBFqnCBsd6RMkjVDRZzb"
    assert OUTPUT_FORMAT == "mp3_44100_128"


def test_synthesis_does_not_retry_nontransient_http(monkeypatch):
    class Failure(Exception):
        pass

    def fail(*args, **kwargs):
        raise Failure

    monkeypatch.setattr("urllib.request.urlopen", fail)
    with pytest.raises(Failure):
        synthesize("test", "secret", timeout=1, max_attempts=1)
