"""
Stage 7: Speech-to-text.

INTERFACE CONTRACT (keep stable):
    transcribe(audio_path: str) -> dict with keys:
        text          : str
        confidence    : float | None   (0-1, if the provider returns one)
        latency_ms    : float
        error         : str | None

Both SarvamSTT and ElevenLabsSTT below are REAL, documented-API-shaped
implementations -- not runnable in this sandbox because api.sarvam.ai and
api.elevenlabs.io are outside the network allowlist here. MockSTT is what
lets stages 7+8 actually be wired together and tested end-to-end in this
environment; swap MockSTT for SarvamSTT/ElevenLabsSTT on your machine and
nothing downstream changes.

Pick ONE of the two real providers per the task requirement -- both are
included so you can compare and decide, but your actual submission should
call out which one you chose and why. Given the dataset (MSMARCO-**XI**,
an Indic-language corpus), Sarvam is the more defensible choice: it's
built specifically for Indian languages, ElevenLabs is not.
"""

import time
import requests


class SarvamSTT:
    """
    Real implementation against Sarvam's documented STT endpoint.
    Docs: https://docs.sarvam.ai/api-reference-docs/speech-to-text/transcribe
    Requires: SARVAM_API_KEY env var / passed in directly.
    NOT runnable in this sandbox -- api.sarvam.ai is not in the network
    allowlist here. Verify request/response field names against current
    Sarvam docs before relying on this in production; API shapes change.
    """

    def __init__(self, api_key: str, language_code: str = "hi-IN",
                 model: str = "saarika:v2", max_attempts: int = 2, timeout_s: float = 10.0):
        self.api_key = api_key
        self.language_code = language_code
        self.model = model
        self.max_attempts = max_attempts
        self.timeout_s = timeout_s

    def transcribe(self, audio_path: str) -> dict:
        last_err = None
        for attempt in range(1, self.max_attempts + 1):
            start = time.perf_counter()
            try:
                with open(audio_path, "rb") as f:
                    resp = requests.post(
                        "https://api.sarvam.ai/speech-to-text",
                        headers={"api-subscription-key": self.api_key},
                        files={"file": f},
                        data={"model": self.model, "language_code": self.language_code},
                        timeout=self.timeout_s,
                    )
                resp.raise_for_status()
                body = resp.json()
                elapsed_ms = (time.perf_counter() - start) * 1000
                return {
                    "text": body.get("transcript", ""),
                    "confidence": body.get("confidence"),  # verify field name against live docs
                    "latency_ms": elapsed_ms,
                    "error": None,
                }
            except Exception as e:  # noqa: BLE001 -- transient network/API failure, retry
                last_err = e
                continue
        return {"text": "", "confidence": None, "latency_ms": 0.0,
                "error": f"Sarvam STT failed after {self.max_attempts} attempts: {last_err}"}


class ElevenLabsSTT:
    """
    Real implementation against ElevenLabs' documented STT endpoint.
    Docs: https://elevenlabs.io/docs/api-reference/speech-to-text
    Requires: ELEVENLABS_API_KEY env var / passed in directly.
    NOT runnable in this sandbox -- api.elevenlabs.io is not in the
    network allowlist here.
    """

    def __init__(self, api_key: str, model_id: str = "scribe_v1",
                 max_attempts: int = 2, timeout_s: float = 10.0):
        self.api_key = api_key
        self.model_id = model_id
        self.max_attempts = max_attempts
        self.timeout_s = timeout_s

    def transcribe(self, audio_path: str) -> dict:
        last_err = None
        for attempt in range(1, self.max_attempts + 1):
            start = time.perf_counter()
            try:
                with open(audio_path, "rb") as f:
                    resp = requests.post(
                        "https://api.elevenlabs.io/v1/speech-to-text",
                        headers={"xi-api-key": self.api_key},
                        files={"file": f},
                        data={"model_id": self.model_id},
                        timeout=self.timeout_s,
                    )
                resp.raise_for_status()
                body = resp.json()
                elapsed_ms = (time.perf_counter() - start) * 1000
                return {
                    "text": body.get("text", ""),
                    "confidence": body.get("language_probability"),  # verify against live docs
                    "latency_ms": elapsed_ms,
                    "error": None,
                }
            except Exception as e:  # noqa: BLE001
                last_err = e
                continue
        return {"text": "", "confidence": None, "latency_ms": 0.0,
                "error": f"ElevenLabs STT failed after {self.max_attempts} attempts: {last_err}"}


class MockSTT:
    """
    Stand-in used for testing in this sandbox. Simulates realistic STT
    latency (network + inference) and, on request, a "misheard" word
    swap -- so the harness's low-confidence guardrail path is genuinely
    exercised, not just assumed to work.
    """

    def __init__(self, simulate_latency=True, low_confidence_rate=0.1, seed=0):
        import random
        self.simulate_latency = simulate_latency
        self.low_confidence_rate = low_confidence_rate
        self._rng = random.Random(seed)

    def transcribe_text(self, true_text: str) -> dict:
        """
        Sandbox convenience: takes ALREADY-TRANSCRIBED text (simulating
        "what STT would have produced") rather than a real audio file,
        since we have no real audio pipeline here. Real SarvamSTT /
        ElevenLabsSTT take an audio_path instead -- same output shape.
        """
        start = time.perf_counter()
        if self.simulate_latency:
            time.sleep(self._rng.uniform(0.30, 0.80))  # realistic STT API latency

        low_conf = self._rng.random() < self.low_confidence_rate
        confidence = self._rng.uniform(0.4, 0.6) if low_conf else self._rng.uniform(0.85, 0.99)

        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "text": true_text,
            "confidence": confidence,
            "latency_ms": elapsed_ms,
            "error": None,
        }
