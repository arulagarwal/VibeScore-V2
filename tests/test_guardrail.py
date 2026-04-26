from unittest.mock import MagicMock

from src.agent_system import HallucinationGuardrail


def test_guardrail_passes_clean_response():
    kb = MagicMock()
    kb.valid_titles = {"sunrise city", "neon noir", "midnight coding"}

    guardrail = HallucinationGuardrail()
    response = (
        'Try **Sunrise City** by Neon Echo for a bright opener, '
        'then "Neon Noir" by Static Moth to drop into a moodier register.'
    )
    result = guardrail.validate(response, kb)

    assert result.is_clean is True
    assert result.flagged_titles == []
    assert result.safe_response == response


def test_guardrail_flags_unknown_title():
    kb = MagicMock()
    kb.valid_titles = {"sunrise city"}

    guardrail = HallucinationGuardrail()
    response = 'I recommend **Imaginary Song** for that vibe.'
    result = guardrail.validate(response, kb)

    assert result.is_clean is False
    assert "Imaginary Song" in result.flagged_titles
    assert "could not be verified" in result.safe_response
    assert response in result.safe_response
