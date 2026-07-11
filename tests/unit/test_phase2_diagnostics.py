import pytest

from eleanity.comparators.diff import compare_generation, compare_logits
from eleanity.models.schemas import ParityProfile, ParityResult, Scenario


@pytest.mark.parametrize(
    ("profile", "expected"), [
        (ParityProfile.STRICT, ParityResult.DIVERGENT),
        (ParityProfile.QUANTIZED, ParityResult.PASS_WITH_TOLERANCE),
        (ParityProfile.FUNCTIONAL, ParityResult.PASS_WITH_TOLERANCE),
        (ParityProfile.API_CONFORMANCE, ParityResult.DIVERGENT),
    ],
)
def test_logits_tolerance_follows_profile(profile, expected):
    scenario = Scenario(name="p", messages=[{"role": "user", "content": "x"}], parity_profile=profile)
    assert compare_logits([1.0], [1.01], scenario.tolerance).result == expected


def test_generation_diff_is_tokenwise():
    result = compare_generation([1, 2, 3], [1, 9, 3])
    assert result.result == ParityResult.DIVERGENT
    assert result.details["first_difference"] == 1
