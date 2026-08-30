"""Tests measured cost priors for ILS operation combinations.

The affordability calculation, safety margin, and prior accessor are pinned so
overstated costs do not reject combinations that fit the remaining budget.
"""
import pytest

from optimizer import ils_polish as ip


# ---- the eval's numbers, verbatim from the preview harness log ----------------
EVAL_BASIS_S = 307.0        # "measured cost basis ARMED: 307s per Explore-equivalent"
EVAL_REMAINING_S = 1000.0   # "> 1000s remaining"
EVAL_LOGGED_EST_S = 1058.0  # "est 1058s"
MARGIN = ip.MEASURED_BASIS_MARGIN   # 1.15


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("FPL26_ILS_MEASURED_PRIORS", raising=False)


def _est(pd):
    return EVAL_BASIS_S * ip.combo_cost_prior(pd) * MARGIN


# ---- DEFAULT OFF = byte-identical -------------------------------------------

def test_default_off_returns_the_shipped_table():
    assert ip.combo_cost_prior("ExtraNetDelay_high") == 3.0
    assert ip.combo_cost_prior(ip.LASTMILE_PD) == 2.5
    assert ip.combo_cost_prior("Explore") == 1.0


def test_default_off_reproduces_the_eval_refusal():
    """The shipped default must still refuse, or we have not reproduced the bug."""
    est = _est("ExtraNetDelay_high")
    assert abs(est - EVAL_LOGGED_EST_S) <= 2.0, (
        f"expected ~{EVAL_LOGGED_EST_S}s (the eval's logged est), got {est:.0f}s")
    assert est > EVAL_REMAINING_S, "shipped prior must refuse, as the eval did"


def test_unknown_directive_falls_back_to_one():
    assert ip.combo_cost_prior("NoSuchDirective") == 1.0


@pytest.mark.parametrize("value", ["0", "off", "false", "no", "", "  "])
def test_only_truthy_values_arm_it(monkeypatch, value):
    monkeypatch.setenv("FPL26_ILS_MEASURED_PRIORS", value)
    assert ip.combo_cost_prior("ExtraNetDelay_high") == 3.0


# ---- ARMED -------------------------------------------------------------------

@pytest.mark.parametrize("value", ["1", "true", "on", "yes", "TRUE", " On "])
def test_armed_returns_the_measured_priors(monkeypatch, value):
    monkeypatch.setenv("FPL26_ILS_MEASURED_PRIORS", value)
    assert ip.combo_cost_prior("ExtraNetDelay_high") == 2.10


def test_armed_flips_the_eval_decision(monkeypatch):
    """THE test: the same 307s basis and 1000s window must now AFFORD the cycle
    that earns optical its gain."""
    monkeypatch.setenv("FPL26_ILS_MEASURED_PRIORS", "1")
    est = _est("ExtraNetDelay_high")
    assert est <= EVAL_REMAINING_S, (
        f"measured prior must make the eval's refusal affordable; est={est:.0f}s")
    assert est == pytest.approx(741.0, abs=2.0)


def test_armed_leaves_every_other_prior_untouched(monkeypatch):
    """Verify prior updates are limited to the two overstated entries.

    All other priors remain unchanged; increasing underpriced entries would
    create additional affordability refusals.
    """
    monkeypatch.setenv("FPL26_ILS_MEASURED_PRIORS", "1")
    for pd in ("Explore", ip.LASTMILE_PD, ip.ROUTE_ONLY_PD, ip.PARTIAL_RUIN_PD,
               ip.ROUTE_REROLL_PD, ip.INCR_ROUTE_PD, "ExtraTimingOpt",
               "AltSpreadLogic_high", "AltSpreadLogic_medium",
               "ExtraNetDelay_low", "SSI_SpreadLogic_high",
               "EarlyBlockPlacement"):
        assert ip.combo_cost_prior(pd) == ip.COMBO_COST_PRIOR.get(pd, 1.0), pd


def test_measured_priors_only_ever_LOWER_a_prior(monkeypatch):
    """Directional invariant: this change may only ever make the gate more
    permissive. If a future edit raises one, it is a different change and must
    be argued separately."""
    monkeypatch.setenv("FPL26_ILS_MEASURED_PRIORS", "1")
    for pd, measured in ip.MEASURED_COMBO_COST_PRIOR.items():
        assert measured <= ip.COMBO_COST_PRIOR.get(pd, 1.0), pd


def test_measured_values_match_the_corpus_medians():
    """Pin the measured numbers to what the corpus actually said, so a future
    tweak has to re-measure rather than nudge."""
    assert ip.MEASURED_COMBO_COST_PRIOR["ExtraNetDelay_high"] == 2.10


def test_lastmile_is_deliberately_NOT_in_the_measured_table():
    """Verify `LASTMILE` is excluded from measured-prior overrides.

    A more accurate cost estimate does not justify an override unless admitting
    the operation is expected to improve the optimization result.
    """
    assert ip.LASTMILE_PD not in ip.MEASURED_COMBO_COST_PRIOR
    assert ip.combo_cost_prior(ip.LASTMILE_PD) == 2.5


def test_accessor_is_the_only_read_path():
    """The half-deploy lesson applied to a constant: if an affordability
    site reads the raw table it silently keeps the old behaviour when armed."""
    import inspect
    src = inspect.getsource(ip)
    reads = [ln.strip() for ln in src.splitlines()
             if "COMBO_COST_PRIOR.get(" in ln and "MEASURED_COMBO" not in ln]
    # Exactly one: the accessor's own fallback. Any other is an affordability
    # site that would silently keep shipped behaviour when the flag is armed.
    assert reads == ["return COMBO_COST_PRIOR.get(pd, 1.0)"], (
        f"raw-table reads outside the accessor: {reads}")
