"""`reactor_separator_recycle` through the case interface selects exactly the plants the study scripts build."""

from __future__ import annotations

import numpy as np
import pytest

import plantbench as pb
from plantbench.cases import load_case
from plantbench.cases.reactor_separator_recycle import hen, integrated, plant
from plantbench.cases.reactor_separator_recycle import scenarios as sc
from plantbench.cases.reactor_separator_recycle.definition import CASE
from plantbench.cases.reactor_separator_recycle.parameters import PlantParameters
from plantbench.core import control as ctl
from plantbench.core.case import build, features_of


@pytest.fixture(scope="module")
def design():
    return plant.solve_design(PlantParameters(), V_boil=28.0)


def test_reference_configuration_is_the_table_19_1_plant(design):
    setup = pb.build(CASE)
    ref = np.sort_complex(ctl.closed_loop_spectrum(design, sc.plantwide(design)))
    got = np.sort_complex(setup.spectrum())
    assert np.array_equal(setup.design.x, design.x)
    assert np.allclose(got, ref, rtol=0, atol=1e-12)


@pytest.mark.parametrize("network", ["D1", "D3", "D4", "D5"])
def test_a_network_option_is_the_stage_2_plant(design, network):
    setup = pb.build(CASE, CASE.config(options={"network": network}))
    integ = integrated.build(design, network)
    ref = np.sort_complex(ctl.closed_loop_spectrum(integ, integrated.stage2(integ)))
    assert np.allclose(np.sort_complex(setup.spectrum()), ref, rtol=0, atol=1e-12)
    assert len(CASE.state_names(setup.design)) == len(setup.design.x)


def test_state_names_cover_the_base_plant(design):
    names = CASE.state_names(design)
    assert len(names) == len(design.x) == 134
    assert names[4] == "T" and names[6] == "M_D" and names[-1] == "xB_B"
    assert len(set(names)) == len(names)


def test_d4_at_its_own_share_is_d4():
    a = pb.build(CASE, CASE.config(options={"network": "D4"}))
    b = pb.build(CASE, CASE.config(options={"network": "D4", "pump_around_share": 0.7}))
    assert np.allclose(np.sort_complex(a.spectrum()), np.sort_complex(b.spectrum()), atol=1e-12)


@pytest.mark.parametrize("config, error", [
    (dict(structure="recycle free", options={"network": "D3"}), ValueError),
    (dict(options={"pump_around_share": 0.3}), ValueError),
    (dict(options={"network": "D3", "pump_around_share": 0.3}), ValueError),
    (dict(options={"network": "D2"}), ValueError),
    (dict(disturbance={"name": "fresh_feed_temperature", "change": -10.0}), ValueError),
    (dict(params={"reactor.nope": 1.0}), KeyError),
])
def test_configurations_the_case_does_not_have_are_refused(config, error):
    with pytest.raises(error):
        pb.build(CASE, CASE.config(**config))


def test_a_run_records_named_measurements_and_derived_inputs():
    tr = pb.run(CASE, CASE.config(disturbance={"name": "throughput", "fraction": 0.05}),
                t_end=200.0, dt=10.0)
    assert np.allclose(tr.y["reactor T"], tr.state("T"))
    assert np.allclose(tr.y["recycle"], tr.u["D"] - tr.u["P"])
    assert "tray 17 T" in tr.y and np.allclose(tr.u["recycle"], tr.y["recycle"])
    assert tr.u["F_out"][-1] == pytest.approx(1.05 * tr.u["F_out"][0])


def test_a_run_that_once_stepped_over_its_upset_stays_finite():
    """Regression: without restarting at the upset, the solver took one step from rest
    across it and landed on an unphysical state; the run went NaN at 20 min."""
    config = CASE.config(options={"network": "D4", "pump_around_share": 0.5},
                         instruments={"reactor C_A": {"order": 2,
                                                      "deadtime": 8.167194937788812}},
                         tuning={"reactor T": {"Kc": -1.6129760432936315,
                                               "tau_I": 5.959930369865621}},
                         disturbance={"name": "inert", "fraction": 0.5, "t": 30.0})
    tr = pb.run(CASE, config, t_end=1500.0, dt=2.0)
    assert np.isfinite(tr.x).all()


# -- the economics feature and the network's column pressure -----------------------------

ECONOMICS = {  # utilities $/h, energy GWP t/h, recovered GJ/h, from studies/reactor_separator_recycle/results
    "none": (477.0, 5.099, 0.0),
    "D1": (477.0, 5.099, 0.0),
    "D3": (381.0, 4.049, 26.5),
    "D5": (388.0, 4.121, 14.8),
}


def _setup(**options):
    case = load_case("reactor_separator_recycle")
    return build(case, case.validate({"structure": "effluent fixed + cascade",
                                      "options": options}))


def _economics(**options):
    return features_of(load_case("reactor_separator_recycle"))["economics"](_setup(**options))


@pytest.mark.parametrize("network", sorted(ECONOMICS))
def test_the_economics_feature_reproduces_the_reference_designs(network):
    """The feature must agree with the design ladder of the design comparison, to the printed digits."""
    cost, gwp, recovered = ECONOMICS[network]
    e = _economics(network=network)
    assert e["utility_cost"] == pytest.approx(cost, abs=0.5)
    assert e["gwp_energy"] == pytest.approx(gwp, abs=5e-4)
    # GJ/h, as the feature documents: the network holds its duties in kJ/min, and a test
    # that converted here instead would pass whichever unit the feature happened to record.
    assert e["recovered"] == pytest.approx(recovered, abs=0.05)


@pytest.mark.parametrize("share,cost,gwp", [(0.0, 576.0, 4.352), (0.3, 550.0, 4.072),
                                            (0.5, 533.0, 3.886), (0.7, 516.0, 3.700)])
def test_the_economics_feature_reproduces_the_pump_around_trade(share, cost, gwp):
    e = _economics(network="D4", pump_around_share=share)
    assert e["utility_cost"] == pytest.approx(cost, abs=0.5)
    assert e["gwp_energy"] == pytest.approx(gwp, abs=5e-4)


def test_the_heat_recovered_is_on_the_same_per_hour_basis_as_the_rest():
    """D4 at full pump-around recovers 67.7 GJ/h, not the 1.13 million kJ/min behind it."""
    assert _economics(network="D4", pump_around_share=0.7)["recovered"] == \
        pytest.approx(67.7, abs=0.05)


def test_the_economics_feature_needs_no_run():
    """It is read from the design point, so a plant that would cycle still has objectives."""
    e = _economics(network="D4", pump_around_share=0.7)
    assert e["profit"] > 0 and e["gwp_energy"] > 0


def test_the_network_pressure_defaults_to_the_one_the_network_was_designed_for():
    for name in ("D3", "D4", "D5"):
        assert _setup(network=name).design.pressure == pytest.approx(hen.DESIGNS[name].pressure)


def test_setting_the_network_pressure_moves_the_design_and_not_the_run_id():
    """`column.P_network` is a parameter, so it never enters a configuration by default."""
    case = load_case("reactor_separator_recycle")
    plain = {"structure": "effluent fixed + cascade", "options": {"network": "D3"}}
    moved = {**plain, "params": {"column.P_network": 0.3}}
    assert case.validate(plain).key() != case.validate(moved).key()
    # The default configuration hashes as it did before the parameter existed.
    assert "P_network" not in case.validate(plain).to_dict()["params"]

    design = build(case, case.validate(moved)).design
    assert design.pressure == pytest.approx(0.3)
    # Lower pressure brings the bottoms further below the reactor, so more heat is recovered.
    assert design.sized.recovered > _setup(network="D3").design.sized.recovered


def test_the_network_pressure_must_be_positive():
    case = load_case("reactor_separator_recycle")
    with pytest.raises(ValueError, match="must be positive"):
        build(case, case.validate({"structure": "effluent fixed + cascade",
                                   "options": {"network": "D3"},
                                   "params": {"column.P_network": -1.0}}))


# The first mismatch draw of rrl's stress set on this case (rrl notes, entry 7), rounded.
MISMATCH = {"reactor.k0": 3.0e11, "reactor.U": 300.0, "reactor.dH": 8.5e4}


def _loops(setup):
    return {e.name: (e.bias, e.setpoint) for e in setup.structure if isinstance(e, ctl.Loop)}


def test_a_mismatch_keeps_the_loops_of_the_nominal_design():
    """Under `params` the design follows the parameters and the tray temperature set-point
    moves; under `mismatch` every bias and set-point is the nominal design's."""
    nominal = pb.build(CASE)
    mismatched = pb.build(CASE, CASE.config(mismatch=MISMATCH))
    moved = pb.build(CASE, CASE.config(params=MISMATCH))
    assert _loops(mismatched) == _loops(nominal)
    assert _loops(moved)["tray T"][1] != pytest.approx(_loops(nominal)["tray T"][1], abs=0.1)
    assert mismatched.design.pp.reactor.U == 300.0
    assert np.array_equal(mismatched.design.x, nominal.design.x)


@pytest.mark.parametrize("path", ["column.P_base", "column.P_network"])
def test_a_column_pressure_cannot_be_a_mismatch(path):
    with pytest.raises(ValueError, match="column pressure"):
        pb.build(CASE, CASE.config(mismatch={path: 1.0}))


def test_the_economics_of_a_mismatch_are_those_of_its_design():
    economics = features_of(CASE)["economics"]
    assert economics(pb.build(CASE, CASE.config(mismatch=MISMATCH))) == economics(pb.build(CASE))
