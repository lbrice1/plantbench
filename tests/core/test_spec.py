"""Specifications expand to a fixed, checked list of configurations."""

from __future__ import annotations

import pytest

from plantbench.core.spec import Spec, _split, expand, set_path

SPEC = {
    "study": {"name": "t", "case": "jacketed_cstr", "seed": 3},
    "base": {"structure": "temperature PI"},
    "variants": [{"options": {"branch": "middle"}}, {"structure": "composition cascade"}],
    "sweep": {"disturbance": [{"name": "feed_temperature", "change": 1.0},
                              {"name": "feed_temperature", "change": -1.0}]},
    "sample": {"n": 4, "tuning.reactor T.Kc": [-2.0, -0.5]},
    "run": {"t_end": 50.0, "dt": 5.0},
}


@pytest.mark.parametrize("path, keys", [
    ("tuning.reactor T.Kc", ["tuning", "reactor T", "Kc"]),
    ("params.reactor.U", ["params", "reactor.U"]),
    ("mismatch.reactor.U", ["mismatch", "reactor.U"]),
    ("x0.T", ["x0", "T"]),
    ("hold.Fj", ["hold", "Fj"]),
    ("noise.seed", ["noise", "seed"]),
    ("noise.std.tray 17 T", ["noise", "std", "tray 17 T"]),
    ("instruments.reactor C_A.deadtime", ["instruments", "reactor C_A", "deadtime"]),
    ("disturbance", ["disturbance"]),
    ("options.network", ["options", "network"]),
])
def test_paths_address_sections_then_keys(path, keys):
    assert _split(path) == keys


@pytest.mark.parametrize("path", ["nope.x", "structure.x"])
def test_bad_paths_are_rejected(path):
    with pytest.raises(KeyError):
        _split(path)


def test_set_path_builds_nested_sections():
    c = {"structure": "s"}
    set_path(c, "tuning.reactor T.Kc", -1.0)
    set_path(c, "params.reactor.U", 300.0)
    assert c == {"structure": "s", "tuning": {"reactor T": {"Kc": -1.0}},
                 "params": {"reactor.U": 300.0}}


def test_expansion_is_variants_by_sweep_by_sample_and_deterministic():
    configs = expand(Spec.from_dict(SPEC))
    assert len(configs) == 2 * 2 * 4
    assert configs == expand(Spec.from_dict(SPEC))
    kcs = [c["tuning"]["reactor T"]["Kc"] for c in configs[:4]]
    assert all(-2.0 <= k <= -0.5 for k in kcs) and len(set(kcs)) == 4
    # A Latin hypercube puts one point in each quarter of the range.
    assert sorted(int((k + 2.0) / 0.375) for k in kcs) == [0, 1, 2, 3]
    assert configs[8]["structure"] == "composition cascade"


def test_variant_groups_combine_as_a_product():
    spec = {**SPEC, "variants": {
        "plant": [{"options": {"branch": "low"}}, {"options": {"branch": "high"}}],
        "loop": [{"structure": "open loop"}, {"structure": "temperature PI"},
                 {"structure": "composition cascade"}]},
        "sweep": {}, "sample": {}}
    configs = expand(Spec.from_dict(spec))
    assert len(configs) == 6
    assert configs[0] == {"structure": "open loop", "options": {"branch": "low"}}
    assert configs[5] == {"structure": "composition cascade", "options": {"branch": "high"}}


def test_a_different_seed_draws_different_points():
    other = {**SPEC, "study": {**SPEC["study"], "seed": 4}}
    assert expand(Spec.from_dict(SPEC)) != expand(Spec.from_dict(other))


@pytest.mark.parametrize("change", [
    {"extra": {}},
    {"study": {"name": "t"}},
    {"run": {"t_end": 1.0, "nope": 1}},
    {"run": {"features": [7]}},
    {"sample": {"tuning.reactor T.Kc": [0.0, 1.0]}},
    {"sample": {"n": 2, "tuning.reactor T.Kc": [1.0, 0.0]}},
    {"sweep": {"nope.x": [1]}},
])
def test_malformed_specifications_are_refused(change):
    with pytest.raises((KeyError, ValueError, TypeError)):
        Spec.from_dict({**SPEC, **change})


def test_a_feature_name_is_checked_against_the_case_not_the_specification():
    """`core` cannot know a case's own features, so the names are checked in `datagen`."""
    spec = Spec.from_dict({**SPEC, "run": {"features": ["nope"]}})  # accepted here
    assert spec.run["features"] == ["nope"]
