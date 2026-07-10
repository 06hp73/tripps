"""Mode selection: --modes / --flights / --no-freerider resolve to the right constraints."""

from __future__ import annotations

from argparse import Namespace

import pytest

from tripps.cli import _cli_constraints, _wants_flights, _wants_freerider
from tripps.models import TransportMode


def _args(**kw) -> Namespace:
    base = dict(modes=None, no_freerider=False, flights=False, max_transfers=None, max_hours=None)
    base.update(kw)
    return Namespace(**base)


def test_default_allows_the_ground_modes_and_freerider_but_not_flight():
    allowed = _cli_constraints(_args()).allowed_modes
    assert {TransportMode.TRAIN, TransportMode.BUS, TransportMode.FERRY, TransportMode.FREERIDER} <= allowed
    assert TransportMode.FLIGHT not in allowed
    assert TransportMode.WALK in allowed and TransportMode.LOCAL_TRANSIT in allowed


def test_modes_flag_selects_exactly_those_travel_modes():
    c = _cli_constraints(_args(modes="train,bus"))
    assert TransportMode.TRAIN in c.allowed_modes
    assert TransportMode.BUS in c.allowed_modes
    assert TransportMode.FERRY not in c.allowed_modes
    assert TransportMode.FREERIDER not in c.allowed_modes
    assert not c.include_freerider


def test_modes_flag_can_add_flight_and_freerider():
    c = _cli_constraints(_args(modes="train,freerider,flight"))
    assert TransportMode.FREERIDER in c.allowed_modes and c.include_freerider
    assert TransportMode.FLIGHT in c.allowed_modes
    assert TransportMode.BUS not in c.allowed_modes


def test_no_freerider_switch_without_modes():
    c = _cli_constraints(_args(no_freerider=True))
    assert TransportMode.FREERIDER not in c.allowed_modes
    assert not c.include_freerider


def test_unknown_mode_is_rejected():
    with pytest.raises(SystemExit, match="unknown mode"):
        _cli_constraints(_args(modes="train,teleport"))


def test_wants_flights_from_flag_or_modes():
    assert _wants_flights(_args(flights=True))
    assert _wants_flights(_args(modes="bus,flight"))
    assert not _wants_flights(_args(modes="train,bus"))
    assert not _wants_flights(_args())


def test_wants_freerider_honours_modes_over_default():
    assert _wants_freerider(_args())  # default on
    assert not _wants_freerider(_args(no_freerider=True))
    assert _wants_freerider(_args(modes="train,freerider"))
    assert not _wants_freerider(_args(modes="train,bus")), "modes without freerider turns it off"
