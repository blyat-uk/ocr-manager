from core.detect import brightness, crop, flags


def test_compose_keeps_order_and_never_repeats_a_reason():
    assert flags.compose_flag(None, "a") == "a"
    assert flags.compose_flag("a", "b") == "a+b"
    assert flags.compose_flag("a+b", "a") == "a+b"


def test_only_informational_blocks_anything_not_listed():
    assert flags.only_informational(None, set())
    assert flags.only_informational("a+b", {"a", "b"})
    assert not flags.only_informational("a+c", {"a", "b"})


def test_is_cancelled_is_a_bool_and_tolerates_no_callable():
    assert flags.is_cancelled(None) is False
    assert flags.is_cancelled(lambda: 1) is True
    assert flags.is_cancelled(lambda: 0) is False


def test_both_detectors_use_the_shared_helpers():
    assert crop._compose_flag is brightness._compose_flag is flags.compose_flag
    assert crop._is_cancelled is brightness._is_cancelled is flags.is_cancelled
