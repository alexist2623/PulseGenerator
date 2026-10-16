"""Ramsey preset timing, parameter resolution, and saved-model regressions."""

from dataclasses import asdict, replace

import pytest

from dc_waveform_core import (
    QickRfAmplitudeParameterSpec,
    QickRfDurationParameterSpec,
    QickRfFrequencyParameterSpec,
    QickRfPhaseParameterSpec,
    QickRfPulseSpec,
    RF_PREDEFINED_COMPOSITE_TEMPLATES,
    build_predefined_composite_items,
)


def ramsey(**changes):
    options = dict(
        gen_ch=0, segment_name="set_0", delay_us=0.1, duration_us=1.0,
        frequency_mhz=50.0, gain=20000, att1_db=0.0, att2_db=0.0,
        pulse_mode="composite", segment_length_mode="extend_by_rf_duration",
        frequency_parameters=(QickRfFrequencyParameterSpec("f0", 50.0),),
        duration_parameters=(QickRfDurationParameterSpec("d0", 0.2),),
        predefined_template="ramsey", predefined_tau_us=1.0,
    )
    options.update(changes)
    return QickRfPulseSpec(**options)


def test_ramsey_body_is_one_full_tau_delay_without_a_refocusing_pi_pulse():
    assert "ramsey" in RF_PREDEFINED_COMPOSITE_TEMPLATES
    items = build_predefined_composite_items(
        "ramsey", n_pulses=99, tau_us=8.0,
        frequency_parameter="f0", duration_parameter="d0",
    )
    assert [(item.kind, item.name, item.duration_us) for item in items] == [
        ("delay", "Ramsey_delay", 8.0)]


@pytest.mark.parametrize("initial,final", [(False, False), (True, False), (False, True), (True, True)])
def test_ramsey_always_has_two_half_width_endpoints_and_no_n_sweep(initial, final):
    spec = ramsey(t2_initial_pi2=initial, t2_final_pi2=final,
                  predefined_n=99, predefined_n_sweep_enabled=True)
    assert spec.t2_initial_pi2 and spec.t2_final_pi2
    assert spec.predefined_n == 1
    assert not spec.predefined_n_sweep_enabled
    assert spec.software_sweep_axes == ()
    assert [item.kind for item in spec.effective_composite_items] == ["pulse", "delay", "pulse"]
    assert [event.name for event in spec.pulse_events] == ["T2_initial_pi2", "T2_final_pi2"]
    assert [event.duration_us for event in spec.pulse_events] == pytest.approx([0.1, 0.1])
    assert [event.delay_us for event in spec.pulse_events] == pytest.approx([0.1, 1.2])
    assert spec.segment_extension_duration_us == pytest.approx(1.3)


def test_ramsey_tau_sweep_changes_only_free_evolution_between_endpoints():
    spec = ramsey(predefined_tau_sweep_enabled=True,
                  predefined_tau_sweep_start_us=0.5,
                  predefined_tau_sweep_stop_us=2.5,
                  predefined_tau_sweep_count=3)
    axis, = spec.software_sweep_axes
    assert (axis.axis_kind, axis.output_name) == ("rf_template_tau", "rf_gen_0_ramsey_tau")
    for tau in axis.points:
        point = spec.with_predefined_parameters(tau_us=tau, n_pulses=7)
        first, last = point.pulse_events
        assert point.software_sweep_axes == ()
        assert point.predefined_n == 1
        assert first.duration_us == last.duration_us == pytest.approx(0.1)
        assert last.delay_us - (first.delay_us + first.duration_us) == pytest.approx(tau)
        assert point.segment_extension_duration_us == pytest.approx(0.3 + tau)


def test_ramsey_pi_width_parameter_sweep_halves_both_endpoints_and_keeps_frequency_axis():
    spec = ramsey(
        frequency_parameters=(QickRfFrequencyParameterSpec("f0", 50, True, 50, 60, 3),),
        duration_parameters=(QickRfDurationParameterSpec("d0", 0.2, True, 0.2, 0.6, 3),),
    )
    assert [(axis.axis_kind, axis.parameter_name) for axis in spec.sweep_axes] == [("rf_frequency", "f0")]
    axis, = spec.software_sweep_axes
    assert (axis.axis_kind, axis.parameter_name) == ("rf_t2_duration", "d0")
    for width in axis.points:
        point = spec.with_duration_parameters({"d0": width})
        first, last = point.pulse_events
        assert point.software_sweep_axes == ()
        assert point.sweep_axes == spec.sweep_axes
        assert first.duration_us == last.duration_us == pytest.approx(width / 2)
        assert last.delay_us == pytest.approx(0.1 + width / 2 + 1.0)
        assert point.segment_extension_duration_us == pytest.approx(1.1 + width)


@pytest.mark.parametrize("mode,value,unit", [("manual", 1234, "gain"), ("calibration", -19.5, "dBm")])
def test_ramsey_resolves_global_frequency_amplitude_and_independent_endpoint_phases(mode, value, unit):
    spec = ramsey(
        frequency_parameters=(QickRfFrequencyParameterSpec("shared_f", 75.0),),
        global_frequency_parameter="shared_f",
        global_amplitude_mode=mode, global_amplitude_parameter="a0",
        global_calibration_database_path="calibration.db", global_calibration_run_id=7,
        amplitude_parameters=(QickRfAmplitudeParameterSpec("a0", value, True, value, value + 2, 3),),
        phase_parameters=(QickRfPhaseParameterSpec("initial", 30),
                          QickRfPhaseParameterSpec("final", 90, True, 90, 270, 3)),
        t2_initial_phase_parameter="initial", t2_final_phase_parameter="final",
    )
    assert [(axis.axis_kind, axis.coordinate_unit) for axis in spec.software_sweep_axes] == [
        ("rf_amplitude", unit), ("rf_phase", "deg")]
    point = spec.with_amplitude_parameters({"a0": value + 2}).with_phase_parameters({"final": 270})
    assert point.software_sweep_axes == ()
    assert [event.frequency_mhz for event in point.pulse_events] == [75.0, 75.0]
    assert [event.phase_degrees for event in point.pulse_events] == [30, 270]
    assert [event.power_calibration_enabled for event in point.pulse_events] == [mode == "calibration"] * 2
    if mode == "calibration":
        assert [event.target_output_power_dbm for event in point.pulse_events] == [value + 2] * 2
        assert {event.power_calibration_database_path for event in point.pulse_events} == {"calibration.db"}
        assert {event.power_calibration_run_id for event in point.pulse_events} == {7}
    else:
        assert [event.gain for event in point.pulse_events] == [value + 2] * 2


def test_ramsey_keeps_selected_pi_reference_and_fine_duration_through_saved_models():
    spec = ramsey(
        duration_parameters=(QickRfDurationParameterSpec("d0", 0.2),
                             QickRfDurationParameterSpec("pi_reference", 0.08)),
        t2_duration_parameter="pi_reference", fine_duration=True,
        t2_initial_phase_degrees=-90, t2_final_phase_degrees=180,
    )
    assert [event.duration_us for event in spec.pulse_events] == pytest.approx([0.04, 0.04])
    assert [event.phase_degrees for event in spec.pulse_events] == [-90, 180]
    restored = QickRfPulseSpec(**asdict(spec))
    assert restored == spec
    assert restored.pulse_events == spec.pulse_events
    assert restored.fine_duration


def test_ramsey_to_custom_preserves_timeline_and_half_width_binding():
    original = ramsey(global_frequency_parameter="f0", global_gain=4321)
    custom = replace(original, predefined_template="custom")
    assert custom.composite_items == original.composite_items
    assert custom.pulse_events == original.pulse_events
    assert custom.t2_duration_parameter == "d0"
    changed = custom.with_duration_parameters({"d0": 0.6})
    assert [event.duration_us for event in changed.pulse_events] == pytest.approx([0.3, 0.3])
    assert [event.gain for event in changed.pulse_events] == [4321, 4321]
    assert QickRfPulseSpec(**asdict(custom)) == custom
    # Custom retains independent endpoint switches, including after Ramsey.
    assert len(replace(custom, t2_initial_pi2=False).pulse_events) == 1


def test_ramsey_rejects_missing_endpoint_references_before_execution():
    with pytest.raises(ValueError, match="unknown duration parameter"):
        ramsey(t2_duration_parameter="missing")
    with pytest.raises(ValueError, match="unknown frequency parameter"):
        ramsey(t2_frequency_parameter="missing")
    with pytest.raises(ValueError, match="require composite pulse mode"):
        ramsey(pulse_mode="single")


@pytest.mark.parametrize("template,body_pulses", [("cpmg", 4), ("udd", 4), ("hahn_echo", 1)])
def test_existing_presets_keep_optional_endpoints(template, body_pulses):
    spec = ramsey(predefined_template=template, t2_initial_pi2=False, t2_final_pi2=False)
    assert not spec.t2_initial_pi2 and not spec.t2_final_pi2
    assert len(spec.pulse_events) == body_pulses
