"""Calibrated DAC levels, infinite edge timing, and GUI start/stop lifecycle."""
import os
import sqlite3
from math import log10
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace
from threading import Event, Thread
from types import SimpleNamespace

import pytest
from PyQt5 import QtCore, QtTest, QtWidgets
from qick.sim.tproc_v1 import TProcV1Sim
from qick.qick_asm import QickConfig

import qick_square_wave as square


def soccfg():
    return QickConfig(dict(sw_version="0.2.357", refclk_freq=300.0,
        tprocs=[dict(type="axis_tproc64x32_x8", f_time=300.0, pmem_size=65536, dmem_size=4096)],
        gens=[dict(type="axis_awg_tuning_v1", gen_type="awg_tuning", tproc_ch=0,
            tmux_ch=0, f_fabric=300.0, n_pts=16, samps_per_clk=16, frac=16,
            cmd_width=160, step_width=24, duration_width=23, fixed_width=48,
            dac_invalid_lsb=2, maxv=32764, minv=-32768, has_mixer=False,
            has_dds=False, b_dds=32, b_phase=32, dac="00", interpolation=1)],
        readouts=[]))


def calibration_database(path, *, slope=80.0):
    # Use the same gain/freq/pwr tables as the existing calibration notebooks.
    with sqlite3.connect(path) as conn:
        conn.executescript("""
            CREATE TABLE experiments (exp_id INTEGER PRIMARY KEY, sample_name TEXT);
            CREATE TABLE runs (run_id INTEGER PRIMARY KEY, exp_id INTEGER,
                              result_table_name TEXT, is_completed INTEGER);
        """)
        for run_id, board, gain_scale, frequencies in (
                (1, "DC_Out_legacy", slope, (.001, 1.0)),
                (2, "DC_Out_higher_response", slope/2, (.001, 1.0)),
                (3, "RF_Out", 20, (.001, 1.0)),
                (4, "DC_Out_high_frequency", 1, (400, 500))):
            conn.execute("INSERT INTO experiments VALUES (?, ?)", (run_id, board))
            conn.execute("INSERT INTO runs VALUES (?, ?, ?, 1)", (run_id, run_id, f"data_{run_id}"))
            conn.execute(f"CREATE TABLE data_{run_id} (gain REAL, freq REAL, pwr REAL)")
            for frequency in frequencies:
                for gain in (1000, 4000, 8000, 16000, 32764):
                    peak_v = gain/gain_scale/1000
                    power_dbm = 10*log10(peak_v**2/(2*50)/.001)
                    conn.execute(f"INSERT INTO data_{run_id} VALUES (?, ?, ?)", (gain, frequency, power_dbm))
    return path


@pytest.fixture
def calibrated_config(tmp_path):
    path = calibration_database(tmp_path / "gain_pwr_calb.db")
    return square.SquareWaveConfig(gen_ch=0, calibration_database_path=str(path), calibration_run_id=1)


def test_measured_calibration_and_integer_quantization(calibrated_config, tmp_path):
    cfg = soccfg()
    config = calibrated_config
    resolved = square.resolve_square_wave_calibration(config)
    assert resolved["codes_per_mv"] == pytest.approx(80)
    assert config.codes(cfg["gens"][0], resolved) == (1920, 320)
    # A fractional multiplication is rounded to the actual DAC quantum.
    path = calibration_database(tmp_path / "fractional.db", slope=80.1234)
    config = replace(config, calibration_database_path=str(path), offset_mv=0.17, zero_code=1121.3)
    high, low = config.codes(cfg["gens"][0], square.resolve_square_wave_calibration(config))
    for code, voltage in zip((high, low), (10.17, -9.83)):
        assert type(code) is int and code % 4 == 0
        assert abs(code - (voltage * 80.1234 + 1121.3)) <= 2


def test_database_selection_is_dc_output_only_and_never_extrapolates(calibrated_config):
    config = replace(calibrated_config, calibration_run_id=0)
    result = square.resolve_square_wave_calibration(config)
    assert result["run_id"] == 2
    assert result["codes_per_mv"] == pytest.approx(40)
    for bad in (replace(config, calibration_run_id=3),
                replace(config, calibration_run_id=4),
                replace(config, frequency_hz=10e6)):
        with pytest.raises(LookupError):
            square.resolve_square_wave_calibration(bad)
    with pytest.raises(ValueError, match="Select a DAC calibration database"):
        square.resolve_square_wave_calibration(square.SquareWaveConfig())


def test_program_reloads_measured_gain_and_reference_impedance(calibrated_config):
    program = square.build_square_wave_program(soccfg(), replace(calibrated_config, calibration_run_id=2))
    assert program.summary["high_code"] == 1520
    assert program.summary["low_code"] == 720
    assert program.summary["calibration"]["run_id"] == 2
    result = square.resolve_square_wave_calibration(replace(calibrated_config, calibration_reference_ohm=200))
    assert result["codes_per_mv"] == pytest.approx(40)
    # A legacy manual slope cannot override measured calibration after reload.
    values = square.normalize_square_wave_settings({"codes_per_mv": 123})
    assert "codes_per_mv" not in values


@pytest.mark.parametrize("kwargs", [dict(amplitude_mv=1000), dict(amplitude_mv=1e-8)])
def test_out_of_range_and_unresolvable_levels_are_rejected(kwargs, calibrated_config):
    with pytest.raises(ValueError):
        square.build_square_wave_program(soccfg(), replace(calibrated_config, **kwargs))


@pytest.mark.parametrize("kwargs", [dict(frequency_hz=0), dict(amplitude_mv=-1),
    dict(duty_percent=100), dict(calibration_reference_ohm=0), dict(zero_code=float("nan")),
    dict(gen_ch=True), dict(calibration_run_id=1.5), dict(calibration_database_path=None)])
def test_invalid_inputs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        square.SquareWaveConfig(**kwargs)


class ReferenceTimeInterpreter(TProcV1Sim):
    def _exec_waiti(self, port, imm):
        # The stock event interpreter conflates wall time and t_ref on WAIT.
        # RTL WAIT stalls execution but leaves t_ref unchanged; only SYNC
        # advances the reference used for the following SET timestamps.
        self.pc += 1

    def _exec_wait(self, page, port, reg):
        self.pc += 1


@pytest.mark.parametrize("frequency,duty", [(40000, 50), (12345, 37), (500000, 95)])
def test_compiled_infinite_loop_preserves_edges_without_readout(frequency, duty, calibrated_config):
    config = replace(calibrated_config, frequency_hz=frequency, duty_percent=duty)
    program = square.build_square_wave_program(soccfg(), config)
    model = ReferenceTimeInterpreter()
    with pytest.raises(RuntimeError, match="exceeded max_steps"):
        model.run(program, max_steps=20000)
    events = model.output_events
    assert len(events) > 100 and not model.ended
    assert not program.ro_chs
    period = round(300e6/frequency)
    high_ticks = round(period*duty/100)
    codes = [e.word & 0xffffffff for e in events]
    assert codes == [1920 if i % 2 == 0 else 320 for i in range(len(events))]
    for i in range(len(events)-1):
        assert events[i+1].cycle-events[i].cycle == (high_ticks if i % 2 == 0 else period-high_ticks)
    assert all(e.tproc_ch == 0 for e in events)
    waits = [i for i in program.prog_list if i["name"] == "waiti"]
    assert waits[0]["args"][0] != 0
    assert program.summary["actual_frequency_hz"] == 300e6/period
    assert program.binprog


def test_clock_override_and_firmware_validation(calibrated_config):
    cfg = soccfg()
    cfg["tprocs"][0]["f_time"] = 400.0
    program = square.build_square_wave_program(cfg, calibrated_config, tproc_mhz=300)
    assert program.summary["period_cycles"] == 7500
    assert cfg["tprocs"][0]["f_time"] == 400.0
    for config in (square.SquareWaveConfig(gen_ch=2),
                   square.SquareWaveConfig(gen_ch=0, frequency_hz=1e7),
                   square.SquareWaveConfig(gen_ch=0, frequency_hz=.001),
                   square.SquareWaveConfig(gen_ch=0, duty_percent=.001)):
        with pytest.raises(ValueError):
            square.build_square_wave_program(cfg, config)
    cfg["gens"][0]["type"] = "axis_signal_gen_v6"
    with pytest.raises(ValueError, match="axis_awg_tuning_v1"):
        square.build_square_wave_program(cfg, square.SquareWaveConfig(gen_ch=0))


class FakeSoc:
    def __init__(self):
        self.calls = []
        self.stop_failures = 0
        self.start_error = False

    def rfb_set_gen_dc(self, ch):
        self.calls.append(("dc", ch))

    def start_src(self, source):
        self.calls.append(("source", source))

    def start_tproc(self):
        self.calls.append(("start",))
        if self.start_error:
            raise RuntimeError("start transport error")

    def stop_tproc(self):
        self.calls.append(("stop",))
        if self.stop_failures:
            self.stop_failures -= 1
            raise RuntimeError("stop transport error")


def fake_factory(cfg, config, **kwargs):
    return SimpleNamespace(summary=dict(actual_frequency_hz=40000, actual_duty_percent=50,
        high_code=1920, low_code=320, tproc_mhz=300),
        config_all=lambda soc, reset: soc.calls.append(("load", reset)))


def worker_for(soc):
    return square.SquareWaveWorker(None, square.SquareWaveConfig(gen_ch=0), tproc_mhz=300,
        connector=lambda connection: (soc, soccfg()), program_factory=fake_factory)


def test_worker_runs_until_stop_and_retries_failed_stop():
    soc = FakeSoc()
    soc.stop_failures = 1
    worker = worker_for(soc)
    started, stop_failed, finished = Event(), Event(), Event()
    worker.started.connect(lambda _: started.set(), QtCore.Qt.DirectConnection)
    worker.stop_failed.connect(lambda _: stop_failed.set(), QtCore.Qt.DirectConnection)
    worker.finished.connect(lambda _: finished.set(), QtCore.Qt.DirectConnection)
    thread = Thread(target=worker.run, daemon=True)
    thread.start()
    try:
        assert started.wait(3)
        assert thread.is_alive() and not finished.is_set()
        worker.request_stop()
        assert stop_failed.wait(3)
        assert thread.is_alive() and not finished.is_set()
        worker.request_stop()
        assert finished.wait(3)
        thread.join(3)
        assert soc.calls == [("load", True), ("dc", 0), ("source", "internal"), ("start",), ("stop",), ("stop",)]
    finally:
        worker.request_stop()
        thread.join(3)


def test_cancel_before_start_and_cleanup_after_start_error():
    soc = FakeSoc()
    worker = worker_for(soc)
    worker.request_stop()
    worker.run()
    assert soc.calls == []
    soc.start_error = True
    worker = worker_for(soc)
    errors = []
    worker.failed.connect(errors.append, QtCore.Qt.DirectConnection)
    worker.run()
    assert soc.calls[-1] == ("stop",)
    assert "start transport error" in errors[0]


def test_cancel_during_program_load_does_not_start_output():
    soc = FakeSoc()
    worker = worker_for(soc)
    worker.program_factory = lambda *a, **k: SimpleNamespace(
        config_all=lambda *a, **k: worker.request_stop())
    worker.run()
    assert ("start",) not in soc.calls
    assert soc.calls[-1] == ("stop",)


def qt_until(predicate):
    for _ in range(300):
        if predicate():
            return
        QtTest.QTest.qWait(10)
    assert predicate()


def test_gui_tab_settings_start_stop_and_close(monkeypatch, calibrated_config):
    import DCWaveform_Generator as gui
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    soc = FakeSoc()
    monkeypatch.setattr(square, "connect_qick", lambda _: (soc, soccfg()))
    monkeypatch.setattr(square, "build_square_wave_program", fake_factory)
    window = gui.MainWindow()
    panel = window._square_wave_panel
    try:
        assert window._control_tabs.tabText(window._control_tabs.indexOf(panel)) == "QICK Square Wave"
        assert not hasattr(panel, "codes_per_mv")
        window._calibration_panel.database_path.setText(calibrated_config.calibration_database_path)
        assert panel.calibration_database_path.text() == ""
        panel.refresh_calibration.click()
        assert "Loaded DC_Out Run 2" in panel.status.text()
        assert panel.calibration_run.findData(3) == -1
        panel.calibration_run.setCurrentIndex(panel.calibration_run.findData(1))
        assert "Loaded DC_Out Run 1" in panel.status.text()
        panel.amplitude_mv.setValue(12.5)
        panel.zero_code.setValue(1104)
        payload = window._settings_to_dict()
        decoded = window._decode_settings(payload)
        assert decoded["square_wave"]["amplitude_mv"] == 12.5
        assert decoded["square_wave"]["zero_code"] == 1104
        assert decoded["square_wave"]["calibration_run_id"] == 1
        panel.load_settings(decoded["square_wave"])
        del payload["square_wave"]
        assert window._decode_settings(payload)["square_wave"] == square.normalize_square_wave_settings()
        panel.start_button.click()
        qt_until(lambda: "Running:" in panel.status.text())
        assert not panel.start_button.isEnabled() and panel.stop_button.isEnabled()
        assert window._experiment_thread.isRunning()
        running_worker = window._experiment_worker
        running_thread = window._experiment_thread
        blocked = []
        monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *args: blocked.append(args[2]))
        window._configure_qick_setup()
        window._identify_qick_configuration()
        window._start_bias_hardware_operation("read", {})
        window._run_sparameter_sweep()
        window._run_noise_acquisition(None)
        window._run_power_calibration("output")
        window._run_qick_experiment()
        assert len(blocked) == 7
        assert window._experiment_worker is running_worker
        assert window._experiment_thread is running_thread
        assert soc.calls.count(("start",)) == 1
        panel.stop_button.click()
        qt_until(lambda: window._experiment_thread is None)
        assert panel.start_button.isEnabled() and not panel.stop_button.isEnabled()
        assert soc.calls[-1] == ("stop",)
        panel.start_button.click()
        qt_until(lambda: "Running:" in panel.status.text())
        window.close()
        qt_until(lambda: window._experiment_thread is None)
        assert soc.calls[-1] == ("stop",)
    finally:
        if window._experiment_worker is not None:
            window._experiment_worker.request_stop()
            qt_until(lambda: window._experiment_thread is None)
        window.close()
        app.processEvents()


def test_invalid_database_prevents_hardware_configuration():
    soc = FakeSoc()
    worker = worker_for(soc)
    worker.program_factory = square.build_square_wave_program
    errors = []
    worker.failed.connect(errors.append, QtCore.Qt.DirectConnection)
    worker.run()
    assert soc.calls == []
    assert "Select a DAC calibration database" in errors[0]
