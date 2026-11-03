"""Parameter-value navigation through the real preview, with no device I/O."""
import copy
import socket

import numpy as np
import pytest
from PyQt5 import QtCore, QtTest, QtWidgets

from sweep_point_selector import (ParameterChoice, SweepPointDialog, SweepValueSpinBox,
                                  SweepValuesModel, nearest_index)
from dc_waveform_core import QickSweepSpec, QickRampRateSweepSpec
from test_square_awg_exclusion import window
from test_sweep_grouping_gui import fixture_sweeps, check_and_group, edit_cell
from sweep_grouping import parameter_key


@pytest.fixture(autouse=True)
def no_device_io(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Preview selection must not contact hardware')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)


@pytest.fixture
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def type_value(spin, text):
    spin.lineEdit().setFocus()
    spin.lineEdit().selectAll()
    QtTest.QTest.keyClicks(spin.lineEdit(), text)
    QtTest.QTest.keyClick(spin.lineEdit(), QtCore.Qt.Key_Return)
    QtWidgets.QApplication.processEvents()


def dropdown_select(combo, index):
    combo.showPopup()
    view = combo.view()
    item = combo.model().index(index, 0)
    view.scrollTo(item)
    QtWidgets.QApplication.processEvents()
    QtTest.QTest.mouseClick(view.viewport(), QtCore.Qt.LeftButton, pos=view.visualRect(item).center())
    QtWidgets.QApplication.processEvents()


def open_selector(window):
    window.show()
    QtWidgets.QApplication.processEvents()
    QtTest.QTest.mouseClick(window._waveform_select_index, QtCore.Qt.LeftButton)
    dialog = window._waveform_sweep_dialog
    assert dialog.isVisible() and not dialog.isModal()
    return dialog


def row(dialog, kind, output=None, segment=None):
    return next(r for key, r in dialog.rows.items()
                if key[-3] == kind and (output is None or key[-2] == output)
                and (segment is None or key[-1] == segment))


@pytest.mark.parametrize('values,requested,current,expected', [
    ((0., 2., 5.), 3.2, 0, 1), ((5., 2., 0.), 3.2, 0, 1),
    ((1., 10., 100.), 60., 0, 2), ((0., 1.), -10., 1, 0),
    ((0., 1.), 100., 0, 1), ((1., 1., 1.), 1., 2, 2),
    ((0., 2.), 1., 1, 1), ((.208333333, .416666667), .45, 0, 1),
])
def test_nearest_actual_point(values, requested, current, expected):
    assert nearest_index(values, requested, current) == expected


@pytest.mark.parametrize('bad',(float('nan'),float('inf'),-float('inf')))
def test_nonfinite_input_is_rejected(bad):
    with pytest.raises(ValueError, match='finite'):
        nearest_index((0.,1.),bad)


@pytest.mark.parametrize('values',((10.,20.,50.),(50.,20.,10.)))
def test_numeric_entry_arrows_and_outside_range(app, values):
    spin = SweepValueSpinBox(values,'mV')
    spin.show()
    spy = QtTest.QSignalSpy(spin.indexChanged)
    type_value(spin,'2.1e1')
    assert values[spin.index]==20. and spin.lineEdit().text()=='20 mV'
    QtTest.QTest.keyClick(spin,QtCore.Qt.Key_Up)
    assert values[spin.index]==50.
    QtTest.QTest.keyClick(spin,QtCore.Qt.Key_Down)
    assert values[spin.index]==20.
    type_value(spin,'-999')
    assert values[spin.index]==10.
    assert len(spy)==4
    spin.close();spin.deleteLater();app.processEvents()


def test_group_members_share_one_dimension_and_preserve_others(app):
    d = SweepPointDialog()
    d.set_choices((ParameterChoice(('a',),'A','mV',(10.,20.,30.),0,'Group 1','g'),
                   ParameterChoice(('b',),'B','mV',(100.,75.,50.),0,'Group 1','g'),
                   ParameterChoice(('c',),'Hold','ns',(1.,2.,3.,4.),1)),(3,4))
    d.set_point(2);d.show()
    spy=QtTest.QSignalSpy(d.pointSelected)
    b=d.rows[('b',)][1]
    QtTest.QTest.keyClick(b,QtCore.Qt.Key_Down)
    assert d.point==6
    assert d.rows[('a',)][1].lineEdit().text()=='20 mV'
    assert d.rows[('c',)][2].currentIndex()==2
    dropdown_select(d.rows[('c',)][2],3)
    assert d.point==7 and len(spy)==2
    assert d.table.item(0,0).background()==d.table.item(1,0).background()
    d.close();d.deleteLater();app.processEvents()


def test_constant_member_preserves_group_coordinate_on_commit(app):
    d=SweepPointDialog()
    d.set_choices((ParameterChoice(('a',),'A','mV',(1.,1.,1.),0,'Group 1','g'),
                   ParameterChoice(('b',),'B','ns',(1.,2.,3.),0,'Group 1','g')),(3,))
    d.set_point(2);d.show()
    type_value(d.rows[('a',)][1],'1')
    assert d.point==2 and d.rows[('b',)][1].index==2
    dropdown_select(d.rows[('a',)][2],0)
    assert d.point==0 and d.rows[('b',)][1].index==0
    d.close();d.deleteLater();app.processEvents()


def test_popup_has_lazy_rows_and_no_cartesian_items(app):
    values=tuple(float(i) for i in range(100_000))
    d=SweepPointDialog()
    d.set_choices((ParameterChoice(('a',),'A','mV',values,0),
                   ParameterChoice(('b',),'B','mV',values,1)),(100_000,100_000))
    model=d.rows[('a',)][2].model()
    assert isinstance(model,SweepValuesModel) and model.rowCount()==100_000
    assert model.data(model.index(99999,0))=='99999: 99999 mV'
    # An inaccessible flat point must not silently clamp unrelated dimensions.
    d.rows[('a',)][2].setCurrentIndex(99999)
    assert d.point==0 and 'exceeds' in d.status.text()
    d.deleteLater();app.processEvents()


def test_main_plot_value_selection_does_not_edit_experiment(window):
    fixture_sweeps(window, (3,3,4))
    d=open_selector(window)
    before=copy.deepcopy(window._settings_to_dict())
    def plateau():
        seq=window._waveform_preview_sequence
        _,_,bounds=seq.waveform_vertices(window._waveform_preview_hardware_index)
        _,start,end=next(b for b in bounds if b[0]=='set_1')
        time=(start+end)/2*1000/window._qick_fabric_mhz
        return np.interp(time,window._plot._line[0].xData,window._plot._line[0].yData)
    choice,spin,combo=row(d,'amplitude','awg_0')
    type_value(spin,'119')
    assert d.isVisible() and spin.index==2
    assert window._waveform_point_index.value()==6
    assert plateau()==pytest.approx(160.)
    dropdown_select(combo,1)
    assert window._waveform_point_index.value()==3
    assert plateau()==pytest.approx(40.)
    window._waveform_point_index.setValue(35)
    assert d.point==35 and combo.currentIndex()==2
    assert row(d,'amplitude','awg_0')[1] is spin  # changing points reuses rows/models
    before['display']['waveform_point_index']=35
    assert window._settings_to_dict()==before


def test_group_selection_count_change_remove_and_readd(window):
    specs=fixture_sweeps(window,(3,3,4))
    check_and_group(window,specs[:2])
    d=open_selector(window)
    choice,spin,combo=row(d,'amplitude','awg_0')
    type_value(spin,'150')
    assert row(d,'amplitude','awg_1')[1].index==2
    assert row(d,'ramp_duration')[1].index==0
    first=window._sweep_grouping.member_id(specs[0])
    window._ungroup_sweep_parameter(first)
    _,a,_=row(d,'amplitude','awg_0')
    _,b,_=row(d,'amplitude','awg_1')
    before=b.index
    a.set_index(1,notify=True)
    assert b.index==before
    check_and_group(window,window._sweep_specs[:2])
    edit_cell(window,parameter_key(specs[0]),4,5)
    assert len(row(d,'amplitude','awg_0')[0].values)==5
    assert len(row(d,'amplitude','awg_1')[0].values)==5
    old_generation=d._generation
    old_key=row(d,'amplitude','awg_1')[0].key
    # Deletion from AWG Outputs must remove the popup row and group binding.
    window._delete_port(1)
    assert not any(c.key[-2]=='awg_1' for c in d.choices)
    previous=d.point
    d._select(old_key,4,old_generation)
    assert d.point==previous
    window._add_port()
    window._sweep_specs.append(QickSweepSpec('set_1','awg_1',.2,-.3,3))
    window._notify_sweep_state_changed()
    assert not row(d,'amplitude','awg_1')[0].group
    saved_index=row(d,'amplitude','awg_0')[1].index
    row(d,'amplitude','awg_1')[2].setCurrentIndex(2)
    assert row(d,'amplitude','awg_0')[1].index==saved_index


def test_fine_grid_and_current_index_survive_json_reload(window,tmp_path):
    window._add_segment(10.,1000.,100.)
    window._pulse[0].set_ramp_duration_ns(1,1/4.8)
    window._sweep_specs=[QickRampRateSweepSpec('ramp_0_to_1',1/4800,30/4800,30)]
    window._notify_sweep_state_changed()
    d=open_selector(window)
    choice,spin,combo=row(d,'ramp_duration')
    assert choice.unit=='ns' and choice.values[1]==pytest.approx(2/4.8)
    type_value(spin,'.45')
    assert spin.index==1 and window._waveform_point_index.value()==1
    path=tmp_path/'selection.json'
    window._save_settings_json(path)
    window._waveform_point_index.setValue(29)
    window._load_settings_json(path)
    assert window._waveform_point_index.value()==1
    assert row(d,'ramp_duration')[1].index==1
    assert d.isVisible()


def test_host_and_hardware_parameters_use_the_same_preview_point(window):
    from test_rf_t2_integration import load_composite
    load_composite(window,pi_sweep=True)
    window._sweep_specs=[QickSweepSpec('set_0','awg_0',0.,.1,3)]
    window._notify_sweep_state_changed()
    d=open_selector(window)
    host=row(d,'rf_delay')
    host[2].setCurrentIndex(2)
    row(d,'amplitude','awg_0')[2].setCurrentIndex(1)
    assert host[1].index==2 and window._waveform_preview_hardware_index==1
    concrete=window._waveform_preview_rf_specs[0]
    assert concrete.delay_parameters[0].duration_us==pytest.approx(3.)
    assert window._plot._line[0].yData[0]==pytest.approx(40.)
    previous=row(d,'amplitude','awg_0')[1].index
    row(d,'rf_t2_duration')[2].setCurrentIndex(1)
    assert row(d,'amplitude','awg_0')[1].index==previous
    assert window._waveform_preview_rf_specs[0].pulse_events[0].duration_us==pytest.approx(.4)


def test_no_sweeps_disable_button_and_empty_open_dialog(window):
    assert not window._waveform_select_index.isEnabled()
    fixture_sweeps(window)
    d=open_selector(window)
    window._clear_sweep()
    assert not window._waveform_select_index.isEnabled()
    assert not d.rows and 'No sweep' in d.status.text()
    d.close()
    fixture_sweeps(window)
    assert open_selector(window) is d
    window.close()
    assert not d.isVisible()
