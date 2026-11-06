"""Review actual generated v8 HWH without opening a window or contacting a board."""
import hashlib
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest
from qick_update import _v5_hardware_review


@pytest.fixture
def hardware():
    path = Path(os.environ.get('QICK_V8_TEST_HWH',
        'C:/GIT_Workspace/QSTL_QICK/qick/firmware/projects/qstl_awg_v8/bitstream.hwh'))
    if not path.is_file():
        pytest.skip('Set QICK_V8_TEST_HWH to the generated v8 HWH.')
    data = path.read_bytes()
    root = ET.fromstring(data)
    digest = hashlib.sha256(data).hexdigest()
    pairs = []
    for awg, rf, dac in ((4, 0, '10'), (5, 1, '11'), (6, 3, '12')):
        a, r = f'axis_awg_tuning_v8_{awg}', f'axis_signal_gen_v6_{rf}'
        pairs.append(dict(awg=a, rf=r, dac=dac, route=dict(
            output_sum_version=7, output_sum_block=f'axis_output_sum_{awg}',
            output_sum_role='awg', output_sum_bypass=False,
            output_sum_awg=a, output_sum_rf=r, output_sum_awg_delay_cycles=2,
            post_output_latency_cycles=2)))
    return root, dict(passed=True, sha256=digest, pairs=pairs), digest


def test_actual_v8_hwh_and_new_register_address(hardware):
    root, topology, digest = hardware
    result = _v5_hardware_review(root, topology, digest, version=8)
    assert result['passed'] and result['output_sum_blocks'] == 3
    capture = next(m for m in root.iter('MODULE') if m.get('MODTYPE') == 'axis_buffer_ddr_sample_v4')
    ranges = root.findall(f".//MEMRANGE[@INSTANCE='{capture.get('INSTANCE')}']")
    assert ranges
    assert all(int(r.get('BASEVALUE'), 16) == 0xA00A0000 for r in ranges)
    assert all(int(r.get('HIGHVALUE'), 16) >= 0xA00A003B for r in ranges)


@pytest.mark.parametrize('module,parameter,value', [
    ('axis_output_sum_4', 'AWG_DELAY_CYCLES', '6'),
    ('axis_awg_tuning_v8_5', 'RC_PRECOMP_VERSION', '0'),
    ('ddr4_axis_buffer_ddr_sample_v4_0', 'IQ_COMPONENT_BITS', '32'),
    ('ddr4_axis_buffer_ddr_sample_v4_0', 'FORMAT_VERSION', '0'),
])
def test_incompatible_parameters_rejected(hardware, module, parameter, value):
    root, topology, digest = hardware
    root.find(f".//MODULE[@INSTANCE='{module}']/PARAMETERS/PARAMETER[@NAME='{parameter}']").set('VALUE', value)
    with pytest.raises(ValueError, match='v8 firmware hardware'):
        _v5_hardware_review(root, topology, digest, version=8)


def test_old_capture_ip_rejected(hardware):
    root, topology, digest = hardware
    capture = next(m for m in root.iter('MODULE') if m.get('MODTYPE') == 'axis_buffer_ddr_sample_v4')
    capture.set('MODTYPE', 'axis_buffer_ddr_sample_v3')
    with pytest.raises(ValueError, match='abort-capable'):
        _v5_hardware_review(root, topology, digest, version=8)


def test_v8_cannot_be_reviewed_as_v7(hardware):
    with pytest.raises(ValueError, match='v7 firmware hardware'):
        _v5_hardware_review(*hardware, version=7)
