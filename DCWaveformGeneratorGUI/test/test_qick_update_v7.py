"""Offline route checks against the actual three-AWG v7 HWH; no board I/O."""
import hashlib
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from qick_update import _v5_hardware_review


@pytest.fixture
def hardware():
    path = Path(os.environ.get('QICK_V7_TEST_HWH',
        'C:/GIT_Workspace/QSTL_QICK/qick/firmware/projects/qstl_awg_v7/bitstream.hwh'))
    if not path.is_file():
        pytest.skip('Set QICK_V7_TEST_HWH to the actual three-AWG v7 HWH.')
    data = path.read_bytes()
    root = ET.fromstring(data)
    digest = hashlib.sha256(data).hexdigest()
    pairs = []
    for awg, rf, dac in ((4, 0, '10'), (5, 1, '11'), (6, 3, '12')):
        a, r = f'axis_awg_tuning_v7_{awg}', f'axis_signal_gen_v6_{rf}'
        pairs.append(dict(awg=a, rf=r, dac=dac, route=dict(
            output_sum_version=7, output_sum_block=f'axis_output_sum_{awg}',
            output_sum_role='awg', output_sum_bypass=False,
            output_sum_awg=a, output_sum_rf=r, output_sum_awg_delay_cycles=2,
            post_output_latency_cycles=2)))
    topology = dict(passed=True, sha256=digest, pairs=pairs)
    return root, topology, digest


def test_actual_three_awg_hwh(hardware):
    result = _v5_hardware_review(*hardware, version=7)
    assert result['passed'] and result['output_sum_blocks'] == 3
    assert result['output_sum_version'] == 7 and result['awg_delay_cycles'] == 2


@pytest.mark.parametrize('module,parameter,value', [
    ('axis_output_sum_4', 'AWG_DELAY_CYCLES', '6'),
    ('axis_output_sum_6', 'PIPELINE_CYCLES', '3'),
    ('axis_awg_tuning_v7_5', 'N_PTS', '8'),
    ('axis_awg_tuning_v7_4', 'RC_PRECOMP_VERSION', '0'),
    ('sum_one_valid', 'CONST_VAL', '0'),
])
def test_wrong_parameter_rejected(hardware, module, parameter, value):
    root, topology, digest = hardware
    root.find(f".//MODULE[@INSTANCE='{module}']/PARAMETERS/PARAMETER[@NAME='{parameter}']").set('VALUE', value)
    with pytest.raises(ValueError, match='v7 firmware hardware'):
        _v5_hardware_review(root, topology, digest, version=7)


def test_zero_dac_wiring_rejected(hardware):
    root, topology, digest = hardware
    root.find(".//MODULE[@INSTANCE='axis_register_slice_13']/PORTS/PORT[@NAME='m_axis_tdata']/CONNECTIONS/CONNECTION").set('PORT', 's10_axis_tdata')
    with pytest.raises(ValueError, match='unexpected connection'):
        _v5_hardware_review(root, topology, digest, version=7)


def test_wrong_manifest_latency_rejected(hardware):
    root, topology, digest = hardware
    topology['pairs'][0]['route']['output_sum_awg_delay_cycles'] = 6
    with pytest.raises(ValueError, match='differs from HWH'):
        _v5_hardware_review(root, topology, digest, version=7)
