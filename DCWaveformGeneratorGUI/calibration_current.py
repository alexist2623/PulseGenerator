"""DAC current and inactive DC-board settings shared by calibration paths."""
import math

DEFAULT_DAC_CURRENT_UA = 20000.0


def dac_current_ua(value=None):
    """Missing legacy metadata uses the nominal 20 mA calibration reference."""
    current = DEFAULT_DAC_CURRENT_UA if value is None else float(value)
    if not math.isfinite(current) or current <= 0:
        raise ValueError("DAC current must be positive and finite")
    return current


def channel_current_ua(records, channel):
    for record in (records or {}).values():
        if int(channel) in record.get("channels", ()):
            return dac_current_ua(record.get("current_ua"))
    return DEFAULT_DAC_CURRENT_UA


def read_channel_current(soc, channel):
    reader = getattr(soc, "get_dac_current_settings", None)
    records = dict(reader()) if callable(reader) else {}
    record = next((value for value in records.values()
                   if int(channel) in value.get("channels", ())), {})
    return {
        "dac_current_ua": dac_current_ua(record.get("current_ua")),
        "dac_current_source": "readback" if record.get("current_ua") is not None else "default_20mA",
    }


def normalize_dc_output(config):
    """DC_Out has neither RF attenuators nor an RF filter, including old JSON."""
    if config.output_board_type == "DC_Out":
        for name, value in (("output_att1_db", 0.0), ("output_att2_db", 0.0),
                            ("output_filter_type", "bypass"),
                            ("output_filter_cutoff_ghz", 2.5),
                            ("output_filter_bandwidth_ghz", 1.0)):
            object.__setattr__(config, name, value)
