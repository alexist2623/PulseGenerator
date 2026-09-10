# FIR DDR integer formats

The GUI uses the loaded QICK configuration to select sample width, scaling,
capture delay and memory accounting. Deploy the updated QICK Python library on
the board/server as well as the updated GUI on the client.

| Firmware | Raw IQ dtype | Bytes/IQ | Automatic delay |
| --- | --- | ---: | --- |
| Legacy 1 MSPS | signed int16 | 4 | Existing software compensation |
| 50 kSPS DDR V2 | signed int16 | 4 | Existing FPGA delay, HWH units |
| Continuous 1 MSPS DDR V2 | signed int16 | 4 | FPGA source-clock delay |
| 1 MSPS DDR V3 + FIR V2 | signed int64 | 16 | 8712 / 300 MHz = 29.04 us |

The V3 raw value is an integer FIR accumulation scaled by 2^46. DDR and raw
Python arrays contain signed integers only. The scale is metadata describing
how to compare those integers with the old input-code units. Display,
calibration, noise and mean calculations use a floating copy multiplied by
2^-46. Raw arrays are never replaced with this copy.

`FineTuneDdrResult.iq`, `NoiseAcquisitionResult.iq` and
`SParameterSweepResult.iq_traces` retain raw integer values. Their
`iq_scale_log2` metadata distinguishes the formats. Fine-tune mean properties
and S-parameter response properties already use input-code units. The noise
GUI keeps the raw capture in `NoiseTraceCollection.raw_iq` alongside the
normalized I trace used for analysis.

QCoDeS AWG/experiment runs save display/calibrated traces in the existing
`i_trace`/`q_trace` parameters and exact V3 arrays in `i_raw_int64`/`q_raw_int64`.
The acquisition metadata records `raw_iq_shape` and `raw_iq_scale_log2`.
`load_qick_raw_int64_arrays(dataset, shape=raw_iq_shape)` restores all original
integer values, including when the selected analysis policy averages shots.
S-parameter databases retain integer I/Q traces and store `iq_scale_log2` with
the result. `load_sparameter_run()` restores that scale before recalculating
the response.

Two IQ64 samples occupy one 256-bit DDR word. Odd sample counts are padded per
trigger: 13 samples occupy 224 bytes, of which 208 are valid IQ. API capture
addresses and strides are bytes; API readback `start` remains a physical
32-bit-word offset. The GUI converts the byte address accordingly.

V3 format registers and HWH must agree before the QICK driver reads or arms
capture. Unsupported or inconsistent formats raise an error. A GUI connected
to an old QICK driver that reports a wide stream without the required format
metadata also raises an error rather than interpreting it as int16.

The three FIR intermediate outputs preserve all 34/51/69 bits. Only the final
69-to-64 conversion rounds five low bits. It is not mathematically lossless
at that final boundary; the proven error is at most 2^-47 input codes.
