# Acquisition anchors and Stability electrodes

Acquisition displays the user-defined segment name. The stored anchor remains
the stable `set_N` identifier, so renaming a segment does not move its trigger.
Renaming a row refreshes Acquisition and RF pulse selectors immediately.

The anchor is the start of the selected SET/flat portion, followed by the
configured trigger delay. It is not a boundary that clips a measurement.
For N stored samples and FIR output rate Fs, the logical measurement duration
is N/Fs. At 1 MSPS, 64 samples cover 64 us; at 50 kSPS they cover 1280 us.

If capture overlaps a following segment, AWG commands retain their original
times and acquisition includes that following waveform. If capture extends
past the entire user waveform, the point-end barrier extends to the capture
end (including the configured input-sample margin) before the next repetition.
FPGA FIR trigger-delay compensation is a pipeline delay, not an extra delay
inserted between every point. Legacy firmware with software warmup uses its
existing warmup scheduling instead.

The output after the last user segment depends on compensation:

- With DC and RC compensation disabled, the final AWG SET level remains until
  another command changes it. Ending acquisition does not itself zero the DAC.
- With DC compensation enabled, the user waveform is set to zero at its own
  end. The DC area pulse runs after the capture window has completed.
- With RC compensation enabled, the nominal input to the RC precompensator is
  set to zero at the user-waveform end. The physical DAC can still have an
  offset from the retained integral. The integral is reset at the repetition
  epilogue, after capture and any DC area pulse.

To acquire only a chosen flat segment, its remaining duration must cover the
trigger delay and requested measurement window.

Stability's X and Y selectors include AWG Tuning generators discovered in the
identified firmware, even when those generators are absent from AWG Tuning's
output list. Generator identities are saved with each axis and survive an
offline settings round trip. Selecting an electrode does not add or remap an
AWG Tuning panel. Existing virtual-gate cross-capacitance entries are retained;
additional independently selected DACs use identity coupling. SquarePulse and
RF DDS generators are not offered as newly discovered AWG electrodes.

Regression coverage: `test_acquisition_electrode_selection.py` checks GUI labels,
physical channel routing, offline settings, matrix preservation and exclusions.
`test_acquisition_segment_timing.py` executes tProcessor instructions for
spillover into the next segment and beyond the final segment, with all four
DC/RC enable combinations and two repetitions.
