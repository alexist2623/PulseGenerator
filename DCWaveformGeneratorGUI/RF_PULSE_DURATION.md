# RF pulse duration and hardware sweeps

RF pulses in AWG Tuning and the shared experiment compiler select their output
mode from the generator fabric-clock length:

- A fixed pulse of 3 through 65,535 clocks uses a one-shot command with that
  length in the generator mode register.
- A duration sweep uses one-shot commands if every executed point fits that
  range. A compact DMEM table supplies the complete mode/length word at each
  hardware-loop point. Repetitions retain the same length; Cartesian axis
  transitions reset the table pointer.
- If any duration coordinate exceeds 65,535 clocks, the entire duration axis
  uses the existing periodic start plus timed zero-gain stop. This includes
  descending sweeps. With a count of one, only the start coordinate executes.
- Fixed pulses longer than 65,535 clocks also keep the periodic implementation.

The cutoff uses each RF generator's `f_fabric`, not the tProcessor frequency.
At 300 MHz, 65,535 clocks are 218.45 us. Short one-shot lengths can vary by one
fabric clock; they do not have the three-clock periodic stop quantization.
The retained long-pulse implementation accepts stop commands at three-clock
block boundaries, so its output can remain active for up to two extra clocks.

This change applies to the shared RF pulse compiler. Firmware and the separate
continuous-tone S-parameter acquisition path are unchanged.

Verification scripts live in QSTL_QICK under
`qick/firmware/ip/rc_precomp_validation/run_rf_duration_validation.py`.
Software tests cover fixed lengths, threshold crossings, descending sweeps,
20 x 20 Cartesian loops with two repetitions, and existing GUI/RC integrations.
The RTL batch checks both short-duration AWG timing modes on the full 20 x 20
grid, plus separate 2 x 2 threshold fixtures with zero AWG waveforms.
