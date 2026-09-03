# Device measurement data

The device-calibrated simulations require the measured conductance traces
(not bundled in this repository; available from the authors upon request):

- `pcm.hdf5` - 1,189 PCM devices, soft SET (10,000 pulses each)
- `fm.hdf5` - 1,268 RRAM devices, soft RESET (5,000 pulses each)

Place the files in this folder. Without them, `reproduce.py pcm_mnist` prints a
warning and runs the FP32 baselines only.
