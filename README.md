# RF-Track simulations for Final Cooling

Please note: to use RF-Track, RF-Track python module should be installed (compiled modules can be found here: https://gitlab.cern.ch/rf-track/download).

The channel is built as follows:
 - Initial beam is created as a 6D bunch, and is trcked through all cells, incl. cutting off the outliying particles (tails).
 - Each cell is composed of 4m solenoid, with peak field 40 T, LH absorber and matching coils placed with a specified offset wrt solenoid center
 - Solenoid segment is followed by i) for cells 1-5: drift, rotating RF and accelerating RF ii) cells 6-8: accelerating RF, drift, rotating RF
 - Last cell, 9 does not inclide rf section and should be later extended with an accelration system to transfer the bunch to the next stage afterr final cooling.

To cut off the outlying particles, scikit-learn library is requiered. Other requiered libraries are standard data processing libraries scipy, numpy, matplotlib, stats, json.

   
