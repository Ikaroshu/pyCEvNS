# pyCEvNS -- an open-source CEvNS calculator with MCMC sampling of new physics parameters and experimental systematics

This package provides convenient methods for calculting neutrino experiment events and fitting new physics parameter.

## Basic usage

import ingredient:
```python
from pyCEvNS.flux import *
from pyCEvNS.oscillation import *
from pyCEvNS.detectors import *
from pyCEvNS.events import *
```

define your neutrino flux, detector, and neutrino interaction:
```python
det = Detector('csi', efficiency=eff_coherent)
flux = NeutrinoFluxFactory().get('coherent')
interaction = NeutrinoNucleusElasticVector(NSIparameters(), HelmFormFactor(5.5))
exposure = 4466
```

calculate the events:
```python
print(interaction.events(det.er_min, det.er_max, 'e', flux, det, exposure) + 
      interaction.events(det.er_min, det.er_max, 'ebar', flux, det, exposure)+
     interaction.events(det.er_min, det.er_max, 'mu', flux, det, exposure)+
     interaction.events(det.er_min, det.er_max, 'mubar', flux, det, exposure)+
     interaction.events(det.er_min, det.er_max, 'tau', flux, det, exposure)+
     interaction.events(det.er_min, det.er_max, 'taubar', flux, det, exposure))
```
```python
138.46516348290166
```

### Oscllations
The neutrino flux can go through a long distance and oscillate, this can be done via:
```python
fs =NeutrinoFluxFactory().get('solar')
osc = OscillatorFactory().get('solar')
fs = osc.transform(fs)
```
Now ``fs`` is the oscillated flux at the detector.