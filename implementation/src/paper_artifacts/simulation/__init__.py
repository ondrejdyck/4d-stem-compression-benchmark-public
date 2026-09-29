"""Multislice simulation of the 4D-STEM dataset shown in the Discussion.

This is the simulation code that produced the dataset of Figure 6 and the
code-length comparison in Table 6. The physics is PySlice's; see ``README.md``
for the pinned version.

Importing this package selects single precision. PySlice defaults to
``float64``/``complex128`` on CUDA, which costs a factor of 3.8 in runtime here
and buys nothing measurable: against double, the ADF fraction differs by 1.5e-7
relative. :mod:`paper_artifacts.simulation._precision` records the measurements
and does the selecting.
"""

from paper_artifacts.simulation import _precision
from paper_artifacts.simulation.config import MULTISLICE_PRECISION

_precision.apply(MULTISLICE_PRECISION)
