import pandas as pd
import numpy as np
import requests

from scipy.linalg import solve_triangular
from numpy.linalg import qr, norm

from Orbit_Utility import getEphem, Cartesian_to_spherical, getEclipses, eclipseFilter


STEP = 240 #Frequency (in minutes) of observations to ask to the horizons database
R_EARTH = 6.378e3
R_MOON = 1.738e3 #All mesures are on km



print("Welcome to my (very inefficient) solar eclipse predictor. Input a year to see the dates of the eclipses of that year:")

date = input()
end_date = str(int(date) + 1)


print("Downloading Earth's and Moon's data for the year", date)
Terra = getEphem(bodyId = 399, startDate=f"{date}-JAN-01", endDate=f"{end_date}-JAN-01", step = STEP)
Lluna = getEphem(bodyId = 301, startDate=f"{date}-JAN-01", endDate=f"{end_date}-JAN-01", step = STEP)
print("Done! Checking for eclipses...")

Terra_esf = Cartesian_to_spherical(Terra)
Lluna_esf = Cartesian_to_spherical(Lluna)



"""
Idea: For an eclipse, the sun, earth and moon should be more or
 less aligned. By trigponometry we may figue out when
 this happens.
"""


comp = eclipseFilter(Terra_esf, Lluna_esf, alpha=0.3)
eclipses = getEclipses(comp, steps=2, interval=2)





