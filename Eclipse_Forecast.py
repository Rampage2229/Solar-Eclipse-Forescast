import pandas as pd
import numpy as np
import requests
import math

from scipy.linalg import solve_triangular
from numpy.linalg import qr

from Orbit_Utility import getEphem, Cartesian_to_spherical


print("Welcome to my (very inefficient) solar eclipse predictor. Input a year to see the dates of the eclipses of that year:")

date = input()
end_date = str(int(date) + 1)




STEP = 20 #Frequency (in minutes) of observations to ask to the horizons database

print("Downloading Earth's data...")
Terra = getEphem(bodyId = 399, startDate=f"{date}-JAN-01", endDate=f"{end_date}-JAN-01", step = STEP)
print("Downloading Moon's data...")
Lluna = getEphem(bodyId = 301, startDate=f"{date}-JAN-01", endDate=f"{end_date}-JAN-01", step = STEP)
print("Done! Checking for eclipses...")
"""
Idea: For an eclipse, the sun, earth and moon should be more or
 less aligned. By trigponometry we may figue out when
 this happens.
"""
R_EARTH = 6.378e3
R_MOON = 1.738e3 #All mesures are on km


Terra_esf = Cartesian_to_spherical(Terra)
Lluna_esf = Cartesian_to_spherical(Lluna)


def d2(x, y):
    return np.sqrt(x**2 + y**2)

def compare2(planet, satelite, alpha=1.0):
    #This function returns a merged DataFrame with extra columns. We are interested, in particular, on the
    #column "eclipse" which contains bools, and will be set to True when the algorithm finds a (potential) eclipse.
    
    #First we merge both DataFrames
    planet = planet.add_suffix('_Planet')
    planet = planet.rename(columns={'UTC_Planet':'UTC'})
    satelite = satelite.add_suffix('_Satelite')
    satelite = satelite.rename(columns={'UTC_Satelite':'UTC'})
    compare = pd.merge(planet, satelite, how='inner', on="UTC")
    
    #Now we do some manipulation on the coordinates to allow easy comparaison later
    #To be non-destructive, we will add this information as new columns
    compare["delta_theta"] = compare.theta_Satelite - compare.theta_Planet
    compare["delta_phi"] = compare.phi_Satelite - compare.phi_Planet
    
    compare["x"] = np.sqrt(compare.R_sq_Satelite) * np.sin(compare.delta_phi)
    compare["y"] = np.sqrt(compare.R_sq_Satelite) * np.sin(compare.delta_theta)
    
    #Now we check if there is an eclipse and return
    compare["delta_dist"] = d2(compare.x, compare.y)
    
    '''
    The alpha value will dictate how "strict" does the solapation between both projections
    (of Earth and the Moon) have to be in order to consider the encounter as an eclipse.
     Values greater than 1 mean stricter solapation, while values less than 1 indicate that
    the Moon and the Earth need not to solapate for an eclipse to be registered.
    
    A value lesser or equal to 0 will count all events as eclipses, as long as the moon
    is closer to the Moon that Earth.
    
    DEFAULT: ALPHA=1.0
    '''
    ALPHA = alpha
    
    compare["observed_radius"] = (R_MOON + R_EARTH * (np.sqrt(compare.R_sq_Satelite / compare.R_sq_Planet)))
    compare["eclipse"] = ( (compare.delta_dist * ALPHA < compare.observed_radius)
    & (compare.R_sq_Satelite < compare.R_sq_Planet) )
    return compare



def analyze(planet, satelite, steps = 2):
    compare = compare2(planet, satelite)
    compare = compare.loc[compare['eclipse']]
    last = compare.index.values[0]
    count = 0
    eclipses = [[]]
    #We classify the eclipses checking if the measurements 
    #are made one next to another in time
    for i in range(compare.shape[0]):
        if(last == (compare.index.values[i]) - 1):
            eclipses[count].append(compare.iloc[i])
            last = compare.index.values[i]
        else:
            count += 1
            last = compare.index.values[i]
            eclipses.append([compare.iloc[i]])
            
            
    #Now we do a quadratic aproximation to get the values
    
    for x in eclipses:
        distances_x = []
        distances_y = []
        for obs in x:
            distances_x.append(obs.x)
            distances_y.append(obs.y)
        c_x = np.array(distances_x, dtype=np.float64)
        c_y = np.array(distances_y, dtype=np.float64)
        n = c_x.shape[0]
        if(n < 3):
            continue
        A_x = np.zeros((n,3))
        A_x[:,0] = 1
        A_y = np.zeros((n,3))
        A_y[:,0] = 1
        for i in range(n):
            A_x[i,1] = i
            A_x[i,2] = (i)**2
            A_y[i,1] = i
            A_y[i,2] = (i)**2
        Q_x, R_x = qr(A_x, mode='reduced')
        Qtc_x = Q_x.T@c_x
        Q_y, R_y = qr(A_y, mode='reduced')
        Qtc_y = Q_y.T@c_y
        p_x = solve_triangular(R_x, Qtc_x)
        p_y = solve_triangular(R_y, Qtc_y)
        
        best_distance = 999999999999
        best_time = 0
        for i in range(steps * n):
            t = (i/steps)
            x_i = p_x[2] * t**2 + p_x[1] * t + p_x[0]
            y_i = p_y[2] * t**2 + p_y[1] * t + p_y[0]
            dist = d2(x_i, y_i)
            if(dist < best_distance):
                best_time = t
                best_distance = dist
        best_time /= steps
        
        
        
        #Check if we indeed have an eclipse, and then print the date of it
        if(x[0].observed_radius > best_distance):
            #A bit of manipulation to get the correct date format...
            best_time_ = int(best_time)
            hours = int(str(x[best_time_].UTC)[18:20])
            minutes = int(str(x[best_time_].UTC)[21:23])
            seconds = int(str(x[best_time_].UTC)[24:26])
            hours += best_time_
            minutes = (best_time - best_time_)*60
            minutes_ = int(minutes)
            seconds = int((minutes - minutes_)*60)
            
            string_h = str(hours) if hours >= 10 else '0'+str(hours)
            string_m = str(minutes_) if minutes_ >= 10 else '0'+str(minutes_)
            string_s = str(seconds) if seconds >= 10 else '0'+str(seconds)
            
            '''
            Notice that, as steps are strictly smaller than the time intervals of
            the measure, it is impossible for the program to register an impossible date:
            In such case, it would have directly picked the next timestep and correctly
            recognize that date on a different day.
            '''
            
            
            s = str(x[best_time_].UTC)[:18] + string_h + ':' + string_m + ':' +  string_s + str(x[best_time_].UTC)[26:]
            #Printing the result
            print('-'*10)
            print("Confirmed eclipse on" + s)
            print("With " + str(n) + " observations.")

        
    return eclipses
        

eclipses = analyze(Terra_esf, Lluna_esf)
