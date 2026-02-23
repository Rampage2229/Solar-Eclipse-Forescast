import pandas as pd
import numpy as np
import requests
from io import StringIO
from numpy.linalg import qr, norm
from scipy.linalg import solve_triangular
R_EARTH = 6.378e3
R_MOON = 1.738e3
#Function to get ephemerides
def getEphem(bodyId, startDate, endDate, step = 60, dataType = 2, verbose = False):
    if(verbose):
        print("Downloading data for object with ID", bodyId)
    url = (
        "https://ssd.jpl.nasa.gov/api/horizons.api?format=text"
        "&MAKE_EPHEM='YES'"
        "&OBJ_DATA='YES'"
        
        f"&COMMAND='{bodyId}'" 
        "&EPHEM_TYPE='VECTORS'"
        "&CENTER='500@10'"
        f"&VEC_TABLE={dataType}"
        "&CSV_FORMAT='YES'"
        f"&START_TIME='{startDate}'"  
        f"&STOP_TIME='{endDate}'"     
        f"&STEP_SIZE='{step}m'"  
    )
    if(dataType == 2):
        request = requests.get(url)
        text = request.content.decode('utf-8')
        start = text.find('$$SOE') + 5
        end = text.find('$$EOE')
        if(text.find('$$SOE') < 0):
            return pd.DataFrame()
        data = "JDTDB,UTC,x,y,z,vx,vy,vz," + text[start:end]
        data = data.replace(', \n', ',\n')
        s = StringIO(data)
        dataFrame = pd.read_csv(s, sep=',')
        if(verbose):
            print("Done!")
        return dataFrame
    if(dataType == 3):
        request = requests.get(url)
        text = request.content.decode('utf-8')
        start = text.find('$$SOE') + 5
        end = text.find('$$EOE')
        
        data = "JDTDB,UTC,x,y,z,vx,vy,vz," + text[start:end]
        data = data.replace(', \n', ',\n')
        s = StringIO(data)
        dataFrame = pd.read_csv(s, sep=',')
        if(verbose):
            print("Done!")
        return dataFrame
        
def eclipseFilter(planet, satelite, alpha=1.0):
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
    compare["delta_dist"] = np.sqrt(compare.x**2 + compare.y**2)
    
    '''
    The alpha value will dictate how "strict" does the solapation between both projections
    (of Earth and the Moon) have to be in order to consider the encounter as an eclipse.
     Values greater than 1 mean stricter solapation, while values less than 1 indicate that
    the Moon and the Earth need not to solapate for an eclipse to be registered.
    
    A value lesser or equal to 0 will count all events as eclipses, as long as the moon
    is closer to the Moon that Earth.
    
    DEFAULT: ALPHA=1.0
    '''
    compare["observed_radius"] = (R_MOON + R_EARTH * (np.sqrt(compare.R_sq_Satelite / compare.R_sq_Planet)))
    compare["eclipse"] = ( (compare.delta_dist * alpha < compare.observed_radius)
    & (compare.R_sq_Satelite < compare.R_sq_Planet) )
    return compare.loc[compare['eclipse']]



def Cartesian_to_spherical(data):
    if(data.shape[0] < 1):
         pass
    spherical = pd.DataFrame(columns = ["R_sq", "phi", "theta", "UTC"])
    spherical["R_sq"] = data.x**2 + data.y**2 + data.z**2
    spherical["phi"] = np.arctan(data.y / data.x)
    spherical["theta"] = np.arccos(data.z/ np.sqrt(spherical.R_sq))
    spherical["UTC"] = data["UTC"]
    return spherical
    

def getEclipses(comp, steps = 2, interval=1, verbose=False):

    eclipses = [d for _, d in comp.groupby(comp.index - np.arange(len(comp)))]
    #We classify the eclipses checking if the measurements 
    #are made one next to another in time
        
            
            
    #Now we do a quadratic aproximation to get the values
    for x in eclipses:
        if(x.shape[0] < 1):
            continue
        if(verbose):
            print("Verifying a possible eclipse...")
        x = pd.DataFrame(x)
        beg = x.UTC.iat[0]
        end = x.UTC.iat[-1]
        Earth = getEphem(399, beg, end, step=interval, dataType = 2)
        Moon = getEphem(301, beg, end, step=interval, dataType = 2)
        if(Earth.empty or Moon.empty):
            if(verbose):
                print("Eclipse discarted")
            continue
        Earth_c = Cartesian_to_spherical(Earth)
        Moon_c = Cartesian_to_spherical(Moon)
        
        x = eclipseFilter(Earth_c, Moon_c, alpha=0.85)
        distances_x = x.x
        distances_y = x.y
        c_x = np.array(distances_x, dtype=np.float64)
        c_y = np.array(distances_y, dtype=np.float64)
        n = c_x.shape[0]
        if(n < 3):
            if(verbose):
                print("Eclipse discarted")
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
            dist = np.sqrt(x_i**2 + y_i**2)
            if(dist < best_distance):
                best_time = t
                best_distance = dist
        best_time /= steps
        
        
        
        #Check if we indeed have an eclipse, and then print the date of it

        best_time_ = int(best_time / (60 /interval))
        hours = int(str(x.UTC.iloc[best_time_])[18:20])
        minutes = int(str(x.UTC.iloc[best_time_])[21:23])
        seconds = int(str(x.UTC.iloc[best_time_])[24:26])
        hours += best_time_
        minutes = ((best_time / (60 /interval)) - best_time_)*60
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
        s = str(x.UTC.iloc[best_time_])[:18] + string_h + ':' + string_m + ':' +  string_s + str(x.UTC.iloc[best_time_])[26:]
        #Printing the result
        print('-'*10)
        print("Confirmed eclipse on" + s)
        print("With " + str(n) + " observations.")
    return eclipses
