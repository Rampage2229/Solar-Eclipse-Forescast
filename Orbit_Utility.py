import pandas as pd
import numpy as np
import requests
import math
from io import StringIO

#Function to get ephemerides
def getEphem(bodyId, startDate, endDate, step = 20):
    url = (
        "https://ssd.jpl.nasa.gov/api/horizons.api?format=text"
        "&MAKE_EPHEM='YES'"
        "&OBJ_DATA='YES'"
        
        f"&COMMAND='{bodyId}'" 
        "&EPHEM_TYPE='VECTORS'"
        "&CENTER='500@10'"
        "&VEC_TABLE=2"
        "&CSV_FORMAT='YES'"
        f"&START_TIME='{startDate}'"  
        f"&STOP_TIME='{endDate}'"     
        f"&STEP_SIZE='{step}m'"  
    )
    request = requests.get(url)
    text = request.content.decode('utf-8')
    start = text.find('$$SOE') + 5
    end = text.find('$$EOE')
    data = "JDTDB,UTC,x,y,z,vx,vy,vz," + text[start:end]
    data = data.replace(', \n', ',\n')
    s = StringIO(data)
    dataFrame = pd.read_csv(s, sep=',')
    return dataFrame




def Cartesian_to_spherical(data):
    spherical = pd.DataFrame(columns = ["R_sq", "phi", "theta", "UTC"])
    spherical["R_sq"] = data.x**2 + data.y**2 + data.z**2
    spherical["phi"] = np.arctan(data.y / data.x)
    spherical["theta"] = np.arccos(data.z/ np.sqrt(spherical.R_sq))
    spherical["UTC"] = data["UTC"]
    return spherical
    

