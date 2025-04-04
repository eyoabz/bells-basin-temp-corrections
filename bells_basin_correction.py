#%%
import numpy as np
import pandas as pd
import math
#Temprature corrections functions
#--------------------------------------
def log10(x):
    """Base 10 logarithm function."""
    if x <= 0:
        # raise ValueError("log10: input must be positive")
        x=0.0000001
    return math.log10(x)

#Bell equations
def bells_sin155(hr):
    if hr >= 11 or hr <= 5:
        if hr < 5:
            hr = hr + 24
        sSin155_0 = np.sin((2 * np.pi) * (hr - 15.5) / 18)
    else:
        sSin155_0 = -1
    return sSin155_0


def bells_sin135(hr):
    if hr >= 9 or hr <= 3:
        if hr < 5:
            hr = hr + 24
        sSin135_0 = np.sin((2 * np.pi) * (hr - 13.5) / 18)
    else:
        sSin135_0 = -1
    return sSin135_0

def bells_temp(Tsur_C, Hac_mm, Tair_C, sSin155, sSin135):
    term1 = 0.95
    term2 = 0.892 * Tsur_C
    term3 = (np.log10(0.5 * Hac_mm) - 1.25) * (-0.448 * Tsur_C + 0.621 * Tair_C + 1.83 * sSin155)
    term4 = 0.042 * Tsur_C * sSin135
    return np.round(term1 + term2 + term3 + term4, 1)

def calculate_delta36(ac_thickness, latitude, defl36, temperature):
    """
    Calculate delta36 value based on equation 15 from the document.
    
    Parameters:
    ac_thickness (float): Asphalt concrete thickness in mm
    latitude (float): Site latitude in degrees
    defl36 (float): Deflection at 36 inches (915mm) from load center in μm
    temperature (float): Mid-depth asphalt temperature in °C
    
    Returns:
    float: delta36 value in μm
    """
    log_ac = log10(ac_thickness)
    log_lat = log10(latitude)
    log_defl36 = log10(defl36)
    
    log_delta36 = (3.05 - 1.13 * log_ac + 
                   0.502 * log_lat * log_defl36 - 
                   0.00487 * temperature * log_lat * log_defl36 + 
                   0.00677 * temperature * log_ac * log_lat)
    
    return 10**log_delta36

def calculate_delta24(ac_thickness, latitude, defl36, temperature):
    """
    Calculate delta24 value based on equation 14 from the document.
    
    Parameters:
    ac_thickness (float): Asphalt concrete thickness in mm
    latitude (float): Site latitude in degrees
    defl36 (float): Deflection at 36 inches (915mm) from load center in μm
    temperature (float): Mid-depth asphalt temperature in °C
    
    Returns:
    float: delta24 value in μm
    """
    log_ac = log10(ac_thickness)
    log_lat = log10(latitude)
    log_defl36 = log10(defl36)
    
    log_delta24 = (3.30 - 1.32 * log_ac + 
                   0.514 * log_lat * log_defl36 - 
                   0.00622 * temperature * log_lat * log_defl36 + 
                   0.00838 * temperature * log_ac * log_lat)
    
    return 10**log_delta24

def calculate_delta12(ac_thickness, latitude, defl36, temperature):
    """
    Calculate delta12 value based on equation 12 from the document.
    
    Parameters:
    ac_thickness (float): Asphalt concrete thickness in mm
    latitude (float): Site latitude in degrees
    defl36 (float): Deflection at 36 inches (915mm) from load center in μm
    temperature (float): Mid-depth asphalt temperature in °C
    
    Returns:
    float: delta12 value in μm
    """
    log_ac = log10(ac_thickness)
    log_lat = log10(latitude)
    log_defl36 = log10(defl36)
    
    log_delta12 = (3.45 - 1.59 * log_ac + 
                   0.489 * log_lat + 
                   0.449 * log_defl36 - 
                   0.0275 * temperature + 
                   0.012 * temperature * log_ac * log_lat)
    
    return 10**log_delta12

def calculate_delta8(ac_thickness, latitude, defl36, temperature):
    """
    Calculate delta8 value based on equation 11 from the document.
    
    Parameters:
    ac_thickness (float): Asphalt concrete thickness in mm
    latitude (float): Site latitude in degrees
    defl36 (float): Deflection at 36 inches (915mm) from load center in μm
    temperature (float): Mid-depth asphalt temperature in °C
    
    Returns:
    float: delta8 value in μm
    """
    log_ac = log10(ac_thickness)
    log_lat = log10(latitude)
    log_defl36 = log10(defl36)
    
    log_delta8 = (3.02 - 1.49 * log_ac + 
                  0.541 * log_lat + 
                  0.394 * log_defl36 - 
                  0.0230 * temperature + 
                  0.0111 * temperature * log_ac * log_lat)
    
    return 10**log_delta8

def calculate_delta60(ac_thickness, latitude, defl36, temperature):
    """
    Calculate delta60 value based on equation 16 from the document.
    
    Parameters:
    ac_thickness (float): Asphalt concrete thickness in mm
    latitude (float): Site latitude in degrees
    defl36 (float): Deflection at 36 inches (915mm) from load center in μm
    temperature (float): Mid-depth asphalt temperature in °C
    
    Returns:
    float: delta60 value in μm
    """
    log_ac = log10(ac_thickness)
    log_defl36 = log10(defl36)
    
    # Note: delta36 is used in the equation instead of defl36
    delta36 = calculate_delta36(ac_thickness, latitude, defl36, temperature)
    log_delta36 = log10(delta36)
    
    log_delta60 = (2.67 - 0.770 * log_ac + 
                   0.650 * log_delta36 + 
                   0.00290 * temperature * log_ac)
    
    return 10**log_delta60

def get_appropriate_delta_function(ac_thickness):
    """
    Determine which delta function to use based on AC thickness.
    
    Parameters:
    ac_thickness (float): Asphalt concrete thickness in mm
    
    Returns:
    function: The appropriate delta calculation function
    """
    if ac_thickness <= 100:
        return calculate_delta24
    elif ac_thickness <= 200:
        return calculate_delta36
    else:
        return calculate_delta60

def calculate_temperature_adjustment_factor(ac_thickness, latitude, defl36, 
                                           measured_temp, reference_temp=21.1):
    """
    Calculate the temperature adjustment factor (TAF) for FWD deflections.
    
    Parameters:
    ac_thickness (float): Asphalt concrete thickness in mm
    latitude (float): Site latitude in degrees
    defl36 (float): Deflection at 36 inches (915mm) from load center in μm
    measured_temp (float): Mid-depth asphalt temperature during measurement in °C
    reference_temp (float): Reference temperature to adjust to, default 20°C
    
    Returns:
    float: Temperature adjustment factor (TAF)
    """
    delta_function = get_appropriate_delta_function(ac_thickness)
    
    delta_measured = delta_function(ac_thickness, latitude, defl36, measured_temp)
    delta_reference = delta_function(ac_thickness, latitude, defl36, reference_temp)
    if defl36 + delta_measured == 0:
        raise ValueError("Division by zero encountered in TAF calculation")
    taf = (defl36 + delta_reference) / (defl36 + delta_measured)
    
    # Calculate TAF using equation 23
    taf = (defl36 + delta_reference) / (defl36 + delta_measured)
    
    return taf

def adjust_deflections(deflections, taf):
    """
    Apply temperature adjustment factor to all deflection measurements.
    
    Parameters:
    deflections (list/array): List of deflection measurements in μm
    taf (float): Temperature adjustment factor
    
    Returns:
    list: Temperature-adjusted deflection measurements
    """
    return [d * taf for d in deflections]

def fwd_temperature_adjustment(deflections, ac_thickness, latitude, measured_temp, 
                              reference_temp=20.0):
    """
    Complete FWD temperature adjustment procedure.
    
    Parameters:
    deflections (list): List of deflection measurements [D0, D8, D12, D24, D36, ...]
    ac_thickness (float): Asphalt concrete thickness in mm
    latitude (float): Site latitude in degrees
    measured_temp (float): Mid-depth asphalt temperature during measurement in °C
    reference_temp (float): Reference temperature to adjust to, default 20°C
    
    Returns:
    list: Temperature-adjusted deflection measurements
    """
    # Extract defl36 from the deflection list (assuming it's the 5th element, index 4)
    if len(deflections) >= 5:
        defl36 = deflections[4]
    else:
        raise ValueError("Input deflections list must include D36 (5th element)")
    
    # Calculate temperature adjustment factor
    taf = calculate_temperature_adjustment_factor(
        ac_thickness, latitude, defl36, measured_temp, reference_temp
    )
    
    # Apply adjustment to all deflections
    adjusted_deflections = adjust_deflections(deflections, taf)
    
    return adjusted_deflections




# Example usage
def example():
    # Input parameters
    ac_thickness = 150  # mm
    latitude = 40       # degrees North
    measured_temp = 30  # °C
    reference_temp = 20 # °C
    
    # Measured deflections (D0, D8, D12,D18, D24, D36)
    deflections = [400,300,250,170,90]  # μm
    
    # Perform temperature adjustment
    adjusted_deflections = fwd_temperature_adjustment(
        deflections, ac_thickness, latitude, measured_temp, reference_temp
    )
    
    # Print results
    print(f"Input parameters:")
    print(f"  AC thickness: {ac_thickness} mm")
    print(f"  Latitude: {latitude} degrees")
    print(f"  Measured temperature: {measured_temp}°C")
    print(f"  Reference temperature: {reference_temp}°C")
    print(f"\nMeasured deflections (μm): {deflections}")
    print(f"Temperature-adjusted deflections (μm): {[round(d, 1) for d in adjusted_deflections]}")
    
    # Calculate adjustment factor for verification
    defl36 = deflections[4]
    taf = calculate_temperature_adjustment_factor(
        ac_thickness, latitude, defl36, measured_temp, reference_temp
    )
    print(f"\nTemperature Adjustment Factor (TAF): {taf:.3f}")

if __name__ == "__main__":
    example()



# %%
