import numpy as np

def radius_of_gyration(positions:list, weights:list):
    '''
    Description
    -----------
    Computes the radius of gyration for an individual given a list of positions.
    Computes center of mass and great circle distance as intermediate steps

    Parameters
    ----------
    positions : list[tuple]
        A list of tuples consisting of (lat, lon)
    weights: list
        List of visit counts / durations 
    
    Returns
    -------
    The Radius of Gyration for a specific individual.
    '''
    cm = center_of_mass(positions, weights)
    total_weight = sum(weights)

    squared_distances = [
        great_circle_distance(pt, cm)**2 * w for pt, w in zip(positions, weights)
    ]
    return np.sqrt(sum(squared_distances) / total_weight)

def normalized_entropy(visit_counts):
    '''
    Description
    -----------
    Computes the normalized entropy (measure of unpredictability, scaled) for an individual
    given a list containing the frequency of visits to computed locations.

    Parameters
    ----------
    visit_counts : list
        A list of counts representing the number of times an individual was at a unique location.
    
    Returns
    -------
    Entropy normalized
    '''
    if not visit_counts or len(visit_counts) <= 1:
        return 0.0 # Single location == zero unpredictability
    
    total = sum(visit_counts)
    N = len(visit_counts)

    probas = [count / total for count in visit_counts]
    shannon = -sum(p * np.log2(p) for p in probas if p > 0)
    
    max_entropy = np.log2(N)
    return shannon / max_entropy

def great_circle_distance(pt1:tuple, pt2:tuple):
    '''
    Description
    -----------
    Implements the haversine formulat to determine the distance between two points in meters.

    Parameters
    ----------
    lat1, lon1, lat2, lon2 : float
        scaler values representing the lat, lon values for two distinct points.
    
    Returns
    -------
    The distance between the two points provided in meters
    '''
    lat1, lon1 = pt1
    lat2, lon2 = pt2

    R = 6371 # Radius in kms
    phi_1, phi_2 = np.radians(lat1), np.radians(lat2) # Equitorial distance scalers
    delta_phi = np.radians(lat2 - lat1) # Change in latitutde (in radians)
    delta_lambda = np.radians(lon2 - lon1) # Change in longitude (in radians)
    a = np.sin(delta_phi / 2)**2 + np.cos(phi_1) * np.cos(phi_2) * np.sin(delta_lambda / 2) ** 2
    c = np.atan2(np.sqrt(a), np.sqrt(1-a))
    return R * c * 1000

def center_of_mass(positions:list, weights:list):
    '''
    Description
    -----------
    Computes the center of mass for an individual 
    given a list of positions and frequency / duration of visits to positions.
    
    Parameters
    ----------
    positions : list[tuple]
        A list of tuples consisting of (lat, lon)
    weights: list
        List of visit counts / durations 
    
    Returns
    -------
    center of mass : tuple
        center of mass for an individuals movements as a tuple (lat, lon)
    '''
    total_weight = sum(weights)
    lat_cm = sum(lat * w for (lat, _), w in zip(positions, weights)) / total_weight
    lon_cm = sum(lon * w for (_, lon), w in zip(positions, weights)) / total_weight
    return (lat_cm, lon_cm)

def jump_length():

    pass

def return_frequency():
    pass