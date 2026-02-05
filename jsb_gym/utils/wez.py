from jsb_gym.utils.units import f2m

def gunsnap(r_wez, max_range=f2m(3_000), min_range=f2m(500)):
    """Convert a distance to target to damage.

    Args:
        r_wez (float): distance to target.

    Returns:
        float: damage.
    """
    if r_wez > max_range:
        return 0.0
    elif r_wez < max_range and r_wez >= min_range:
        return (max_range - r_wez)/(max_range - min_range)
    else:
        return 0.0
