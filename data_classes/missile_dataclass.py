from dataclasses import dataclass

@dataclass
class MissileLimits:
    phi_min: float
    phi_max: float
    theta_min: float
    theta_max: float
    psi_min: float
    psi_max: float
    alt_min: float
    alt_max: float
    thr_min: float
    thr_max: float


@dataclass
class PIDGains:
    P: float
    I: float
    D: float
    Deriv: float
    Integ: float
    Integ_max: float
    Integ_min: float

@dataclass
class MissilePIDGains:
    Roll: PIDGains
    Pitch: PIDGains
    Heading: PIDGains


@dataclass
class MissileNavigation:
    N: float
    dt: float
    cp: float
    acceleration_stage_in_sec: float
    dive_at: float  
    tan_ref: float
    tan_ref_min: float
    tan_ref_max: float
    theta_min_cruise: float
    theta_max_cruise: float
    theta_min: float
    theta_max: float
    alt_cruise: float
    
@dataclass
class MissileSimulation:
    Sim_time_step: float
    Control_time_step: float

@dataclass
class MissilePerformance:
    target_lost_below_mach: float
    target_lost_below_alt: float
    lost_count: float
    effective_radius: float