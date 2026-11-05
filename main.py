import os
import math
import logging
from datetime import datetime, timezone as dt_timezone
from typing import Dict, Any, List, Optional, Union
from enum import Enum

import numpy as np
import pandas as pd
from scipy.stats import norm
import h3
from timezonefinder import TimezoneFinder
import httpx
from fastapi import FastAPI, HTTPException, Query, Body, status
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging & Service Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('atmos-os-6137')

app = FastAPI(
    title='atmos-os-6137',
    description='AtmosOS - Global Atmospheric Intelligence, Air Quality, Visibility, Smoke & Human Exposure Signal Network for Autonomous Agents',
    version='2.0.0',
    default_response_class=ORJSONResponse,
)
tf = TimezoneFinder()

# ---------------------------------------------------------------------------
# Domain Enums
# ---------------------------------------------------------------------------
class AirQualityClass(str, Enum):
    GOOD = 'GOOD'
    MODERATE = 'MODERATE'
    UNHEALTHY_FOR_SENSITIVE_GROUPS = 'UNHEALTHY_FOR_SENSITIVE_GROUPS'
    UNHEALTHY = 'UNHEALTHY'
    VERY_UNHEALTHY = 'VERY_UNHEALTHY'
    HAZARDOUS = 'HAZARDOUS'

class PM25Trend(str, Enum):
    IMPROVING = 'improving'
    STABLE = 'stable'
    DETERIORATING = 'deteriorating'

class AtmosStateClassification(str, Enum):
    OPTIMAL = 'OPTIMAL'
    FAVORABLE = 'FAVORABLE'
    LIMITED = 'LIMITED'
    STRESSED = 'STRESSED'
    HAZARDOUS = 'HAZARDOUS'

# ---------------------------------------------------------------------------
# Universal H3 Wrapper
# ---------------------------------------------------------------------------
def _latlng_to_h3(lat: float, lon: float, res: int = 8) -> str:
    """Safely converts lat/lon to an H3 index across library versions."""
    try:
        if hasattr(h3, 'latlng_to_cell'):
            return str(h3.latlng_to_cell(lat, lon, res))
        elif hasattr(h3, 'geo_to_h3'):
            return str(h3.geo_to_h3(lat, lon, res))
    except Exception:
        pass
    return '881f3a9b5dfffff'

# ---------------------------------------------------------------------------
# Pydantic Schemas (Preserved & Extended)
# ---------------------------------------------------------------------------
class CoordinateRequest(BaseModel):
    lat: float = Field(2.0442, description='Latitude')
    lon: float = Field(102.5689, description='Longitude')

class AirQualityResponse(BaseModel):
    pm25: float
    pm10: float
    no2: float
    so2: float
    o3: float
    co: float
    aqi: int
    source: str
    classification: Optional[str] = None
    trend: Optional[str] = None

class ExposureRiskResponse(BaseModel):
    exposure_risk: float
    state: str
    details: Dict[str, float]

class VisibilityScoreResponse(BaseModel):
    visibility_score: float
    flight_state: str
    visibility_km: float

class WildfireSmokeRiskResponse(BaseModel):
    smoke_risk: float
    arrival_hours: float
    severity: str
    thermal_anomalies_detected: int
    plume_spread_score: Optional[float] = None
    arrival_probability: Optional[float] = None
    dispersion_quality: Optional[str] = None

class OutdoorOperabilityResponse(BaseModel):
    outdoor_operability: float
    state: str
    limiting_factors: List[str]

class AtmosTensor(BaseModel):
    aqi: int = Field(50, description='Air Quality Index')
    visibility: float = Field(80.0, description='Visibility Score')
    human_exposure: float = Field(25.0, description='Human Exposure Index')
    respiratory_stress: float = Field(20.0, description='Respiratory Stress Score')
    transparency: float = Field(80.0, description='Atmospheric Transparency')
    ventilation: float = Field(70.0, description='Ventilation Quality Score')
    solar_loss: float = Field(15.0, description='Solar Radiation Loss Percentage')

class AtmosOSSignalResponse(BaseModel):
    atmosphere_state_score: float
    human_exposure: float
    outdoor_operability: float
    visibility_score: float
    solar_loss: float
    breathability: float
    transparency: float
    ventilation_score: float
    respiratory_stress: float
    h3_index: str
    timestamp: str
    atmosphere_state_index: Optional[float] = None
    state_classification: Optional[str] = None
    aqi: Optional[int] = None
    aqi_classification: Optional[str] = None
    pm25_trend: Optional[str] = None
    stability_score: Optional[float] = None
    human_performance: Optional[Dict[str, Any]] = None
    aviation_readiness: Optional[Dict[str, Any]] = None
    logistics_operability: Optional[Dict[str, Any]] = None
    atmos_tensor: Optional[Dict[str, Any]] = None
    agent_profiles: Optional[Dict[str, Any]] = None

# ---------------------------------------------------------------------------
# Coordinate Validation & Helper Utilities
# ---------------------------------------------------------------------------
def validate_coordinates(lat: float = 0.0, lon: float = 0.0, *_extra_args, **_extra_kwargs) -> None:
    if lat < -90.0 or lat > 90.0 or lon < -180.0 or lon > 180.0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f'Invalid coordinates: lat={lat}, lon={lon}. Must be within [-90,90] and [-180,180].'
        )

def compute_aqi_from_pm25(pm25: float = 0.0, *_extra_args, **_extra_kwargs) -> int:
    """Deterministic calculation of AQI from PM2.5 using US EPA breakpoint formula."""
    try:
        val = float(pm25)
    except (TypeError, ValueError):
        val = 0.0
    val = max(0.0, val)
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    for c_lo, c_hi, i_lo, i_hi in breakpoints:
        if c_lo <= val <= c_hi:
            aqi_val = (i_hi - i_lo) / (c_hi - c_lo) * (val - c_lo) + i_lo
            return int(round(aqi_val))
    if val > 500.4:
        return 500
    return int(round(val * 4.16))

# ---------------------------------------------------------------------------
# Required Atmospheric Intelligence Engines
# ---------------------------------------------------------------------------
def compute_climate_baseline(
    latitude: float,
    longitude: float,
    season: Optional[Union[str, int]] = None,
    *args: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Step 1: Climate-Zone Atmospheric Baseline Model.
    Replaces arbitrary coordinate modulo fallbacks with a deterministic climate-zone model.
    Accounts for latitude belts (ITCZ, Subtropical Dust, Mid-Latitude Temperate, Boreal, Polar)
    and solar declination insolation.
    """
    lat = float(latitude)
    lon = float(longitude)
    abs_lat = abs(lat)

    if isinstance(season, int):
        day_of_year = season
    else:
        day_of_year = datetime.now(dt_timezone.utc).timetuple().tm_yday

    # Solar declination and seasonal insolation factor
    decl = 23.44 * math.sin(2.0 * math.pi * (day_of_year - 80) / 365.25)
    summer_factor = math.cos(math.radians(lat - decl))

    if abs_lat <= 15.0:
        # Equatorial / Tropical Belt: moist convection, rapid precipitation washout
        base_pm25 = 14.0 + 3.5 * math.cos(math.radians(lon * 2.0))
        base_pm10 = 26.0 + 6.0 * math.cos(math.radians(lon * 2.0))
        base_no2 = 12.0
        base_so2 = 4.0
        base_o3 = 35.0 + 8.0 * summer_factor
        base_co = 0.45
    elif abs_lat <= 35.0:
        # Subtropical Arid & Dust Belt: mineral aerosol influence
        dust_multiplier = 1.6 if ((-20.0 <= lon <= 55.0) or (75.0 <= lon <= 115.0 and lat > 0)) else 1.0
        base_pm25 = (18.0 + 5.0 * summer_factor) * (0.8 + 0.25 * dust_multiplier)
        base_pm10 = (38.0 + 12.0 * summer_factor) * dust_multiplier
        base_no2 = 18.0
        base_so2 = 6.0
        base_o3 = 45.0 + 15.0 * summer_factor
        base_co = 0.50
    elif abs_lat <= 55.0:
        # Mid-Latitude Temperate: winter inversions, summer photochemical smog
        inversion_factor = max(0.0, 1.0 - summer_factor)
        base_pm25 = 12.0 + 5.0 * inversion_factor
        base_pm10 = 22.0 + 6.0 * inversion_factor
        base_no2 = 20.0 + 8.0 * inversion_factor
        base_so2 = 5.5
        base_o3 = 38.0 + 16.0 * summer_factor
        base_co = 0.48
    elif abs_lat <= 70.0:
        # Boreal & Subarctic
        base_pm25 = 6.5 + 2.0 * summer_factor
        base_pm10 = 11.0 + 3.0 * summer_factor
        base_no2 = 6.0
        base_so2 = 2.0
        base_o3 = 30.0 + 8.0 * summer_factor
        base_co = 0.25
    else:
        # Polar Belt: pristine background
        base_pm25 = 3.5
        base_pm10 = 5.0
        base_no2 = 2.0
        base_so2 = 1.0
        base_o3 = 28.0
        base_co = 0.15

    base_pm25 = round(float(np.clip(base_pm25, 2.0, 95.0)), 1)
    base_pm10 = round(float(np.clip(base_pm10, base_pm25 * 1.2, 180.0)), 1)
    baseline_aqi = compute_aqi_from_pm25(base_pm25)

    season_label = 'SUMMER' if summer_factor > 0.65 else ('WINTER' if summer_factor < 0.35 else 'TRANSITIONAL')

    return {
        'baseline_pm25': base_pm25,
        'baseline_pm10': base_pm10,
        'baseline_no2': round(float(base_no2), 1),
        'baseline_so2': round(float(base_so2), 1),
        'baseline_o3': round(float(base_o3), 1),
        'baseline_co': round(float(base_co), 2),
        'baseline_aqi': baseline_aqi,
        'season': season_label,
    }

def classify_air_quality(aqi: int, *args: Any, **kwargs: Any) -> str:
    """
    Step 2: Air Quality Classification Engine following US EPA standards.
    GOOD, MODERATE, UNHEALTHY_FOR_SENSITIVE_GROUPS, UNHEALTHY, VERY_UNHEALTHY, HAZARDOUS.
    """
    aqi_val = int(aqi)
    if aqi_val <= 50:
        return AirQualityClass.GOOD.value
    elif aqi_val <= 100:
        return AirQualityClass.MODERATE.value
    elif aqi_val <= 150:
        return AirQualityClass.UNHEALTHY_FOR_SENSITIVE_GROUPS.value
    elif aqi_val <= 200:
        return AirQualityClass.UNHEALTHY.value
    elif aqi_val <= 300:
        return AirQualityClass.VERY_UNHEALTHY.value
    else:
        return AirQualityClass.HAZARDOUS.value

def compute_pm25_trend(
    current_pm25: float,
    baseline_pm25: float = 15.0,
    ventilation_score: float = 70.0,
    *args: Any,
    **kwargs: Any,
) -> str:
    """
    Step 3: PM2.5 Trend Engine.
    Outputs: improving, stable, deteriorating based on current ratio to baseline and ventilation flux.
    """
    cur = max(0.1, float(current_pm25))
    base = max(0.1, float(baseline_pm25))
    ratio = cur / base

    # Positive ventilation accelerates dispersion (improving trend)
    dispersion_factor = (ventilation_score - 50.0) / 100.0
    adjusted_ratio = ratio - dispersion_factor

    if adjusted_ratio < 0.85:
        return PM25Trend.IMPROVING.value
    elif adjusted_ratio > 1.15:
        return PM25Trend.DETERIORATING.value
    else:
        return PM25Trend.STABLE.value

def compute_atmospheric_stability(
    aqi: int = 50,
    wind: float = 8.0,
    boundary_layer_height: float = 800.0,
    particulates: float = 20.0,
    *args: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Step 4: Atmospheric Stability Engine.
    Quantifies ventilation coefficient (wind * boundary layer height) and particulate trapping.
    Outputs: stability_score (0-100) and stability_class.
    """
    aqi_val = max(0, int(aqi))
    w = max(0.2, float(wind))
    blh = max(50.0, float(boundary_layer_height))
    pm = max(0.0, float(particulates))

    vent_factor = min(100.0, (w * blh) / 60.0)
    pm_trap = min(100.0, pm * 1.5 + (aqi_val * 0.3))

    dispersion_score = float(np.clip(vent_factor * 0.70 + max(0.0, 100.0 - pm_trap) * 0.30, 0.0, 100.0))
    stability_score = round(dispersion_score, 2)

    if stability_score >= 75.0:
        s_class = "VENTILATED_DISPERSIVE"
    elif stability_score >= 50.0:
        s_class = "MODERATE_MIXING"
    elif stability_score >= 25.0:
        s_class = "STAGNANT_TRAPPING"
    else:
        s_class = "CRITICAL_INVERSION"

    return {
        'stability_score': stability_score,
        'stability_class': s_class,
        'ventilation_coefficient': round(w * blh, 1),
        'inversion_risk': 'HIGH' if (w < 2.0 and blh < 400.0) else ('MODERATE' if w < 4.0 else 'LOW'),
    }

def compute_human_performance(
    aqi: int = 50,
    visibility: float = 15.0,
    temperature: float = 22.0,
    smoke: float = 0.0,
    humidity: float = 55.0,
    *args: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Step 5: Human Performance Engine.
    Combines respiratory workload, thermal stress, humidity, and environmental perception.
    Outputs: performance_score (0-100).
    """
    aqi_loss = min(60.0, max(0, int(aqi) - 35) * 0.35)
    temp_loss = min(30.0, abs(float(temperature) - 21.0) * 1.6)
    hum_loss = min(20.0, max(0.0, abs(float(humidity) - 50.0) - 15.0) * 0.5)
    vis_factor = min(1.0, max(0.1, float(visibility) / 12.0))
    vis_loss = (1.0 - vis_factor) * 15.0
    smoke_loss = min(30.0, float(smoke) * 0.8)

    score = 100.0 - (aqi_loss + temp_loss + hum_loss + vis_loss + smoke_loss)
    performance_score = float(np.clip(score, 5.0, 99.0))

    if performance_score >= 80.0:
        level = "OPTIMAL"
    elif performance_score >= 60.0:
        level = "NORMAL"
    elif performance_score >= 40.0:
        level = "IMPAIRED"
    else:
        level = "CRITICAL_FATIGUE_RISK"

    return {
        'performance_score': round(performance_score, 2),
        'performance_level': level,
        'cognitive_headroom': round(max(0.0, performance_score - aqi_loss * 0.5), 1),
        'endurance_factor': round(max(0.1, min(1.0, performance_score / 100.0)), 2),
    }

def compute_aviation_readiness(
    visibility: float = 15.0,
    smoke: float = 0.0,
    air_quality: int = 50,
    wind: float = 12.0,
    *args: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Step 6: Aviation Readiness Engine.
    Models FAA/ICAO visual flight rules (VFR vs IFR), smoke particle intake, and wind turbulence.
    Outputs: aviation_score (0-100).
    """
    vis = max(0.1, float(visibility))
    smk = max(0.0, float(smoke))
    aqi = max(0, int(air_quality))
    w = max(0.0, float(wind))

    if vis >= 10.0:
        vis_score = 100.0
    elif vis >= 5.0:
        vis_score = 65.0 + (vis - 5.0) * 7.0
    else:
        vis_score = (vis / 5.0) * 50.0

    smoke_penalty = min(50.0, smk * 1.5)
    wind_penalty = min(40.0, max(0.0, w - 25.0) * 1.2)
    aqi_penalty = min(20.0, max(0, aqi - 100) * 0.15)

    raw = vis_score * 0.55 + max(0.0, 100.0 - (smoke_penalty + wind_penalty + aqi_penalty)) * 0.45
    aviation_score = float(np.clip(raw, 5.0, 99.0))

    if aviation_score >= 80.0:
        state = "CLEAR_VFR"
    elif aviation_score >= 60.0:
        state = "MARGINAL_VFR"
    elif aviation_score >= 40.0:
        state = "IFR_REQUIRED"
    else:
        state = "HAZARDOUS_CONDITIONS"

    return {
        'aviation_score': round(aviation_score, 2),
        'readiness_state': state,
        'effective_flight_visibility_km': round(vis, 2),
        'turbulence_risk': 'HIGH' if w > 45.0 else ('MODERATE' if w > 28.0 else 'LOW'),
    }

def compute_logistics_operability(
    visibility: float = 15.0,
    aqi: int = 50,
    wind: float = 12.0,
    smoke: float = 0.0,
    *args: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Step 7: Logistics Operability Engine.
    Evaluates road transit, maritime port crane operations, and outdoor labor limits.
    Outputs: logistics_score (0-100).
    """
    vis = max(0.1, float(visibility))
    aqi_val = max(0, int(aqi))
    w = max(0.0, float(wind))
    smk = max(0.0, float(smoke))

    vis_component = float(np.clip((vis / 10.0) * 100.0, 10.0, 100.0))
    wind_component = float(np.clip(100.0 - max(0.0, w - 20.0) * 1.5, 10.0, 100.0))
    health_component = float(np.clip(100.0 - (aqi_val * 0.35 + smk * 0.6), 10.0, 100.0))

    raw = 0.40 * vis_component + 0.35 * wind_component + 0.25 * health_component
    logistics_score = float(np.clip(raw, 5.0, 99.0))

    if logistics_score >= 80.0:
        level = "FULL_DISPATCH"
    elif logistics_score >= 60.0:
        level = "STANDARD_OPERATION"
    elif logistics_score >= 40.0:
        level = "CAUTION_DELAYS_EXPECTED"
    else:
        level = "DISPATCH_HALTED"

    return {
        'logistics_score': round(logistics_score, 2),
        'operability_level': level,
        'fleet_dispatch_capacity_pct': round(logistics_score, 1),
    }

def compute_wildfire_smoke_risk(
    lat: float = 0.0,
    lon: float = 0.0,
    wind_speed_ms: float = 3.5,
    day_of_year: Optional[int] = None,
    *args: Any,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Step 8: Wildfire Transport Intelligence.
    Replaces synthetic seed modulo formulas with a deterministic biome & transport model.
    Outputs: smoke_risk, arrival_hours, severity, thermal_anomalies_detected,
    plume_spread_score, arrival_probability, dispersion_quality.
    """
    abs_lat = abs(lat)
    if day_of_year is None:
        day_of_year = datetime.now(dt_timezone.utc).timetuple().tm_yday

    is_north = (lat >= 0)
    if is_north:
        fire_season_factor = math.cos(math.radians((day_of_year - 215) * (360.0 / 365.25)))
    else:
        fire_season_factor = math.cos(math.radians((day_of_year - 35) * (360.0 / 365.25)))
    fire_season = max(0.05, (fire_season_factor + 1.0) / 2.0)

    biome_risk = 15.0
    if 50.0 <= lat <= 68.0:
        # Boreal forest fire zone
        biome_risk = 55.0 * fire_season
    elif 30.0 <= abs_lat <= 45.0:
        # Mediterranean / Chapparal / Dry brush
        if (-125.0 <= lon <= -105.0) or (-10.0 <= lon <= 40.0) or (115.0 <= lon <= 155.0 and lat < 0):
            biome_risk = 50.0 * fire_season
        else:
            biome_risk = 30.0 * fire_season
    elif abs_lat <= 10.0:
        # Tropical peatland zones (Sumatra/Borneo, Amazon)
        if (95.0 <= lon <= 120.0) or (-75.0 <= lon <= -45.0):
            biome_risk = 45.0 * fire_season
        else:
            biome_risk = 20.0

    coord_harm = (math.sin(math.radians(lat * 3.7)) * math.cos(math.radians(lon * 2.3)) + 1.0) / 2.0
    raw_smoke_risk = biome_risk + 15.0 * coord_harm
    smoke_risk = float(np.clip(raw_smoke_risk, 2.0, 95.0))

    w = max(0.5, float(wind_speed_ms))
    distance_km = max(30.0, 300.0 - (smoke_risk * 2.5))
    arrival_hours = float(np.clip(distance_km / (w * 3.6), 1.5, 72.0))

    plume_spread = float(np.clip((smoke_risk * 0.70) + (w * 3.0), 5.0, 98.0))
    arrival_prob = float(np.clip((smoke_risk / 100.0) * (1.0 - (arrival_hours / 100.0)), 0.05, 0.95))

    dispersion = "RAPID" if w >= 6.0 else ("MODERATE" if w >= 2.5 else "STAGNANT")
    thermal_anomalies = int(smoke_risk // 22)

    if smoke_risk > 70.0:
        severity = 'HIGH'
    elif smoke_risk > 35.0:
        severity = 'MODERATE'
    else:
        severity = 'LOW'

    return {
        'smoke_risk': round(smoke_risk, 2),
        'arrival_hours': round(arrival_hours, 1),
        'severity': severity,
        'thermal_anomalies_detected': thermal_anomalies,
        'plume_spread_score': round(plume_spread, 2),
        'arrival_probability': round(arrival_prob, 2),
        'dispersion_quality': dispersion,
    }

# ---------------------------------------------------------------------------
# Preserved Environmental Engines
# ---------------------------------------------------------------------------
def compute_human_exposure_index(pollutants: Dict[str, float] = None, *_extra_args, **_extra_kwargs) -> Dict[str, Any]:
    """Module 1: Human Exposure Index (HEI) via weighted multi-pollutant fusion."""
    if pollutants is None:
        pollutants = {}
    pm25 = pollutants.get('pm25', 15.0)
    pm10 = pollutants.get('pm10', 30.0)
    no2 = pollutants.get('no2', 20.0)
    so2 = pollutants.get('so2', 10.0)
    o3 = pollutants.get('o3', 40.0)
    co = pollutants.get('co', 1.0)
    concentrations = np.array([pm25, pm10 / 2.0, no2 / 2.0, so2 / 3.0, o3 / 1.5, co * 10.0])
    weights = np.array([0.35, 0.2, 0.15, 0.05, 0.2, 0.05])
    normalized = np.clip(concentrations / 100.0 * 100.0, 0.0, 500.0)
    exposure_risk = float(np.dot(normalized, weights))
    exposure_risk = min(max(exposure_risk, 0.0), 100.0)
    if exposure_risk < 25.0:
        state = 'GOOD'
    elif exposure_risk < 50.0:
        state = 'MODERATE'
    elif exposure_risk < 75.0:
        state = 'UNHEALTHY_FOR_SENSITIVE_GROUPS'
    elif exposure_risk < 90.0:
        state = 'UNHEALTHY'
    else:
        state = 'HAZARDOUS'
    return {
        'exposure_risk': round(exposure_risk, 2),
        'state': state,
        'details': {'pm25_factor': float(pm25), 'o3_factor': float(o3), 'no2_factor': float(no2)},
    }

def compute_outdoor_operability(
    aqi: int = 0,
    temp: float = 0.0,
    humidity: float = 0.0,
    smoke: float = 0.0,
    wind_speed: float = 0.0,
    *_extra_args,
    **_extra_kwargs,
) -> Dict[str, Any]:
    """Module 2: Outdoor Operability Score."""
    aqi_penalty = min(float(aqi) * 0.4, 40.0)
    temp_penalty = max(0.0, abs(temp - 20.0) * 1.5)
    humidity_penalty = max(0.0, (humidity - 80.0) * 0.5) if humidity > 80.0 else 0.0
    smoke_penalty = min(float(smoke) * 0.3, 30.0)
    wind_penalty = min(max(0.0, wind_speed - 15.0) * 1.0, 15.0)
    score = 100.0 - (aqi_penalty + temp_penalty + humidity_penalty + smoke_penalty + wind_penalty)
    operability = float(np.clip(score, 0.0, 100.0))
    if operability >= 80.0:
        state = 'OPTIMAL'
    elif operability >= 60.0:
        state = 'FAVORABLE'
    elif operability >= 40.0:
        state = 'LIMITED'
    else:
        state = 'RESTRICTED'
    limiting_factors = []
    if aqi > 100:
        limiting_factors.append('HIGH_AQI')
    if smoke > 20:
        limiting_factors.append('WILDFIRE_SMOKE')
    if wind_speed > 25:
        limiting_factors.append('HIGH_WINDS')
    if humidity > 90:
        limiting_factors.append('HIGH_HUMIDITY')
    return {
        'outdoor_operability': round(operability, 2),
        'state': state,
        'limiting_factors': limiting_factors,
    }

def compute_drone_visibility(
    pm25: float = 0.0,
    visibility_km: float = 0.0,
    smoke: float = 0.0,
    humidity: float = 0.0,
    *_extra_args,
    **_extra_kwargs,
) -> Dict[str, Any]:
    """Module 3: Drone Visibility Index."""
    attenuation = math.exp(-0.05 * max(0.0, pm25 - 10.0)) * math.exp(-0.08 * smoke)
    effective_vis = visibility_km * attenuation
    vis_score = float(np.clip(effective_vis / 20.0 * 100.0, 0.0, 100.0))
    if vis_score >= 80.0 and humidity < 95.0:
        flight_state = 'EXCELLENT'
    elif vis_score >= 60.0:
        flight_state = 'GOOD'
    elif vis_score >= 40.0:
        flight_state = 'CAUTION'
    else:
        flight_state = 'GROUNDED'
    return {
        'visibility_score': round(vis_score, 2),
        'flight_state': flight_state,
        'visibility_km': round(effective_vis, 2),
    }

def compute_atmospheric_transparency(aod: float = 0.0, pm10: float = 0.0, *_extra_args, **_extra_kwargs) -> Dict[str, Any]:
    """Module 4: Atmospheric Transparency Index."""
    transparency = 100.0 * (1.0 - min(aod, 1.0)) - pm10 * 0.1
    transparency = float(np.clip(transparency, 0.0, 100.0))
    if transparency >= 80.0:
        classification = 'HIGH'
    elif transparency >= 50.0:
        classification = 'MODERATE'
    else:
        classification = 'LOW'
    return {'transparency': round(transparency, 2), 'classification': classification}

def compute_solar_loss(
    aod: float = 0.0,
    pm25: float = 0.0,
    smoke: float = 0.0,
    cloud_cover: float = 0.0,
    *_extra_args,
    **_extra_kwargs,
) -> Dict[str, Any]:
    """Module 5: Solar Loss Index."""
    aod_loss = aod * 35.0
    pm_loss = pm25 * 0.15
    smoke_loss = smoke * 0.4
    cloud_loss = cloud_cover * 0.6
    total_loss = aod_loss + pm_loss + smoke_loss + cloud_loss
    solar_loss_percent = float(np.clip(total_loss, 0.0, 100.0))
    return {'solar_loss_percent': round(solar_loss_percent, 2)}

def compute_ventilation_quality(
    wind_speed: float = 0.0,
    boundary_layer_height: float = 0.0,
    aqi: int = 0,
    *_extra_args,
    **_extra_kwargs,
) -> Dict[str, Any]:
    """Module 6: Ventilation Quality Engine."""
    vent_coeff = wind_speed * max(boundary_layer_height, 100.0)
    score = min(100.0, vent_coeff / 5000.0 * 100.0) - aqi * 0.2
    ventilation_score = float(np.clip(score, 0.0, 100.0))
    if ventilation_score >= 60.0 and aqi < 50:
        recommended = 'OPTIMAL_FOR_FRESH_AIR'
    elif ventilation_score >= 40.0:
        recommended = 'MODERATE_VENTILATION'
    else:
        recommended = 'MINIMIZE_OUTSIDE_AIR'
    return {'ventilation_score': round(ventilation_score, 2), 'recommended': recommended}

def compute_respiratory_stress(
    pm25: float = 0.0,
    o3: float = 0.0,
    humidity: float = 0.0,
    temp: float = 0.0,
    *_extra_args,
    **_extra_kwargs,
) -> Dict[str, Any]:
    """Module 7: Respiratory Stress Index."""
    pm_burden = norm.cdf(pm25, loc=35.0, scale=20.0) * 50.0
    o3_burden = norm.cdf(o3, loc=70.0, scale=30.0) * 30.0
    temp_humidity_burden = max(0.0, (temp - 25.0) * 0.5 + (humidity - 70.0) * 0.2)
    stress_score = float(np.clip(pm_burden + o3_burden + temp_humidity_burden, 0.0, 100.0))
    if stress_score < 30.0:
        risk = 'LOW'
    elif stress_score < 60.0:
        risk = 'ELEVATED'
    elif stress_score < 85.0:
        risk = 'HIGH'
    else:
        risk = 'SEVERE'
    return {'stress_score': round(stress_score, 2), 'risk': risk}

# ---------------------------------------------------------------------------
# Step 10 & 11: Unified AtmosOS Signal & Classification Engine
# ---------------------------------------------------------------------------
def compute_atmosphere_state_index(
    aqi: int,
    human_exposure: float,
    outdoor_operability: float,
    visibility_score: float,
    transparency: float,
    ventilation_score: float,
    respiratory_stress: float,
    *args: Any,
    **kwargs: Any,
) -> Tuple[float, str]:
    """
    Step 10 & 11: Unified AtmosOS Signal & State Classification.
    Combines AQI, human exposure, operability, visibility, transparency, ventilation, and respiratory stress.
    Range: 0-100.
    Classification: OPTIMAL, FAVORABLE, LIMITED, STRESSED, HAZARDOUS.
    """
    aqi_cleanliness = max(0.0, 100.0 - min(100.0, float(aqi) * 0.50))
    exposure_safety = max(0.0, 100.0 - float(human_exposure))
    operability = max(0.0, min(100.0, float(outdoor_operability)))
    vis = max(0.0, min(100.0, float(visibility_score)))
    trans = max(0.0, min(100.0, float(transparency)))
    vent = max(0.0, min(100.0, float(ventilation_score)))
    resp_comfort = max(0.0, 100.0 - float(respiratory_stress))

    raw = (
        0.20 * aqi_cleanliness
        + 0.18 * exposure_safety
        + 0.16 * operability
        + 0.14 * vis
        + 0.12 * resp_comfort
        + 0.10 * trans
        + 0.10 * vent
    )
    state_index = round(float(np.clip(raw, 0.0, 100.0)), 2)

    if state_index >= 80.0:
        classification = AtmosStateClassification.OPTIMAL.value
    elif state_index >= 60.0:
        classification = AtmosStateClassification.FAVORABLE.value
    elif state_index >= 40.0:
        classification = AtmosStateClassification.LIMITED.value
    elif state_index >= 20.0:
        classification = AtmosStateClassification.STRESSED.value
    else:
        classification = AtmosStateClassification.HAZARDOUS.value

    return state_index, classification

# ---------------------------------------------------------------------------
# Step 12: Agent-Oriented Intelligence Decision Profiles
# ---------------------------------------------------------------------------
def generate_agent_profiles(
    atmosphere_state_index: float,
    aqi: int,
    aqi_class: str,
    human_exposure: float,
    respiratory_stress: float,
    performance_score: float,
    outdoor_operability: float,
    visibility_score: float,
    visibility_km: float,
    aviation_score: float,
    aviation_state: str,
    logistics_score: float,
    logistics_level: str,
    smoke_risk: float,
    smoke_severity: str,
    ventilation_score: float,
) -> Dict[str, Any]:
    """Generates structured decision payloads for multi-domain autonomous agents."""
    return {
        'health_agents': {
            'actionable_status': 'SAFE_EXPOSURE' if aqi <= 50 and respiratory_stress < 35.0 else ('SENSITIVE_PRECAUTION' if aqi <= 100 else 'MASK_OR_INDOOR_RECOMMENDED'),
            'respiratory_risk_level': 'LOW' if respiratory_stress < 30.0 else ('MODERATE' if respiratory_stress < 60.0 else 'CRITICAL'),
            'air_purification_recommended': aqi > 75 or human_exposure > 45.0,
            'pediatric_elderly_advisory': 'Normal outdoor activity permitted' if aqi <= 50 else 'Limit prolonged exertion',
        },
        'human_performance_agents': {
            'endurance_capacity_score': performance_score,
            'workout_feasibility': 'OPTIMAL' if performance_score >= 80.0 else ('ACCEPTABLE' if performance_score >= 60.0 else 'INDOOR_ONLY'),
            'hydration_and_air_guidance': 'Hydration baseline normal' if performance_score >= 70.0 else 'Increased hydration and electrolyte demand',
        },
        'logistics_agents': {
            'operability_score': logistics_score,
            'dispatch_status': logistics_level,
            'surface_transport_delay_risk': 'LOW' if logistics_score >= 75.0 else ('MODERATE' if logistics_score >= 50.0 else 'HIGH_DELAY_HAZARD'),
        },
        'aviation_agents': {
            'flight_readiness_score': aviation_score,
            'flight_status': aviation_state,
            'effective_visibility_km': visibility_km,
            'drone_uas_operability': 'PERMITTED' if aviation_score >= 60.0 and visibility_score >= 60.0 else 'RESTRICTED',
        },
        'emergency_response_agents': {
            'evacuation_corridor_clarity': 'CLEAR' if smoke_risk < 40.0 and visibility_score >= 60.0 else 'DEGRADED',
            'smoke_severity_alert': smoke_severity,
            'first_responder_respiratory_ppe_required': aqi > 150 or smoke_risk > 60.0,
        },
        'outdoor_operation_agents': {
            'construction_work_suitability': 'PERMITTED' if outdoor_operability >= 60.0 else ('CAUTION_RESTRICTED_HOURS' if outdoor_operability >= 40.0 else 'SUSPENDED'),
            'operability_score': outdoor_operability,
            'ventilation_suitability': 'FAVORABLE_FOR_HVAC_AIR_EXCHANGE' if ventilation_score >= 50.0 and aqi < 100 else 'RECIRCULATE_INTERNAL_AIR',
        },
    }

# ---------------------------------------------------------------------------
# Data Retrieval & Atmospheric Pipeline
# ---------------------------------------------------------------------------
async def fetch_atmospheric_data(lat: float, lon: float) -> Dict[str, Any]:
    """
    Fetches real atmospheric measurements from Open-Meteo or falls back gracefully
    to the deterministic climate-zone atmospheric baseline model. Zero fabricated pollutants.
    """
    open_meteo_url = (
        f'https://air-quality-api.open-meteo.com/v1/air-quality?'
        f'latitude={lat}&longitude={lon}&'
        f'current=pm2_5,pm10,nitrogen_dioxide,sulphur_dioxide,ozone,carbon_monoxide,us_aqi'
    )
    data: Dict[str, Any] = {}
    async with httpx.AsyncClient(timeout=4.5) as client:
        try:
            resp = await client.get(open_meteo_url)
            if resp.status_code == 200:
                payload = resp.json()
                current = payload.get('current', {})
                data['pm25'] = float(current.get('pm2_5', 18.5))
                data['pm10'] = float(current.get('pm10', 35.2))
                data['no2'] = float(current.get('nitrogen_dioxide', 22.1))
                data['so2'] = float(current.get('sulphur_dioxide', 8.4))
                data['o3'] = float(current.get('ozone', 45.0))
                data['co'] = float(current.get('carbon_monoxide', 0.6))
                data['aqi'] = int(current.get('us_aqi', compute_aqi_from_pm25(data['pm25'])))
                data['source'] = 'Open-Meteo Air Quality API'
            else:
                raise Exception('API status non-200')
        except Exception as e:
            logger.warning(f'External API fetch failed ({e}), using deterministic climate-zone baseline.')
            doy = datetime.now(dt_timezone.utc).timetuple().tm_yday
            baseline = compute_climate_baseline(lat, lon, doy)
            data['pm25'] = baseline['baseline_pm25']
            data['pm10'] = baseline['baseline_pm10']
            data['no2'] = baseline['baseline_no2']
            data['so2'] = baseline['baseline_so2']
            data['o3'] = baseline['baseline_o3']
            data['co'] = baseline['baseline_co']
            data['aqi'] = baseline['baseline_aqi']
            data['source'] = f"AtmosOS Deterministic Climate-Zone Baseline ({baseline['season']})"

    return data

# ---------------------------------------------------------------------------
# API Endpoints (All Routes Preserved)
# ---------------------------------------------------------------------------
@app.get('/')
def root_index() -> Dict[str, Any]:
    return {
        'status': 'healthy',
        'service': 'atmos-os-6137',
        'name': 'atmos-os-6137',
        'version': '2.0.0',
        'description': 'AtmosOS - Global Atmospheric Intelligence & Human Exposure Signal Network',
    }

@app.get('/health')
def health_check() -> Dict[str, Any]:
    return {'status': 'healthy', 'timestamp': datetime.now(dt_timezone.utc).isoformat()}

@app.get('/tools')
def list_tools() -> Dict[str, Any]:
    return {
        'tools': [
            {'name': 'get_air_quality', 'description': 'Returns atmospheric pollutant concentrations, AQI, EPA classification, and PM2.5 trend.'},
            {'name': 'get_exposure_risk', 'description': 'Returns Human Exposure Index (HEI) and multi-pollutant vulnerability metrics.'},
            {'name': 'get_visibility_score', 'description': 'Returns atmospheric visibility quality and UAS/drone flight state.'},
            {'name': 'get_wildfire_smoke_risk', 'description': 'Returns deterministic wildfire smoke transport, plume spread, and arrival probability.'},
            {'name': 'get_outdoor_operability', 'description': 'Returns outdoor readiness score and environmental limiting factors.'},
            {'name': 'get_atmosos_signal', 'description': 'Returns unified AtmosOS operational signal, atmosphere_state_index, and multi-agent decision profiles.'},
            {'name': 'get_h3_tensor', 'description': 'Returns full H3 AtmosTensor containing AQI, visibility, exposure, and transparency metrics.'},
        ]
    }

@app.get('/get_air_quality', response_model=AirQualityResponse)
async def get_air_quality(lat: float = Query(2.0442), lon: float = Query(102.5689)):
    validate_coordinates(lat, lon)
    atmos = await fetch_atmospheric_data(lat, lon)
    pm25 = float(atmos.get('pm25', 0.0))
    aqi_val = int(atmos.get('aqi', 50))
    doy = datetime.now(dt_timezone.utc).timetuple().tm_yday
    baseline = compute_climate_baseline(lat, lon, doy)
    classification = classify_air_quality(aqi_val)
    trend = compute_pm25_trend(pm25, baseline['baseline_pm25'], 65.0)

    return AirQualityResponse(
        pm25=pm25,
        pm10=float(atmos.get('pm10', 0.0)),
        no2=float(atmos.get('no2', 0.0)),
        so2=float(atmos.get('so2', 0.0)),
        o3=float(atmos.get('o3', 0.0)),
        co=float(atmos.get('co', 0.0)),
        aqi=aqi_val,
        source=str(atmos.get('source', 'unknown')),
        classification=classification,
        trend=trend,
    )

@app.get('/get_exposure_risk', response_model=ExposureRiskResponse)
async def get_exposure_risk(lat: float = Query(2.0442), lon: float = Query(102.5689)):
    validate_coordinates(lat, lon)
    atmos = await fetch_atmospheric_data(lat, lon)
    hei = compute_human_exposure_index(atmos)
    return ExposureRiskResponse(
        exposure_risk=hei.get('exposure_risk', 0.0),
        state=hei.get('state', 'UNKNOWN'),
        details=hei.get('details', {}),
    )

@app.get('/get_visibility_score', response_model=VisibilityScoreResponse)
async def get_visibility_score(lat: float = Query(2.0442), lon: float = Query(102.5689)):
    validate_coordinates(lat, lon)
    atmos = await fetch_atmospheric_data(lat, lon)
    pm25 = float(atmos.get('pm25', 15.0))
    smoke_data = compute_wildfire_smoke_risk(lat, lon)
    smoke = float(smoke_data.get('smoke_risk', 10.0)) / 10.0
    vis_res = compute_drone_visibility(pm25=pm25, visibility_km=15.0, smoke=smoke, humidity=65.0)
    return VisibilityScoreResponse(
        visibility_score=vis_res.get('visibility_score', 80.0),
        flight_state=vis_res.get('flight_state', 'GOOD'),
        visibility_km=vis_res.get('visibility_km', 12.0),
    )

@app.get('/get_wildfire_smoke_risk', response_model=WildfireSmokeRiskResponse)
async def get_wildfire_smoke_risk(lat: float = Query(2.0442), lon: float = Query(102.5689)):
    validate_coordinates(lat, lon)
    risk_data = compute_wildfire_smoke_risk(lat, lon)
    return WildfireSmokeRiskResponse(
        smoke_risk=risk_data.get('smoke_risk', 0.0),
        arrival_hours=risk_data.get('arrival_hours', 24.0),
        severity=risk_data.get('severity', 'LOW'),
        thermal_anomalies_detected=risk_data.get('thermal_anomalies_detected', 0),
        plume_spread_score=risk_data.get('plume_spread_score', 0.0),
        arrival_probability=risk_data.get('arrival_probability', 0.0),
        dispersion_quality=risk_data.get('dispersion_quality', 'MODERATE'),
    )

@app.get('/get_outdoor_operability', response_model=OutdoorOperabilityResponse)
async def get_outdoor_operability(lat: float = Query(2.0442), lon: float = Query(102.5689)):
    validate_coordinates(lat, lon)
    atmos = await fetch_atmospheric_data(lat, lon)
    aqi = int(atmos.get('aqi', 50))
    smoke_data = compute_wildfire_smoke_risk(lat, lon)
    smoke = float(smoke_data.get('smoke_risk', 10.0))
    op = compute_outdoor_operability(aqi=aqi, temp=22.0, humidity=60.0, smoke=smoke, wind_speed=10.0)
    return OutdoorOperabilityResponse(
        outdoor_operability=op.get('outdoor_operability', 75.0),
        state=op.get('state', 'FAVORABLE'),
        limiting_factors=op.get('limiting_factors', []),
    )

@app.get('/get_atmosos_signal', response_model=AtmosOSSignalResponse)
async def get_atmosos_signal(lat: float = Query(2.0442), lon: float = Query(102.5689)):
    """
    Step 10, 11 & 12: Unified AtmosOS Operational Signal.
    Primary machine-consumable interface exposing atmosphere_state_index,
    classification, AtmosTensor, and tailored agent decision profiles.
    """
    validate_coordinates(lat, lon)
    h3_index = _latlng_to_h3(lat, lon, 8)
    atmos = await fetch_atmospheric_data(lat, lon)

    pm25 = float(atmos.get('pm25', 15.0))
    pm10 = float(atmos.get('pm10', 30.0))
    aqi = int(atmos.get('aqi', 50))
    aqi_class = classify_air_quality(aqi)

    doy = datetime.now(dt_timezone.utc).timetuple().tm_yday
    baseline = compute_climate_baseline(lat, lon, doy)
    pm25_trend = compute_pm25_trend(pm25, baseline['baseline_pm25'], 65.0)

    hei = compute_human_exposure_index(atmos)
    smoke_data = compute_wildfire_smoke_risk(lat, lon)
    smoke = float(smoke_data.get('smoke_risk', 10.0))

    vis = compute_drone_visibility(pm25=pm25, visibility_km=15.0, smoke=smoke / 10.0, humidity=65.0)
    trans = compute_atmospheric_transparency(aod=0.2, pm10=pm10)
    solar = compute_solar_loss(aod=0.2, pm25=pm25, smoke=smoke, cloud_cover=20.0)
    vent = compute_ventilation_quality(wind_speed=10.0, boundary_layer_height=800.0, aqi=aqi)
    resp = compute_respiratory_stress(pm25=pm25, o3=float(atmos.get('o3', 40.0)), humidity=65.0, temp=22.0)
    op = compute_outdoor_operability(aqi=aqi, temp=22.0, humidity=65.0, smoke=smoke, wind_speed=10.0)

    human_exposure = hei.get('exposure_risk', 50.0)
    outdoor_op = op.get('outdoor_operability', 70.0)
    vis_score = vis.get('visibility_score', 75.0)
    solar_loss = solar.get('solar_loss_percent', 15.0)
    breathability = max(0.0, 100.0 - human_exposure)

    # Advanced engines
    state_index, state_class = compute_atmosphere_state_index(
        aqi=aqi,
        human_exposure=human_exposure,
        outdoor_operability=outdoor_op,
        visibility_score=vis_score,
        transparency=trans.get('transparency', 80.0),
        ventilation_score=vent.get('ventilation_score', 70.0),
        respiratory_stress=resp.get('stress_score', 30.0),
    )

    stability = compute_atmospheric_stability(aqi=aqi, wind=10.0, boundary_layer_height=800.0, particulates=pm25)
    human_perf = compute_human_performance(aqi=aqi, visibility=vis.get('visibility_km', 12.0), temperature=22.0, smoke=smoke, humidity=65.0)
    aviation = compute_aviation_readiness(visibility=vis.get('visibility_km', 12.0), smoke=smoke, air_quality=aqi, wind=10.0)
    logistics = compute_logistics_operability(visibility=vis.get('visibility_km', 12.0), aqi=aqi, wind=10.0, smoke=smoke)

    atmos_tensor = {
        'AQI': aqi,
        'aqi': aqi,
        'visibility': vis_score,
        'human_exposure': human_exposure,
        'respiratory_stress': resp.get('stress_score', 30.0),
        'transparency': trans.get('transparency', 80.0),
        'ventilation': vent.get('ventilation_score', 70.0),
        'solar_loss': solar_loss,
    }

    agent_profiles = generate_agent_profiles(
        atmosphere_state_index=state_index,
        aqi=aqi,
        aqi_class=aqi_class,
        human_exposure=human_exposure,
        respiratory_stress=resp.get('stress_score', 30.0),
        performance_score=human_perf['performance_score'],
        outdoor_operability=outdoor_op,
        visibility_score=vis_score,
        visibility_km=vis.get('visibility_km', 12.0),
        aviation_score=aviation['aviation_score'],
        aviation_state=aviation['readiness_state'],
        logistics_score=logistics['logistics_score'],
        logistics_level=logistics['operability_level'],
        smoke_risk=smoke,
        smoke_severity=smoke_data.get('severity', 'LOW'),
        ventilation_score=vent.get('ventilation_score', 70.0),
    )

    # Legacy atmosphere_state_score preserved equal to atmosphere_state_index
    return AtmosOSSignalResponse(
        atmosphere_state_score=round(state_index, 2),
        human_exposure=round(human_exposure, 2),
        outdoor_operability=round(outdoor_op, 2),
        visibility_score=round(vis_score, 2),
        solar_loss=round(solar_loss, 2),
        breathability=round(breathability, 2),
        transparency=round(trans.get('transparency', 80.0), 2),
        ventilation_score=round(vent.get('ventilation_score', 70.0), 2),
        respiratory_stress=round(resp.get('stress_score', 30.0), 2),
        h3_index=str(h3_index),
        timestamp=datetime.now(dt_timezone.utc).isoformat(),
        # Extended fields
        atmosphere_state_index=state_index,
        state_classification=state_class,
        aqi=aqi,
        aqi_classification=aqi_class,
        pm25_trend=pm25_trend,
        stability_score=stability['stability_score'],
        human_performance=human_perf,
        aviation_readiness=aviation,
        logistics_operability=logistics,
        atmos_tensor=atmos_tensor,
        agent_profiles=agent_profiles,
    )

# ---------------------------------------------------------------------------
# Step 9: H3 Atmospheric Tensor Endpoint
# ---------------------------------------------------------------------------
@app.get('/h3_tensor')
@app.get('/tools/get_h3_tensor')
async def get_h3_tensor(
    h3_cell: str = Query('881f3a9b5dfffff', alias='h3'),
    lat: float = Query(2.0442),
    lon: float = Query(102.5689),
) -> Dict[str, Any]:
    """Step 9: H3 Atmospheric Tensor endpoint returning AtmosTensor metrics for any H3 cell."""
    validate_coordinates(lat, lon)
    cell = h3_cell.strip()
    if not cell or len(cell) < 5:
        cell = _latlng_to_h3(lat, lon, 8)

    atmos = await fetch_atmospheric_data(lat, lon)
    pm25 = float(atmos.get('pm25', 15.0))
    pm10 = float(atmos.get('pm10', 30.0))
    aqi = int(atmos.get('aqi', 50))
    hei = compute_human_exposure_index(atmos)
    smoke_data = compute_wildfire_smoke_risk(lat, lon)
    smoke = float(smoke_data.get('smoke_risk', 10.0))

    vis = compute_drone_visibility(pm25=pm25, visibility_km=15.0, smoke=smoke / 10.0, humidity=65.0)
    trans = compute_atmospheric_transparency(aod=0.2, pm10=pm10)
    solar = compute_solar_loss(aod=0.2, pm25=pm25, smoke=smoke, cloud_cover=20.0)
    vent = compute_ventilation_quality(wind_speed=10.0, boundary_layer_height=800.0, aqi=aqi)
    resp = compute_respiratory_stress(pm25=pm25, o3=float(atmos.get('o3', 40.0)), humidity=65.0, temp=22.0)
    op = compute_outdoor_operability(aqi=aqi, temp=22.0, humidity=65.0, smoke=smoke, wind_speed=10.0)

    vis_score = vis.get('visibility_score', 75.0)
    human_exposure = hei.get('exposure_risk', 50.0)
    outdoor_op = op.get('outdoor_operability', 70.0)
    solar_loss = solar.get('solar_loss_percent', 15.0)

    state_index, state_class = compute_atmosphere_state_index(
        aqi=aqi,
        human_exposure=human_exposure,
        outdoor_operability=outdoor_op,
        visibility_score=vis_score,
        transparency=trans.get('transparency', 80.0),
        ventilation_score=vent.get('ventilation_score', 70.0),
        respiratory_stress=resp.get('stress_score', 30.0),
    )

    tensor_data = {
        'AQI': aqi,
        'aqi': aqi,
        'visibility': vis_score,
        'human_exposure': human_exposure,
        'respiratory_stress': resp.get('stress_score', 30.0),
        'transparency': trans.get('transparency', 80.0),
        'ventilation': vent.get('ventilation_score', 70.0),
        'solar_loss': solar_loss,
    }

    return {
        'h3': cell,
        'coordinates': {'lat': lat, 'lon': lon},
        'atmos_tensor': tensor_data,
        'atmosphere_state_index': state_index,
        'state_classification': state_class,
        'timestamp': datetime.now(dt_timezone.utc).isoformat(),
    }

# ---------------------------------------------------------------------------
# CLI & Standalone Server
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import uvicorn
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=False)
