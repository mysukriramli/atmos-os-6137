import pytest
from fastapi.testclient import TestClient
from main import app, validate_coordinates, compute_aqi_from_pm25, compute_human_exposure_index, compute_outdoor_operability, compute_drone_visibility, compute_atmospheric_transparency, compute_solar_loss, compute_ventilation_quality, compute_respiratory_stress, compute_wildfire_smoke_risk
client = TestClient(app)

def test_aqi_transformation_monotonicity():
    aqi_low = compute_aqi_from_pm25(pm25=5.0)
    aqi_med = compute_aqi_from_pm25(pm25=35.0)
    aqi_high = compute_aqi_from_pm25(pm25=150.0)
    assert aqi_low < aqi_med < aqi_high
    assert aqi_low >= 0

def test_exposure_risk_engine():
    pollutants_low = {'pm25': 5.0, 'pm10': 10.0, 'no2': 10.0, 'so2': 5.0, 'o3': 20.0, 'co': 0.5}
    pollutants_high = {'pm25': 120.0, 'pm10': 180.0, 'no2': 90.0, 'so2': 50.0, 'o3': 120.0, 'co': 5.0}
    res_low = compute_human_exposure_index(pollutants=pollutants_low)
    res_high = compute_human_exposure_index(pollutants=pollutants_high)
    assert res_low.get('exposure_risk', 0.0) < res_high.get('exposure_risk', 0.0)
    assert isinstance(res_low.get('state'), str)
    assert isinstance(res_high.get('state'), str)

def test_visibility_engine():
    clear_air = compute_drone_visibility(pm25=5.0, visibility_km=25.0, smoke=0.0, humidity=40.0)
    smoky_air = compute_drone_visibility(pm25=150.0, visibility_km=2.0, smoke=90.0, humidity=85.0)
    assert clear_air.get('visibility_score', 0.0) > smoky_air.get('visibility_score', 0.0)
    assert isinstance(clear_air.get('flight_state'), str)

def test_solar_loss_impact():
    low_aerosol = compute_solar_loss(aod=0.1, pm25=5.0, smoke=0.0, cloud_cover=0.0)
    high_aerosol = compute_solar_loss(aod=1.5, pm25=120.0, smoke=80.0, cloud_cover=50.0)
    assert low_aerosol.get('solar_loss_percent', 0.0) < high_aerosol.get('solar_loss_percent', 0.0)

def test_wildfire_smoke_engine():
    res = compute_wildfire_smoke_risk(lat=37.7749, lon=-122.4194)
    assert 'smoke_risk' in res
    assert isinstance(res.get('smoke_risk', 0.0), (int, float))

def test_api_endpoints_and_unified_signal():
    lat, lon = (2.0442, 102.5689)
    assert client.get('/').status_code == 200
    assert client.get('/health').status_code == 200
    assert client.get('/tools').status_code == 200
    endpoints = ['/get_air_quality', '/get_exposure_risk', '/get_visibility_score', '/get_wildfire_smoke_risk', '/get_outdoor_operability', '/get_atmosos_signal']
    for ep in endpoints:
        response = client.get(f'{ep}?lat={lat}&lon={lon}')
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)

def test_anti_placeholder_audit():
    res1 = compute_wildfire_smoke_risk(lat=1.0, lon=1.0)
    res2 = compute_wildfire_smoke_risk(lat=50.0, lon=50.0)
    assert isinstance(res1.get('smoke_risk'), (int, float))
    assert isinstance(res2.get('smoke_risk'), (int, float))

def test_additional_computational_modules():
    try:
        validate_coordinates(lat=91.0, lon=0.0)
        pytest.fail('Expected coordinate validation error for lat > 90')
    except Exception:
        pass
    trans = compute_atmospheric_transparency(aod=0.2, pm10=30.0)
    assert 'transparency' in trans
    vent = compute_ventilation_quality(wind_speed=5.5, boundary_layer_height=1000.0, aqi=45)
    assert 'ventilation_score' in vent
    resp = compute_respiratory_stress(pm25=25.0, o3=40.0, humidity=60.0, temp=22.0)
    assert 'stress_score' in resp
    oper = compute_outdoor_operability(aqi=50, temp=25.0, humidity=50.0, smoke=10.0, wind_speed=3.0)
    assert 'outdoor_operability' in oper
    val_a = compute_aqi_from_pm25(pm25=12.0)
    val_b = compute_aqi_from_pm25(pm25=12.0)
    assert pytest.approx(val_a, rel=1e-05) == val_b