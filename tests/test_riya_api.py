from fastapi.testclient import TestClient

from backend.api.server import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/api/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ONLINE"
    assert "solver" in data
    assert "note" in data


def test_model_stats_endpoint():
    response = client.get("/api/model-stats")

    assert response.status_code == 200

    data = response.json()

    assert "model" in data
    assert "presolver" in data
    assert "objective" in data

    model = data["model"]

    assert model["variables"] > 0
    assert model["constraints"] > 0
    assert model["non_zeros"] > 0


def test_architecture_endpoint():
    response = client.get("/api/architecture")

    assert response.status_code == 200

    data = response.json()

    assert "stages" in data
    assert len(data["stages"]) > 0

    stage_ids = [stage["id"] for stage in data["stages"]]

    assert "opt_model" in stage_ids
    assert "presolver" in stage_ids
    assert "indioptima" in stage_ids
    assert "backend" in stage_ids
    assert "solution" in stage_ids