# tests/test_api.py

import pytest
from httpx import AsyncClient
from app.main import app

@pytest.mark.asyncio
async def test_analyze_text():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/analyze", json={"text": "This product is awesome, I love it!"})
    assert response.status_code == 200
    data = response.json()
    assert data["type"] == "text"
    assert "sentiment_analysis" in data["result"]

@pytest.mark.asyncio
async def test_analyze_no_input():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/analyze")
    assert response.status_code == 400
    assert "error" in response.json()
