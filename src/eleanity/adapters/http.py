from __future__ import annotations

from typing import Any

import httpx


def post_json(url: str, payload: dict[str, Any], *, timeout: float = 30.0) -> dict[str, Any]:
    """POST JSON to an OpenAI-compatible or custom inference endpoint."""

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, json=payload, headers={"Content-Type": "application/json"})
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as error:
        raise RuntimeError(str(error)) from error
    except ValueError as error:
        raise RuntimeError(f"invalid JSON from {url}: {error}") from error
    if not isinstance(data, dict):
        raise RuntimeError("endpoint returned a non-object JSON response")
    return data


def get_json(url: str, *, timeout: float = 10.0) -> dict[str, Any]:
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.get(url)
            response.raise_for_status()
            data = response.json()
    except httpx.HTTPError as error:
        raise RuntimeError(str(error)) from error
    except ValueError as error:
        raise RuntimeError(f"invalid JSON from {url}: {error}") from error
    if not isinstance(data, dict):
        raise RuntimeError("endpoint returned a non-object JSON response")
    return data
