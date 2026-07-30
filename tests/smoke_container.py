"""Verify a running container answers the way the source tree does.

The rest of the suite exercises the app in-process, which proves the code is
right but says nothing about the image around it. A container can build cleanly
and still fail to serve: the artifact may not have been packaged, a runtime
dependency may have been left in the dev extra, or the unprivileged user may not
be able to read what it needs. None of that shows up until the process starts.

Deliberately stdlib-only. This runs on the CI runner against the published port,
so requiring pandas and scikit-learn here would mean installing the whole
dependency tree a second time purely to send one HTTP request.

Usage:
    python tests/smoke_container.py [--base-url http://localhost:8000]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

#: The same fixed input the artifact contract test pins, expressed on the wire.
#: test_model_contract.py asserts these two constants still agree with the
#: model, so a retrain cannot leave this file quietly out of date.
PAYLOAD = {
    "gold_diff": 2400,
    "exp_diff": 1100,
    "champ_level_diff": 0.6,
    "is_first_tower": False,
    "is_first_blood": True,
    "killed_fire_drake": 1,
    "killed_water_drake": 0,
    "killed_air_drake": 0,
    "killed_earth_drake": 0,
    "lost_fire_drake": 0,
    "lost_water_drake": 0,
    "lost_air_drake": 0,
    "lost_earth_drake": 0,
    "killed_rift_herald": 1,
    "lost_rift_herald": 0,
    "destroyed_top_inner_turret": 0,
    "destroyed_mid_inner_turret": 0,
    "destroyed_bot_inner_turret": 0,
    "lost_top_inner_turret": 0,
    "lost_mid_inner_turret": 0,
    "lost_bot_inner_turret": 0,
    "destroyed_top_outer_turret": 0,
    "destroyed_mid_outer_turret": 0,
    "destroyed_bot_outer_turret": 0,
    "lost_top_outer_turret": 0,
    "lost_mid_outer_turret": 0,
    "lost_bot_outer_turret": 0,
    "kills": 6,
    "deaths": 3,
    "assists": 8,
    "wards_placed": 18,
    "wards_destroyed": 4,
    "wards_lost": 5,
}

EXPECTED_PROBABILITY = 0.8504427853741403
EXPECTED_N_FEATURES = 33

#: Floating point crosses a JSON boundary here, so an exact match is the wrong
#: assertion. This is still tight enough that a different set of weights fails.
TOLERANCE = 1e-9

#: The image declares a 10s start period for its own health check, which is the
#: honest estimate of how long loading the artifact takes. Allow well past that
#: so a slow runner reads as slow rather than as broken.
STARTUP_TIMEOUT_SECONDS = 60
POLL_INTERVAL_SECONDS = 2


def get_json(url: str, payload: dict | None = None, timeout: float = 10.0) -> dict:
    """Send a request and decode the JSON body."""
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"content-type": "application/json"} if data else {}
    request = urllib.request.Request(url, data=data, headers=headers)

    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def wait_for_health(base_url: str) -> dict:
    """Poll /health until the container reports a loaded model.

    Raises:
        TimeoutError: if the container never becomes ready, which is a failure
            worth surfacing as itself rather than as a confusing refused
            connection from the first prediction.
    """
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    last_error = "no response"

    while time.monotonic() < deadline:
        try:
            body = get_json(f"{base_url}/health")
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            last_error = str(exc)
        else:
            if body.get("model_loaded"):
                return body
            last_error = f"reported {body}"

        time.sleep(POLL_INTERVAL_SECONDS)

    raise TimeoutError(
        f"container not ready after {STARTUP_TIMEOUT_SECONDS}s: {last_error}"
    )


def check_health(base_url: str) -> None:
    body = wait_for_health(base_url)

    if body.get("status") != "ok":
        raise AssertionError(f"expected status ok, got {body.get('status')!r}")

    if body.get("n_features") != EXPECTED_N_FEATURES:
        raise AssertionError(
            f"expected {EXPECTED_N_FEATURES} features, got {body.get('n_features')}"
        )

    print(f"  /health   ok, model loaded, {body['n_features']} features")


def check_prediction(base_url: str) -> None:
    """Assert the containerised model returns the probability it should.

    This is the assertion that makes the job worth running: it proves the image
    is serving the same weights the repository pinned, not merely that something
    is listening on the port.
    """
    body = get_json(f"{base_url}/predict", PAYLOAD)
    probability = body["win_probability"]

    if abs(probability - EXPECTED_PROBABILITY) > TOLERANCE:
        raise AssertionError(
            f"expected win_probability {EXPECTED_PROBABILITY!r}, "
            f"got {probability!r}. The image is not serving the pinned artifact."
        )

    if body["will_win"] is not True:
        raise AssertionError(f"expected will_win true, got {body['will_win']!r}")

    print(f"  /predict  ok, win_probability {probability}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    print(f"smoke testing {base_url}")

    try:
        check_health(base_url)
        check_prediction(base_url)
    except (AssertionError, TimeoutError, urllib.error.HTTPError) as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    print("container smoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
