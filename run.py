"""run.py

Startup script for the project. Binds to the port provided by the environment
(`PORT`, `RENDER_PORT`, or `RENDER_INTERNAL_PORT`) so Render.com and similar
platforms can control the listening port. Tries to start an ASGI app using
uvicorn (common for FastAPI), then falls back to running a Flask `app` if
present in `app.py`.
"""
import os
import sys


def get_port(default: int = 8000) -> int:
    """Return port from environment used by Render/other hosts, with fallback."""
    for name in ("PORT", "RENDER_PORT", "RENDER_INTERNAL_PORT"):
        val = os.environ.get(name)
        if val:
            try:
                return int(val)
            except ValueError:
                print(f"WARNING: env {name}={val!r} is not an int, ignoring")
    return default


def start_uvicorn(app_target: str, host: str, port: int) -> None:
    import uvicorn

    # uvicorn.run accepts either an app instance or a "module:app" string
    print(f"Starting ASGI app '{app_target}' on {host}:{port}")
    uvicorn.run(app_target, host=host, port=port)


def main() -> None:
    host = "0.0.0.0"
    port = get_port()

    # Try ASGI (FastAPI / Starlette / Responder) first
    try:
        # Heuristic candidates to look for an app object
        candidates = ["api:app"]
        for cand in candidates:
            module_name = cand.split(":")[0]
            try:
                __import__(module_name)
            except Exception:
                continue

            # If import succeeded try to start with uvicorn
            try:
                start_uvicorn(cand, host, port)
                return
            except Exception as e:
                print(f"uvicorn start attempt for {cand} failed: {e}")
    except Exception as e:  # defensive
        print(f"ASGI startup check failed: {e}")

    # Fallback: try Flask app in app.py
    try:
        from app import app as flask_app

        print(f"Starting Flask app on {host}:{port}")
        flask_app.run(host=host, port=port)
        return
    except Exception as e:
        print(f"Flask startup failed or `app` not found in app.py: {e}")

    print("ERROR: No runnable `app` found. Ensure `api.py` or `app.py` defines `app`.")
    sys.exit(1)


if __name__ == "__main__":
    main()
