"""Run the FastAPI service used by the Streamlit PathSnap interface."""
import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run("pathsnap.api:app", host=args.host, port=args.port, reload=False)

if __name__ == "__main__":
    main()
