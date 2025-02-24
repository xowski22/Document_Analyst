import subprocess
import time
import webbrowser
import os
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('launcher.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def verify_project_structure():
    "verify that all files exist"
    requested_files = [
        "api/main.py",
        "frontend/app.py",
        "src/models/summarizer.py",
        "src/models/qa_model.py",
        "src/utils/document_parser.py"
    ]

    for file in requested_files:
        if not os.path.exists(file):
            raise FileNotFoundError(f"File {file} not found")

def create_cache_dir():
    """create cache dir if it doesn't exist"""
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    logger.info("Cache dir verified/created")

def run_backend():
    """Run the FastAPI backend"""
    try:
        process = subprocess.Popen(
            ["uvicorn", "api.main:app", "--host", "localhost", "--port", "8000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        logger.info("Backend server started")
        return process
    except Exception as e:
        logger.error(f"Failed to start backend: {str(e)}")
        raise e

def run_frontend():
    """Run the StreamLit frontend"""
    try:
        process = subprocess.Popen(
            ["streamlit", "run", "frontend/app.py", "--server.port", "8501"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        logger.info("Frontend server started")
        return process
    except Exception as e:
        logger.error(f"Failed to start frontend: {str(e)}")
        raise e

def main():
    try:
        print("Starting Document Analyzer...")

        verify_project_structure()

        create_cache_dir()

        print("Starting backend server...")
        backend_process = run_backend()
        time.sleep(3)

        if backend_process.poll() is not None:
            raise Exception("Backend server failed to start.")

        print("Starting Frontend server...")
        frontend_process = run_frontend()
        time.sleep(3)

        if frontend_process.poll() is not None:
            backend_process.terminate()
            raise Exception("Frontend server failed to start.")

        # webbrowser.open("http://localhost:8501")

        print("Application is running!")
        print("Backend URL: http://localhost:8000")
        print("Frontend URL: http://localhost:8501")
        print("Press Ctrl+C to exit.")

        while True:
            time.sleep(1)

            if backend_process.poll() is not None:
                raise Exception("Backend server stopped unexpectedly.")
            if frontend_process.poll() is not None:
                raise Exception("Frontend server stopped unexpectedly.")
    except KeyboardInterrupt:
        print("\nStopping application...")

    except Exception as e:
        print(f"Error: {str(e)}")
        logger.error(f"Application error: {str(e)}")

    finally:
        try:
            backend_process.terminate()
            frontend_process.terminate()
            print("Application stopped!")
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

if __name__ == "__main__":
    main()