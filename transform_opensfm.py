import subprocess
import sys
import argparse
from pathlib import Path

def run_docker_and_script(volume_path):
    # Check if the container already exists
    container_exists_command = ["sudo", "docker", "ps", "-a", "--filter", "name=kapture_container", "--format", "{{.Names}}"]
    container_exists_result = subprocess.run(container_exists_command, check=True, stdout=subprocess.PIPE, universal_newlines=True)

    # If the container does not exist, create and start it
    if "kapture_container" not in container_exists_result.stdout:
        # Define the Docker run command
        docker_command = [
            "sudo", "docker", "run", "--name", "kapture_container", "-d", "--gpus", "all",
            "-v", f'{volume_path}:/data', "hub.newayz.com/pgv/kapture:1.1", "bash", "-c", "while true; do sleep 60; done"
        ]

        # Run the Docker container
        subprocess.run(docker_command, check=True)
    else:
        # If the container exists, start it
        subprocess.run(["sudo", "docker", "start", "kapture_container"], check=True)

    # Define the Python script command to run inside the Docker container
    python_script_command1 = [
        "sudo", "docker", "exec", "kapture_container",
        "python3", "/opt/src/kapture/tools/kapture_import_colmap.py",
        "-db", "/data/database.db", "-txt", "/data/0", "-im", "/data/images", "-o", "/data/kapture", "-v"
    ]

    # Run convert sfm to kapture
    subprocess.run(python_script_command1, check=True)

    # Define the second Python script command to run inside the Docker container
    python_script_command2 = [
        "sudo", "docker", "exec", "kapture_container",
        "python3", "/opt/src/kapture/tools/kapture_export_opensfm.py",
        "-k", "/data/kapture", "-o", "/data/opensfm", "-v"
    ]
    # Run convert kapture to openmvg
    subprocess.run(python_script_command2, check=True)

    # Stop and remove the container
    subprocess.run(["sudo", "docker", "stop", "kapture_container"], check=True)
    subprocess.run(["sudo", "docker", "rm", "kapture_container"], check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert SFM to OpenMVG format!")
    parser.add_argument("-path", type=Path, required=True, help="Path to the dataset")
    args = parser.parse_args()
    volume_path = args.path
    run_docker_and_script(volume_path)
