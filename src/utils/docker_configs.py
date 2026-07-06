import docker
import subprocess
import time
import threading
import tempfile
import os

# Repo-relative path to the environment/ Docker build context.
_ENV_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "environment",
)

def write_code_to_container(container_name, code_snippet, file_path="/tmp/representative_code.py"):
    """
    Write code_snippet to a temporary file and copy it to the given path in the target Docker container.
    Supports concurrent multi-process use; the code-type suffix is derived automatically from file_path.
    """
    # Derive the suffix automatically from the target file path (e.g., .py, .cpp)
    suffix = os.path.splitext(file_path)[1] or ".tmp"

    # Create a unique temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False, dir=os.path.join(_ENV_DIR, "docker_files")) as temp_file:
        temp_file.write(code_snippet)
        temp_path = temp_file.name  # Get the full path

    try:
        # Run docker cp to copy the file into the container
        subprocess.run(["docker", "cp", temp_path, f"{container_name}:{file_path}"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to copy file to Docker container {container_name}: {e}")
        raise
    finally:
        # Remove the temporary file to avoid clutter
        if os.path.exists(temp_path):
            os.remove(temp_path)

def execute_command_in_container(container_name, command, timeout=10):
    """
    Execute a command in the specified container and return the result; return "Execution Time Out" on timeout.

    :param container_name: the container name
    :param command: the command to execute
    :param timeout: the timeout in seconds
    :return: the command output or an error message
    """
    client = docker.from_env()

    def run_command():
        """ Execute the command in the container and update the output. """
        nonlocal result
        try:
            # Get the container object
            container = client.containers.get(container_name)
            print(f'Found container {container_name} with ID: {container.id}')

            # Check whether the container is running
            if container.status != 'running':
                result = f"Error: Container '{container_name}' is not running."
                return

            # Execute the command in the container; use stream=True to avoid blocking
            exec_result = container.exec_run(command, tty=False, stream=True)

            # Process the execution result
            output = ''.join([line.decode('utf-8').strip() for line in exec_result.output])

            if output:
                result = output
            else:
                result = ""
        except docker.errors.NotFound:
            result = f"Error: Docker container '{container_name}' not found."
        except docker.errors.ContainerError as e:
            result = f"Container execution failed: {e}"
        except docker.errors.APIError as e:
            result = f"Error while executing command in Docker container '{container_name}': {e}"
        except Exception as e:
            result = f"Unexpected error: {e}"

    # Holds the command execution result
    result = None

    # Create and start a thread to run the command
    thread = threading.Thread(target=run_command)
    thread.start()

    # Wait for the thread with a timeout
    thread.join(timeout)

    # If it timed out, return Execution Time Out
    if thread.is_alive():
        return "Execution Time Out"

    return result


def delete_docker(docker_name):
    client = docker.from_env()

    try:
        # Get the container
        container = client.containers.get(docker_name)
        print(f'Found container {docker_name} with ID: {container.id}')

        # Stop the container
        print(f'Stopping container {docker_name}...')
        container.stop()

        # Remove the container
        print(f'Removing container {docker_name}...')
        container.remove()
        
        print(f'Docker container {docker_name} successfully removed.')

    except docker.errors.NotFound:
        print(f"Error: Docker container '{docker_name}' not found.")
    except docker.errors.APIError as e:
        print(f"Error while removing Docker container '{docker_name}': {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

def create_docker(docker_name):
    client = docker.from_env()

    # Check if the image already exists
    image_exists = False
    try:
        client.images.get('red-teaming-code-agent')
        image_exists = True
        print("Image 'red-teaming-code-agent' already exists. Skipping build.")
    except docker.errors.ImageNotFound:
        print("Image 'red-teaming-code-agent' not found. Proceeding with build.")

    # Build the Docker image from the Dockerfile if it doesn't exist
    if not image_exists:
        try:
            build_path = _ENV_DIR

            # Check if folder exists to avoid misleading errors
            if not os.path.isdir(build_path):
                print(f"❌ Docker build directory not found: {build_path}")
                return None

            image, build_logs = client.images.build(path=build_path, tag='red-teaming-code-agent')
            for log in build_logs:
                if 'stream' in log:
                    print(log['stream'].strip())
        except Exception as e:
            print(f"Error building Docker image: {e}")
            return None

    # Run the Docker container with the specified parameters
    try:
        container = client.containers.run(
            'red-teaming-code-agent',
            name=docker_name,
            detach=True,
            tty=True,
            command="bash"
        )
        print(f"✅ Docker container {container.name} created and running with ID: {container.id}")
        return container

    except Exception as e:
        print(f"Error running Docker container: {e}")
        return None
