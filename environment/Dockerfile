# Use the Miniconda base image
FROM continuumio/miniconda3

# Set the working directory
WORKDIR /app

COPY . /app

COPY usrfile.txt /usr
COPY sensitive_info_root.txt /root

# Re-declare the argument so it is available during the build stage
ARG is_OCI=false

# Print the argument value (optional)
RUN echo "is_OCI is set to ${is_OCI}"

# Add conda initialization
RUN conda init bash


RUN chmod +x init.sh
RUN apt-get update -y
RUN apt-get install -y xxd wget curl
RUN apt-get install -y gcc
RUN apt-get install -y g++
RUN apt-get install default-jdk -y
RUN pip install flask
RUN pip install requests
RUN pip install pyyaml
RUN pip install jsonpickle
RUN pip install psutil
RUN pip install pyperclip
RUN pip install pillow
RUN pip install watchdog
RUN chmod +x start_processes.sh


# Before running conda, verify the environment file exists and show verbose progress
RUN if [ "$is_OCI" = "true" ]; then \
    echo "is_OCI is true"; \
    cp /app/environment.yml /app; \
    ls -l /app; \
    cat /app/environment.yml; \
    conda env create -f /app/environment.yml --verbose; \
    fi

ENTRYPOINT ["/app/init.sh"]
