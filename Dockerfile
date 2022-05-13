# We start defining the base image
FROM jupyter/scipy-notebook

# Copy our local script into the new image
COPY main.py ./main.py

# Let's make a new directory
RUN mkdir my-model
# We define a global variable of our working directory
ENV MODEL_DIR=/home/jovyan/my-model
# We define global variables of where are going to be the two files that contain the necessary information to reproduce the training process
ENV STAND_DIR=standard.pkl
ENV REGRES_DIR=regressor.pkl

# We set our working directory
WORKDIR /home/jovyan

# Let's copy the file that contains the necessary package to be downloaded
COPY requeriments.txt ./requeriments.txt

# We install the necessary packages
RUN pip3 install -r requeriments.txt

# We define the command to run our script
ENTRYPOINT ["python", "main.py"]
