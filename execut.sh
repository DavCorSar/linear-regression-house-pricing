#!/bin/bash
set -e
# BUILD
# This is a simple CI pipeline that first of all builds the socker image locally
echo Building the docker...
sudo docker build -t davcorsar/linear-regression-house-pricing:latest .

# TEST
# Then the image is executed to make sure that everything is correct
echo Testing the python script...
sudo docker run davcorsar/linear-regression-house-pricing

# RELEASE
# Finally, if there are no errors in the previous stages, the docker image is commited and pushed into the docker hub page
echo Pushing the script to DockerHub...
ID=$(sudo docker container ps -l -q)

sudo docker commit $ID davcorsar/linear-regression-house-pricing:latest

sudo docker push davcorsar/linear-regression-house-pricing:latest

echo Finish!

set +e
