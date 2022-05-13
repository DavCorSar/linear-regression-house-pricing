#!/bin/bash

sudo docker build -t py-regressor .

sudo docker run py-regressor

ID=$(sudo docker container ps -l -q)

sudo docker commit $ID py-regressor
