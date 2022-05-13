# linear-regression-house-pricing
Python script for a linear regression to predict house prices

## How to use
Download all the files and execute the shell script execut.sh. This will automatically check the building process of the docker, then it will test the python script by running it and finally it will push the Docker image into a public repository in http://dockerhub.com.

After that you will be able to run the script for training and testing the Linear Regression model with the load_boston dataset, available at: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html. The arguments to pass to the script are the followings:

--mode: 'train' or 'predict', depending what do we want our model to do.  
--crim: A float representing the per capita crime rate.  
--zn: A float representing the proportion of residential land zoned for lots over 25000 sq. ft.  
--indus: A float representing the proportion of non-retail business acres per town.  
--chas: Binary variable, 1 if tract bounds river, 0 otherwise.  
--nox: Nitric oxides concentration (parts per 10 million) (float).  
--rm: Average number of rooms per dwelling.  
--age: Proportion of owner-occupied units built prior to 1940.  
--dis: Weighted distances to five Boston employment centres.  
--rad: An index of accessibility to radial highways.  
--tax: Full-value property-tax rate per $10,000.  
--ptratio: Pupil-teacher ratio by town.  
--lstat: Percentage of lower status of the population.  

To execute the script run:

sudo docker run -t davcorsar/linear-regression-house-pricing [arguments]
