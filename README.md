# Optimization-Industrial-Process-ADVANCED

Repo to show and example of end-to-end industrial project using machine learning models and optimization with gurobi. This repo is the continuation of the example of end-to-end that use Free Licence of Gurobi show in: https://github.com/joseortegalabra/Optimization-Industrial-Process. But now, more and more complex models were trained, so to run this codes it is necessary has a gurobi lic


### Abstract
The idea is solve a optimization of a industrial process using gurobi and machine learning models. 
The industrial process is divided into 3 small subprocess where the output of one subprocess is the input of another subprocess.
Each output of subprocess could be modeladed as a Machine Learning Model

The objetive is that the output of the final subprocess (output of the process) needs to be controled into a range of optimal operation changing the values of differents features in the differents subprocess. Also, in addition to have the output of the process in control it needs to be it with the minimal costs

### List models
The list of models trained and used in optimization models are divided into:
- d0eop_microkappa
- d0eop_blancura
- d1_brillo
- p_blancura


### Improves vs base codes
This repo has the following improves (complexity increment) againts the base codes (that you can find in this repo: https://github.com/joseortegalabra/Optimization-Industrial-Process)

- **Adding models** (In the base end-to-end project 3 models are development and including into optimization engine. In this notebook at least there are 4 models). Important: Adding models in the same stage and share some features NC and features C. So, this is a little problem to thinking and solve.

- **Include transformations in the pipeline of models** (standarscaler, polynomial features, operations in columns). Interesting transformations that are not supported: minmaxscaler, custom function in columns operations.

- **Training more complex models** (Ensembles models: bagging, boosting, etc)

- **Using Gurobi Licence**: Using gurobi licence to include all the models trained

- **Train piecewise models**: Not exactly but following this idea. Split the data according one important feature and train differents models for each segment generated. The idea is have a better global model with the split of the data because a difference in the distribution of the data had identified.

- **Update optimization engine**: first, update optimization engine with the updates in the training models: using piecewise models, more complex models, transformations in columns, operations in columns (the improves defined previosly)


- FALTA: Parametrize and automatization of the codes of Gurobi Optimizer (creation of decision var, constraints, etc)

- FALTA: Relajar restricciones


### Folder
- **0_templates**: folder that contains a templates. Template how to load a licence gurobi using differents ways and a template notebook with the first lines of codes that can be used in all the folders (see that the folder artifacts, config, etc there are located in the root path and the notebooks are in folders) 

- **1_data**: folder where the datalake data (almost raw data) is downloaded from a datalake in bigquert (GCP)

- **2_basic_process_data**: folder where the data getting form the datalake is transformed in a version of data that a data scientist can manipulate (pivot data, delete hidden duplicates, etc)

- **3_eda**: folder that contains codes for simple EDA for each model tranined (correlations) and select the features to use to train the models

- **4_modeling_ml**: train the ml models with models suported by gurobi

- **5_offline_evaluation_ml**: evaluate ml models (offline evaluation)

- **6_optimization**: join the ml models with the optimization model. The constraints are:
    - the 3 ml models deffined as constraints
    - delta of decision variables. Delta of decision variables that are features machine learning models. Delta between the base value and the optimal value because there are physical constraints that these features can't increment too much

- **7_offline_evaluation_optimization**: evaluate gurobi optimization engine (offline evaluation)

- **8_app_streamlit**: app in streamlit to test the resolution of gurobi ml

- **artifacts**: folder that contains artifacts as data, ml models, etc

- **config**: folder that contains configuration file. For example, list of features, classification of the features, operational limits, upper and lower bound of decision variables in optimization, etc