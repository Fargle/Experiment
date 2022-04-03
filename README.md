# Objective
A project that attempts to abstract away the minutia of running data science experiments. From developing a model to interpreting the results, the time it takes to get results that are both reliable and interpretable can be a long and tedious process. This project attempts to create a user interface with your data and model to offer a wide variety of tools that will help you understand your model and your data without the need to write any code.

# Setup
* Install python 3.6.2, and create a virtual environment with `python -m venv <name of env>`.
* Activate your environment and upgrade pip: `python -m pip install --upgrade pip`.
* Install required packages, run `pip install -e <path to setup.py>`.

# Example_exp
* Included is an example experiment already set up with the required fields. You will find the full data set, marked Iris.csv, along with the training and testing data.

# Setting up Your own Experiments
### Inside a the same directory, the following is required
* pretrained model or model_config.
* train_X, train_y, test_X, test_y

# Use
* `python experiment.py -exp <path to experiment> `
* `python analysis.py -exp <path to experiment>`
* `streamlit run analysis.py -- -exp <path to experiment>` for use of front end app. 
