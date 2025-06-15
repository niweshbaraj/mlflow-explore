# Welcome to the Code Repository for MLFlow Exploration
Files created for exploring/testing mlflow

## Running MLflow [https://mlflow.org/docs/latest/ml/tracking/quickstart]

### 1. Get MLflow - Installing MLFlow Library

```bash
pip install mlflow

# or install a specific rc version
pip install mlflow==3.1.0rc0
```

### 2. Start a Tracking Server - locally

```bash
mlflow server --host 127.0.0.1 --port 8080    # Choose any port, provided that it's not already in use. If host not provided, it will run on localhost by default.
```

### 3. Using MLflow in project/file

```bash
import mlflow

mlflow.set_tracking_uri(uri="http://<host>:<port>")

# or particulary
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
```

### 4. Train a model and prepare metadata for logging

 - Load and prepare the Iris dataset for modeling.
 - Train a Logistic Regression model and evaluate its performance.
 - Prepare the model hyperparameters and calculate metrics for logging.
   

```bash
import mlflow
from mlflow.models import infer_signature

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Load the Iris dataset
X, y = datasets.load_iris(return_X_y=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the model hyperparameters
params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "multi_class": "auto",
    "random_state": 8888,
}

# Train the model
lr = LogisticRegression(**params)
lr.fit(X_train, y_train)

# Predict on the test set
y_pred = lr.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
```

### 5. Log the model and its metadata to MLflow

Now, we're going to use the model that we trained, the hyperparameters that we specified for the model's fit, and the loss metrics that were calculated by evaluating the model's performance on the test data to log to MLflow.

The steps that we will take are:

 - Initiate an MLflow **run** context to start a new run that we will log the model and metadata to.
 - **Log** model **parameters** and performance **metrics**.
 - **Tag** the run for easy retrieval.
 - **Register** the model in the MLflow Model Registry while **logging** (saving) the model.

```bash
# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Create a new MLflow Experiment
mlflow.set_experiment("MLflow Quickstart")

# Start an MLflow run
with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(params)

    # Log the loss metric
    mlflow.log_metric("accuracy", accuracy)

    # Infer the model signature
    signature = infer_signature(X_train, lr.predict(X_train))

    # Log the model, which inherits the parameters and metric
    model_info = mlflow.sklearn.log_model(
        sk_model=lr,
        name="iris_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="tracking-quickstart",
    )

    # Set a tag that we can use to remind ourselves what this model was for
    mlflow.set_logged_model_tags(
        model_info.model_id, {"Training Info": "Basic LR model for iris data"}
    )
```

### 6. View the Run and Model in the MLflow UI

In order to see the results of our run, we can navigate to the MLflow UI. Since we have already started the Tracking Server at http://localhost:8080, we can simply navigate to that URL in our browser.

When opening the site, you will see a screen similar to the following:

![image](https://github.com/user-attachments/assets/d004134f-9d26-44ab-aecf-61e27434870e)

Clicking on the name of the Experiment that we created ("MLflow Quickstart") will give us a list of runs associated with the Experiment. You should see a random name that has been generated for the run and nothing else show up in the Table list view to the right.

Clicking on the name of the run will take you to the Run page, where the details of what we've logged will be shown. The elements have been highlighted below to show how and where this data is recorded within the UI.

![image](https://github.com/user-attachments/assets/e1964d1a-6637-43e8-a0dc-0ed0545f5210)

Switch to the Models tab in the experiments page to view all the logged models under the Experiment, where you can see an entry for the logged model we just created ("iris_model").

![image](https://github.com/user-attachments/assets/ed4e6c79-017b-4201-b5d6-9cf074e94c2f)

Clicking on the name of the model will take you to the Logged Model page, with details on the logged model and its metadata.

![image](https://github.com/user-attachments/assets/cf56ca09-6d3e-4e73-9879-3506f6552869)


### Common Gotchas while running MLFlow in Jupyter environment:

1. Not running the logging code in the same cell in the notebook.
2. Partial logging or logging parameters before the model training.
3. Not running the server before hand and running the logging code inside your notebook.
4. Writing the parameter names by hand in a complex sklearn pipeline or column transformer object with multiple sub-parameters.



