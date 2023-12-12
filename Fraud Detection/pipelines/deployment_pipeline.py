import numpy as np
import pandas as pd
from zenml import pipeline, step
from zenml.config import DockerSettings

from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.integrations.constants import MLFLOW
from zenml.steps import BaseParameters, Output
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService


from steps.clean_data import clean_data
from steps.divide_data import divide_data
from steps.evaluate_model import evaluate_model
from steps.fetch_data import fetch_data
from steps.train_model import train_model

docker = DockerSettings(required_integrations=[MLFLOW])


@pipeline(enable_cache=True, settings={'docker': docker})
def continuous_deployment_pipeline( path, workers: int = 1, timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT):
    """Pipeline to train and deploy a model to MLFlow."""
    df = fetch_data(path)
    df = clean_data(df)
    train_X, train_y, test_X, test_y = divide_data(df)
    model = train_model(train_X, train_y, to_load=False)
    results = evaluate_model(model, test_X, test_y)
    mlflow_model_deployer_step(
        model = model, 
        workers = workers, 
        timeout = timeout
    )

@pipeline(settings={'docker': docker})
def inference_pipeline(
    pipeline_name: str = 'continuous_deployment_pipeline',
    pipeline_step_name: str = 'mlflow_model_deployer_step'
):
    """Pipeline to run inference against a deployed model."""
    mlflow_model_deployer_step(
        pipeline_name = pipeline_name,
        pipeline_step_name = pipeline_step_name
    )