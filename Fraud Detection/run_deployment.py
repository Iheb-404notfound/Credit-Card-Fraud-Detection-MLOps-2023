from pipelines.deployment_pipeline import continuous_deployment_pipeline, inference_pipeline
from rich import print
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService
import click
from typing import cast

DEPLOY = 'deploy'
PREDICT = 'predict'
DEPLOY_AND_PREDICT = 'deploy_and_predict'
@click.command()
@click.option(
    '--config',
    '-c',
    type=click.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT]),
    default=DEPLOY_AND_PREDICT,
    help="You can chooose to train and deploy a model (`deploy`), or to"
    "(`predict`). By default will be run (`deploy_and_predict`)"
)
def run_deployment(config):
    path = './data/creditcard_2023.csv'
    
    print(f'[bold green] Starting to process your command [/bold green]')

    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    if config == DEPLOY:
        continuous_deployment_pipeline(path, workers=3, timeout=180)
    elif config == PREDICT:
        inference_pipeline()
    elif config == DEPLOY_AND_PREDICT:
        continuous_deployment_pipeline(path, workers=3, timeout=180)
        inference_pipeline()
    else:
        raise ValueError(f"Unknown config: {config}")
    
    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        model_name="model",
    )

    if existing_services:
        service = cast(MLFlowDeploymentService, existing_services[0])
        if service.is_running:
            print(
                f"The MLflow prediction server is running locally as a daemon "
                f"process service and accepts inference requests at:\n"
                f"    {service.prediction_url}\n"
                f"To stop the service, run "
                f"[italic green]`zenml model-deployer models delete "
                f"{str(service.uuid)}`[/italic green]."
            )
        elif service.is_failed:
            print(
                f"The MLflow prediction server is in a failed state:\n"
                f" Last state: '{service.status.state.value}'\n"
                f" Last error: '{service.status.last_error}'"
            )
    else:
        print(
            "No MLflow prediction server is currently running. The deployment "
            "pipeline must run first to train a model and deploy it. Execute "
            "the same command with the `--deploy` argument to deploy a model."
         )



if __name__ == '__main__':

    run_deployment()