from pipelines.training_pipeline import training_pipeline
import os
from zenml.client import Client
import logging as log

tracking_uri = Client().active_stack.experiment_tracker.get_tracking_uri()

if __name__ == '__main__':
    log.warning(f'Tracking URI: {tracking_uri}')
    training_pipeline('./data/creditcard_2023.csv')
    