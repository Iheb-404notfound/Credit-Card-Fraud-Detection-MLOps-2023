from zenml import pipeline
from steps.divide_data import divide_data
from steps.train_model import train_model
from steps.evaluate_model import evaluate_model
from steps.clean_data import clean_data
from steps.fetch_data import fetch_data


@pipeline(enable_cache=True)
def training_pipeline(path: str):
    df = fetch_data(path)
    df = clean_data(df)
    train_X, train_y, test_X, test_y = divide_data(df)
    model = train_model(train_X, train_y)
    evaluate_model(model, test_X, test_y)
