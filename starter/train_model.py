# Script to train machine learning model.
import pandas as pd
import csv
import logging

from pathlib import Path
from starter.ml.model import train_model, save_model, inference, compute_model_metrics
from starter.ml.data import process_data, dataset_with_selected_column_value, trim_dataframe, \
    list_categorical_features
from sklearn.model_selection import train_test_split

logging.basicConfig(
    format='%(asctime)s:%(levelname)s:%(message)s',
    level=logging.INFO)
MODEL_DIR = Path(__file__).parent.parent.absolute() / 'model'
DATA_DIR = Path(__file__).parent.parent.absolute() / 'data'


def main():
    """
    Main function of module.
    """
    # Add code to load in the data.
    logging.info(f'Reading dataframe from {f"{DATA_DIR}/census.csv"}')
    data_df = pd.read_csv(f'{DATA_DIR}/census.csv')
    logging.info(f'Read dataframe, number of rows, {len(data_df)}')

    # Trim column names and data elements
    data_df = trim_dataframe(data_df)

    # Optional enhancement, use K-fold cross validation instead of a
    # train-test split.
    train, test = train_test_split(data_df, test_size=0.20, random_state=42)
    cat_features = list_categorical_features
    logging.info('Calling the process_data function, with training=True.')
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    logging.info('process_data function call completed')
    # Proces the test data with the process_data function.
    logging.info('Calling the process_data function, with training=False.')
    X_test, y_test, _, _ = process_data(test,
                                        categorical_features=cat_features,
                                        label="salary",
                                        training=False,
                                        encoder=encoder,
                                        lb=lb)
    logging.info('process_data function call completed')

    # Train and save a model.
    logging.info('Train the model.')
    model = train_model(X_train, y_train)
    logging.info('Model training completed')

    # Print model metric
    logging.info('Show model metric on test set.')
    pred = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, pred)
    print('precision, recall, fbeta: ', precision, recall, fbeta)

    # Save model
    logging.info('Save the model.')
    save_model(model, f"{MODEL_DIR}/model.pkl")
    logging.info(f'Model saved to {f"{MODEL_DIR}/model.pkl"}')

    # Save encoder and label binarizer
    logging.info('Saving encoder and label binarizer.')
    save_model(encoder, f"{MODEL_DIR}/encoder.pkl")
    save_model(lb, f"{MODEL_DIR}/lb.pkl")
    logging.info(f'Encoder and label binarizer saved to {f"{MODEL_DIR}"}')

    # compute performance on data slice
    logging.info('Running the create_metric_data function.')
    list_output = create_metric_data(input_df=test,
                                     column_name='education',
                                     categorical_features=cat_features,
                                     encoder=encoder,
                                     lb=lb,
                                     model=model)
    logging.info('create_metric_data function call completed.')

    # Save metric data
    logging.info(f'Saving metric to {f"{DATA_DIR}/slice_output.csv"}.')
    save_metric_data(list_output, f"{DATA_DIR}/slice_output.csv")


def save_metric_data(list_metric_data: list, file_path: str):
    """
    Saves metric data to provided filepath.
    """
    # Write CSV file
    with open(file_path, "wt") as fp:
        writer = csv.writer(fp, delimiter=",")
        writer.writerow(['column_name', 'column_value',
                        'precision', 'recall', 'fbeta'])
        writer.writerows(list_metric_data)


def create_metric_data(input_df: pd.DataFrame,
                       column_name: str,
                       categorical_features: list,
                       encoder,
                       lb,
                       model) -> list:
    """
    Creates the metric data as a list.
    """
    list_column_value = input_df[column_name].unique()
    list_output = []
    print('column_name', 'column_value', 'precision', 'recall', 'fbeta')
    for column_value in list_column_value:
        precision, recall, fbeta = compute_model_metrics_on_column_value(
            input_df, column_name, column_value, categorical_features, encoder, lb, model)
        list_output.append(
            [column_name, column_value, precision, recall, fbeta])
        print([column_name, column_value, precision, recall, fbeta])
    return list_output


def compute_model_metrics_on_column_value(input_df: pd.DataFrame,
                                          column_name: str,
                                          column_value: str,
                                          categorical_features: list,
                                          encoder,
                                          lb,
                                          model):
    """
    Computes model metrics for selected input column.
    """
    df = dataset_with_selected_column_value(
        input_df, column_name, column_value)

    # Proces the test data with the process_data function.
    X, y, _, _ = process_data(df,
                              categorical_features=categorical_features,
                              label='salary',
                              training=False,
                              encoder=encoder,
                              lb=lb)
    pred = inference(model, X)
    precision, recall, fbeta = compute_model_metrics(y, pred)

    return precision, recall, fbeta


if __name__ == '__main__':
    main()
