# Script to train machine learning model.
import pandas as pd
import csv
import logging

from pathlib import Path
from starter.ml import model, data
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
    data_df = data.trim_dataframe(data_df)

    # Optional enhancement, use K-fold cross validation instead of a
    # train-test split.
    train, test = train_test_split(data_df, test_size=0.20, random_state=42)
    cat_features = data.list_categorical_features
    logging.info('Calling the process_data function, with training=True.')
    X_train, y_train, encoder, lb = data.process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    logging.info('process_data function call completed')
    # Proces the test data with the process_data function.
    logging.info('Calling the process_data function, with training=False.')
    X_test, y_test, _, _ = data.process_data(test,
                                             categorical_features=cat_features,
                                             label="salary",
                                             training=False,
                                             encoder=encoder,
                                             lb=lb)
    logging.info('process_data function call completed')

    # Train and save a model.
    logging.info('Train the model.')
    classifier = model.train_model(X_train, y_train)
    logging.info('Model training completed')

    # Print model metric
    logging.info('Show model metric on test set.')
    pred = model.inference(classifier, X_test)
    precision, recall, fbeta = model.compute_model_metrics(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    print('precision, recall, fbeta, accuracy: ',
          precision, recall, fbeta, accuracy)

    # Save model
    logging.info('Save the model.')
    model.save_model(classifier, f"{MODEL_DIR}/model.pkl")
    logging.info(f'Model saved to {f"{MODEL_DIR}/model.pkl"}')

    # Save encoder and label binarizer
    logging.info('Saving encoder and label binarizer.')
    model.save_model(encoder, f"{MODEL_DIR}/encoder.pkl")
    model.save_model(lb, f"{MODEL_DIR}/lb.pkl")
    logging.info(f'Encoder and label binarizer saved to {f"{MODEL_DIR}"}')

    # compute performance on data slice
    logging.info('Running the create_metric_data function.')
    list_output = create_metric_data(input_df=test,
                                     column_name='race',
                                     categorical_features=cat_features,
                                     encoder=encoder,
                                     lb=lb,
                                     classifier=classifier)
    logging.info('create_metric_data function call completed.')

    # Save metric data
    logging.info(f'Saving metric to {f"{DATA_DIR}/slice_output.csv"}.')
    save_metric_data(list_output, f"{DATA_DIR}/slice_output.csv")


def save_metric_data(list_row_data: list, file_path: str):
    """
    Saves metric data to provided filepath.
    """
    # Write CSV file
    with open(file_path, "wt") as fp:
        writer = csv.writer(fp, delimiter=",")
        writer.writerow(data.list_header + list(model.dict_metric.keys()))
        writer.writerows(list_row_data)


def create_metric_data(input_df: pd.DataFrame,
                       column_name: str,
                       categorical_features: list,
                       encoder,
                       lb,
                       classifier) -> list:
    """
    Creates the metric data as a list.
    """
    list_column_values = input_df[column_name].unique()
    list_output = []
    print(data.list_header + list(model.dict_metric.keys()))
    for column_value in list_column_values:
        list_metric = compute_model_metrics_on_column_value(
            input_df,
            column_name,
            column_value,
            categorical_features,
            encoder,
            lb,
            classifier,
            model.dict_metric)
        list_output.append([column_name, column_value] + list_metric)
        print(f'{column_name}, {column_value}, ', list_metric)
    return list_output


def compute_model_metrics_on_column_value(input_df: pd.DataFrame,
                                          column_name: str,
                                          column_value: str,
                                          categorical_features: list,
                                          encoder,
                                          lb,
                                          classifier,
                                          dict_metric):
    """
    Computes model metrics for selected input column.
    """
    df = data.dataset_with_selected_column_value(
        input_df, column_name, column_value)

    # Proces the test data with the process_data function.
    X, y, _, _ = data.process_data(df,
                                   categorical_features=categorical_features,
                                   label='salary',
                                   training=False,
                                   encoder=encoder,
                                   lb=lb)
    pred = model.inference(classifier, X)
    list_metric = []
    for metric in dict_metric.values():
        list_metric.append(metric(y, pred))

    return list_metric
