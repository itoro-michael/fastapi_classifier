# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- Developed by Itoro Ikon.
- Uses the RandomForestClassifier.
- Used for predicting the salary of an individual.

## Intended Use
- Intended for use with US census data.

## Training Data
- US census data, census.csv.


## Evaluation Data
- 20% of the entire dataset is used for model evaluation.

## Metrics
_Please include the metrics used and your model's performance on those metrics._
- The metric are precision, recall, fbeta with values 0.80, 0.54, and 0.64 respectively.

## Ethical Considerations
- Data used is publicly available.

## Caveats and Recommendations
- Multiple classifiers could be used to arrive at optimum performance.
- The optimal classifier can still be fine tuned with hyper parameter search for improved performance.