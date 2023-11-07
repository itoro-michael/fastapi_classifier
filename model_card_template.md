# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- Developed by Itoro Ikon.
- Uses the RandomForestClassifier.
- Used for predicting the salary of an individual.

## Intended Use
- Intended for use with US census data.

## Training Data
- census.csv: The US census data.


## Evaluation Data
- 20% of the entire dataset is used for model evaluation.

## Metrics
- The metrics are precision=0.80, recall=0.54, fbeta=0.64, and 0.85 for accuracy.

## Ethical Considerations
- Data used is publicly available.
- Model performs effectively across the different racial groups in the dataset.
- Least accuracy on test set for a racial group is 0.82 for Asian-Pac-Islander, and the most is 0.93 for Blacks.

## Caveats and Recommendations
- Multiple classifiers could be used to arrive at optimum performance.
- The optimal classifier can still be fine tuned with hyper parameter search for improved performance.