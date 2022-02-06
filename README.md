# breast-cancer-classification
For Machine Learning Part 1 Project - co-authored by Tom Aldridge and Reece Hill, University of Nottingham

Logistic regression model with specifyable random seeding and stratification of data. Manages missing data entries from dataset ("?") in one of four ways: remove the entire row containing the missing data, replace the missing data with: mean, maximum or minimum for feature. Iterates over all possible training set sizes (from 1 training entry to 699 (number of data points) - 1 training entries). Metrics (accuracy, specificity, sensitivity and precision) are stored in a report then output to .csv for analysis. Metrics are plotted and graphs are customised. Final report written in LaTeX.

Note: all code is in one file as project submission required this. 
