import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# sigmoid function for logistic regression
def sig(value):
    sig = 1 / (1 + np.exp(-value))  # exp(-4v + 12)
    return sig


def loss(y, y_pred):
    loss = -np.mean(y * (np.log(y_pred)) - (1 - y) * np.log(1 - y_pred))
    return loss


cancer = pd.read_csv('breast-cancer-wisconsin.data',
                     sep=',',
                     names=["Sample", "Clump Thickness", "Cell Size", "Cell Shape", "Marginal Adhesion",
                            "Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitosis",
                            "Class"])

cancer.drop(['Sample', 'Class'], axis=1).plot(kind='box', subplots=True,
                                              layout=(3, 3), sharex=False, sharey=False, figsize=(9, 9))

# plt.show() # visualise data

features = ["Clump Thickness", "Cell Size", "Cell Shape", "Marginal Adhesion",
            "Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitosis"]

# There are 9 features, 1 target so X is (699 x 10), y is (699 x 1), meaning beta is (10 x 1)
X = cancer[features]
y = cancer["Class"]
# We want to use a 70:30 split for training:test data - NOTE we should visualise how training:test split affects results
# As we have 699 items, we want 489 items in the training set, leaving 210 items in the test set
train_set = cancer[1:489]
test_set = cancer[490:699]

# for a linear model, the optimal beta* is (X^T*X)^-1*X^T*Y

X = np.loadtxt('breast-cancer-wisconsin.data', delimiter=',', usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9), dtype=int)
y = np.loadtxt('breast-cancer-wisconsin.data', delimiter=',', usecols=(10), dtype=int)
# y = (y-3) # normalises targets to -1 or 1 instead of 2 or 4
y = (y-2)/2 # normalises targets to 0 or 1 instead of 2 or 4

# add column of ones to front of feature matrix
X = np.c_[np.ones(699, dtype=int), X]

# cut X and y down to first 489 items to form training set, remainder is test set
X_train = X[:489]
y_train = y[:489]

X_test = X[489:]
y_test = y[489:]

# calculate optimal beta* - done in stages because i dont know how python matrices work
XTX = np.matmul(X_train.transpose(), X_train)
XTX1X = np.matmul(np.linalg.inv(XTX), X_train.transpose())
beta = np.matmul(XTX1X, y_train)
print("Beta = ", beta)

# form predictions - y^ = X*beta
y_pred = np.matmul(X_test, beta)
print("Standard predictions = ", y_pred)
print("Chance of being malignant = ", sig(y_pred))
print("Loss = ", loss(y_test,np.abs(y_pred)))

# print("Average loss = ", np.sum(loss) / np.count_nonzero(loss))
