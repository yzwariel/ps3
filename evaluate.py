import pandas as pd
import pylab as pl
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
from sklearn.svm import SVC

def evaluate(y_test, yhat):
    '''
    Evaluate classifier
    Input:
        y_test: actual values
        yhat: prediceed values
    Returns:
        confusion_matrix: confusion_matrix using pandas
        report: Classification report
    '''
    confusion_matrix  = pd.crosstab(yhat, y_test, rownames=["Actual"], colnames=["Predicted"])
    report = classification_report(y_test, yhat, labels=[0, 1])
    return (confusion_matrix , report)


def plot_roc(name, probs,y_test):
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    pl.clf()
    pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    pl.plot([0, 1], [0, 1], 'k--')
    pl.xlim([0.0, 1.05])
    pl.ylim([0.0, 1.05])
    pl.xlabel('False Positive Rate')
    pl.ylabel('True Positive Rate')
    pl.title(name)
    pl.legend(loc="lower right")
    pl.show()