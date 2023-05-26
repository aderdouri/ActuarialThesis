# Evergrowing list of imports

# The basics
import pandas as pd
import numpy as np
from joblib import dump, load
import re

# The viz
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap

# Data preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn.utils.class_weight import compute_class_weight

# Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import xgboost.sklearn as xgb
import lightgbm.sklearn as lgbm

from sklearn.linear_model import LogisticRegression

# Over- under- samplers
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import ADASYN, RandomOverSampler
from imblearn.under_sampling import NearMiss, RandomUnderSampler, TomekLinks

# Clustering
import hdbscan

# Explaining
from shap import TreeExplainer
from shap import summary_plot
import shap
from lime.lime_tabular import LimeTabularExplainer

# Models parameters tunning
import optuna
from optuna.integration import LightGBMPruningCallback


def getPrecisionRecallCurve(model, X, y):
  yhat = model.predict_proba(X)
  model_probs = yhat[:, 1]
  precision, recall, _ = precision_recall_curve(y, model_probs)
  return precision, recall

def positiveClassProportion(y):
  return len(y[y==1])/len(y)

# Auxilary function to simplify metric calculation
def plot_pr_auc(model, X_train, y_train, X_test, y_test, OurModelName= '_', title=''):
  # Calculate the no skill line as the proportion of the positive class
  no_skill_train = positiveClassProportion(y_train)
  no_skill_test = positiveClassProportion(y_test)

  fig, (ax1, ax2) = plt.subplots(1, 2)

  # Plot the no skill precision-recall curve
  ax1.plot([0, 1], [no_skill_train, no_skill_train], linestyle='--', label='Dummy')    
  ax2.plot([0, 1], [no_skill_test, no_skill_test], linestyle='--', label='Dummy')

  # calculate the precision-recall auc
  precision_train, recall_train = getPrecisionRecallCurve(model, X_train, y_train)
  precision_test, recall_test = getPrecisionRecallCurve(model, X_test, y_test)

  # plot model precision-recall curve
  ax1.plot(recall_train, precision_train, marker='.', label=OurModelName)
  ax2.plot(recall_test, precision_test, marker='.', label=OurModelName)

  # axis labels
  ax1.set_xlabel('Recall')
  ax1.set_ylabel('Precision')
  ax1.set_title('Train - Area Under the curve precision-recall curve')
  ax1.legend()

  ax2.set_xlabel('Recall')
  ax2.set_ylabel('Precision')
  ax2.set_title('Test - Area Under the curve precision-recall curve')
  ax2.legend()

  fig = plt.gcf()
  return fig

def plot_classification_report_confusion_matrix(model, X_test, y_test):
  fig, (ax1, ax2) = plt.subplots(1, 2)

  y_pred = model.predict(X_test)
  clf_report = classification_report(y_test, y_pred, output_dict=True)

  ax = sns.heatmap(pd.DataFrame(clf_report).T, 
              annot=True, 
              cbar=False, 
              square=False,
              fmt='g',
              linewidths=0.5,
              #cmap=ListedColormap(['red']),
              cmap=plt.cm.Blues,
              ax=ax1
              );  
  ax.xaxis.tick_top()              

  cm = confusion_matrix(y_test, y_pred, labels=[1, 0])
  sns.heatmap(cm, annot=True, fmt="d")
  ax2.set_ylabel('Actual label')
  ax2.set_xlabel('Predicted label')
  ax2.set_title('Confusion matrix')

  fig = plt.gcf()
  return fig

