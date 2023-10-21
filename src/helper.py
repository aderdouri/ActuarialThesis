# Evergrowing list of imports

# The basics
import pandas as pd
import numpy as np
from joblib import dump, load
import re

# The viz
import matplotlib.pyplot as plt
import seaborn as sns
import scikitplot as skplt
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
from sklearn.metrics import mean_squared_error
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

# Others
import kaleido


colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

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

  clf_report_df = pd.DataFrame(clf_report)
  clf_report_df.drop(['support'], inplace=True)

  ax = sns.heatmap(clf_report_df.T, 
              annot=True, 
              cbar=False, 
              square=False,
              fmt='g',
              linewidths=0.9,
              cmap=plt.cm.Blues,
              ax=ax1
              );  
  ax.xaxis.tick_top()
  ax.set_title('Classification report')
  ax.set_xlabel('Accuracy (balanced): {:.5f}'.format(balanced_accuracy_score(y_test, y_pred)))

  skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize=True, ax=ax2)

  fig = plt.gcf()
  return fig  

def df_ObservedVsPredicted(y_observed, y_predicted):
  df = pd.DataFrame({
      'Observed': y_observed,
      'Predicted': y_predicted
      })
  return df


def plotObservedVsPrediced(y_observed, y_predicted):
  # Create two subplots and unpack the output array immediately
  fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True,  figsize=(15, 4))

  # Create scatter plot with actual and predicted values
  sns.scatterplot(ax=ax1, x=y_observed, y=y_predicted)
  ax1.set_xlabel('Actual Values')
  ax1.set_ylabel('Predicted Values')
  ax1.set_title('Actual vs Predicted Values')

  # Create regression plot with actual and predicted values
  sns.regplot(ax=ax2, x=y_observed, y=y_predicted, scatter_kws={'s': 10}, line_kws={'color': 'red'})
  ax2.set_xlabel('Predicted Values')
  ax2.set_ylabel('Residuals')
  ax2.set_title('Residual Plot of Actual vs Predicted Values');
  return fig

def plotLiftChart(df, nbBins=10):
  df['Exposure'] = np.ones(len(df))
  df.sort_values(by='Predicted', inplace=True, ascending=True)
  df['cumExpo'] = np.cumsum(df['Exposure'])  
  df['ct'] = pd.qcut(df['cumExpo'], nbBins, labels=False)
  avgObserved = (df.groupby(by=['ct'])['Observed'].mean()) / df['Predicted'].mean()
  avgPredicted = (df.groupby(by=['ct'])['Predicted'].mean()) / df['Predicted'].mean()

  fig, ax = plt.subplots(figsize=(12, 6))
  ax.plot(range(nbBins), avgObserved, linestyle="-", label='Observed', marker='o')
  ax.plot(range(nbBins), avgPredicted, linestyle="-", label='Predicted', marker='o')
  ax.legend(loc="upper left")
  ax.set_title('Lift Chart Observed vs Predicetd', fontsize=18)
  ax.set_xlabel('Deciles')
  ax.set_ylabel('Mean normalized CHARGE')
  ax.axline((0, avgPredicted.iloc[0]), (nbBins, avgPredicted.iloc[0]), linewidth=0.5, color='r')
  ax.axline((0, avgPredicted.iloc[nbBins-1]), (nbBins, avgPredicted.iloc[nbBins-1]), linewidth=0.5, color='r')
  plt.plot()
  return fig, ax

def lorenz_curve(y_true, y_pred, exposure):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    exposure = np.asarray(exposure)

    # order samples by increasing predicted risk:
    ranking = np.argsort(y_pred)
    ranked_exposure = exposure[ranking]
    ranked_pure_premium = y_true[ranking]
    cumulated_claim_amount = np.cumsum(ranked_pure_premium * ranked_exposure)
    cumulated_claim_amount /= cumulated_claim_amount[-1]
    cumulated_samples = np.linspace(0, 1, len(cumulated_claim_amount))
    return cumulated_samples, cumulated_claim_amount


def plotLorenzCurve(df):  
  fig, ax = plt.subplots(figsize=(8, 10))
  y_pred = df['Predicted']
  df['Exposure'] = np.ones(len(df))
  label = 'Model'

  ordered_samples, cum_claims = lorenz_curve(
      df["Observed"], y_pred, df["Exposure"]
      )
  gini = 1 - 2 * auc(ordered_samples, cum_claims)
  label += " (Gini index: {:.3f})".format(gini)
  ax.plot(ordered_samples, cum_claims, linestyle="-", color=colors[1], label=label)

  # Oracle model: y_pred == y_test
  ordered_samples, cum_claims = lorenz_curve(
      df["Observed"], df["Observed"], df["Exposure"]
  )
  gini = 1 - 2 * auc(ordered_samples, cum_claims)
  label = "Oracle (Gini index: {:.3f})".format(gini)
  ax.plot(ordered_samples, cum_claims, linestyle="-.", color="gray", label=label)

  # Random baseline
  ax.plot([0, 1], [0, 1], linestyle="--", color="black", label="Random baseline")
  ax.set(
      title="Lorenz Curves",
      xlabel="Fraction of policyholders\n(ordered by model from safest to riskiest)",
      ylabel="Fraction of total claim amount",
  )
  ax.legend(loc="upper left")
  plt.plot()    
  return fig, ax