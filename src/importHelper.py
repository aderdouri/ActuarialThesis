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

