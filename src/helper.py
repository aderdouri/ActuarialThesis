import sys
import subprocess

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap


# Metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import matthews_corrcoef, confusion_matrix


def installALL():
  subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'imbalanced-learn'])
  subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'catboost'])
  subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'hdbscan'])
  subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'shape'])
  subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'lime'])
  subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'optuna'])

  subprocess.call('rm -rf ActuarialThesis', shell=True)
  subprocess.call('git clone https://github.com/aderdouri/ActuarialThesis.git', shell=True)



# Auxilary function to simplify metric calculation
def plot_pr_auc(model, x, y, OurModelName= '_', title=''):

    yhat = model.predict_proba(x)
    model_probs = yhat[:, 1]

    # calculate the precision-recall auc
    precision, recall, _ = precision_recall_curve(y, model_probs)
    auc_score = auc(recall, precision)

    # calculate the no skill line as the proportion of the positive class
    no_skill = len(y[y==1]) / len(y)
    # plot the no skill precision-recall curve
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='Dummy')

    # plot model precision-recall curve
    precision, recall, _ = precision_recall_curve(y, model_probs)
    plt.plot(recall, precision, marker='.', label='Our model')

    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    #plt.title(title + auc_score)
    #plt.title(title + str(auc_score), y=-0.25)
    plt.legend()
    plt.show()

    fig = plt.gcf()
    return fig

# Auxilary function to simplify metric calculation
def plot_pr_auc_val(model, X_val, y_val, OurModelName='_', title=''):

    yhat = model.predict_proba(X_val)
    y_pred = model.predict(X_val)
    model_probs = yhat[:, 1]

    precision, recall, _ = precision_recall_curve(y_val, model_probs)
    auc_score = auc(recall, precision)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.set_size_inches(20, 6)

    # calculate the no skill line as the proportion of the positive class
    no_skill_val = len(y_val[y_val==1]) / len(y_val)

    # plot model precision-recall curve
    precision, recall, _ = precision_recall_curve(y_val, model_probs)

    # plot the no skill precision-recall curve
    ax1.plot([0, 1], [no_skill_val, no_skill_val], linestyle='--', label='Dummy')
    ax1.plot(recall, precision, marker='.', label=OurModelName)
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    #ax1.set_title(title + str(auc_score_val), y=-0.25)
    ax1.legend()

    clf_report = classification_report(y_val, y_pred, output_dict=True)

    from matplotlib.colors import ListedColormap
    ax = sns.heatmap(pd.DataFrame(clf_report).T, 
                annot=True, 
                cbar=False, 
                square=False,
                fmt='g',
                linewidths=0.5,
                #cmap=ListedColormap(['red']),
                cmap=plt.cm.Blues,
                ax=ax2
                );  
    ax.xaxis.tick_top()              

    #ax2.set_xlabel('Recall')
    #ax2.set_ylabel('Precision')
    Accuracy = accuracy_score(y_val, y_pred)
    Accuracy_balanced = balanced_accuracy_score(y_val, y_pred)

    ax2.set_title('(Accuracy, balanced Accuracy) = ({:.3f}, {:.3f}) '.format(Accuracy, Accuracy_balanced), y=-0.25)
    #ax2.axis('off')

    cm = confusion_matrix(y_val, y_pred, labels=[1, 0])
    sns.heatmap(cm, annot=True, fmt="d")
    ax3.set_ylabel('Actual label')
    ax3.set_xlabel('Predicted label')
    ax3.set_title('Confusion matrix')
    
    return fig


