import sys
import subprocess

def installALL():
  subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'imbalanced-learn'])
  subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'catboost'])
  subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'hdbscan'])
  subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'shap'])
  subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'lime'])
  subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'optuna'])
  subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'kaleido'])
