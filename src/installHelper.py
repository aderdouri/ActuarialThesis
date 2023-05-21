import sys
import subprocess

def installALL():
  subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'imbalanced-learn'])
  subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'catboost'])
  subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'hdbscan'])
  subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'shape'])
  subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'lime'])
  subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'optuna'])

  subprocess.call('rm -rf ActuarialThesis', shell=True)
  subprocess.call('git clone https://github.com/aderdouri/ActuarialThesis.git', shell=True)

