import shutil
import pandas as pd

# shutil.copy('input/out/submission.csv', 'submission.csv')
xx = pd.read_csv('input/out/submission.csv')
print(xx.shape)

xx = pd.read_csv('input/submission.csv')
print(xx.get(0))
