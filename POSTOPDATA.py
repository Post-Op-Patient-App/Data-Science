import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np
import csv



columns = ['L-CORE', 'L-SURF', 'L-O2', 'L-BP', 'SURF-STBL', 'CORE-STBL', 'BP-STBL', 'COMFORT', 'DECISION']
data = pd.read_csv('post-operative1.data', index_col=1)
data.columns = columns
data2 = pd.read_csv('accuracy.data')
data = data.dropna(how='any')
features = ['L-CORE', 'L-SURF', 'L-O2', 'L-BP', 'SURF-STBL', 'CORE-STBL', 'BP-STBL', 'COMFORT']
target = 'DECISION'
model = LogisticRegression(solver='newton-cg')
train, test = train_test_split(data)
model.fit(train[features], train[target])
tup1 = (1, 2, 3, 2, 1, 3, 2, 12, 3)
leave = model.predict([list(tup1)[:-1]])
yhat = model.predict(test[features])
acc = sum(yhat == test['DECISION']) / len(yhat)
leave = np.round(leave)
leave = leave.astype(float)
leave = leave[0]
tup1 = pd.DataFrame(tup1)
new_row = tup1
new_row = new_row.T
new_row.columns = columns
data = data.append(new_row, ignore_index=True)
data.to_csv(r'C:\Users\Snick\Documents\Personal_Sync\Personal\Python\Python_Projects\post-operative1.data', index=True)
