import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

weights = {1: 20, 2: 10, 3: 1}
columns = ['L-CORE', 'L-SURF', 'L-O2', 'L-BP', 'SURF-STBL', 'CORE-STBL', 'BP-STBL', 'COMFORT', 'DECISION']
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/postoperative-patient-data/post-operative.data', na_values="?")
print(data.shape)
data.columns = columns


L_CORE_ranks = {'low': 1, 'mid': 2, 'high': 3}
data['L-CORE'] = data['L-CORE'].map(L_CORE_ranks)

L_SURF_ranks = {'low': 1, 'mid': 2, 'high': 3}
data['L-SURF'] = data['L-SURF'].map(L_SURF_ranks)

L_O2_ranks = {'poor': 1, 'fair': 2, 'good': 3, 'excellent': 4}
data['L-O2'] = data['L-O2'].map(L_O2_ranks)

L_BP_ranks = {'low': 1, 'mid': 2, 'high': 3}
data['L-BP'] = data['L-BP'].map(L_BP_ranks)

L_SURF_STBL_ranks = {'unstable': 1, 'mod-stable': 2, 'stable': 3}
data['SURF-STBL'] = data['SURF-STBL'].map(L_SURF_STBL_ranks)

L_CORE_STBL_ranks = {'unstable': 1, 'mod-stable': 2, 'stable': 3}
data['CORE-STBL'] = data['CORE-STBL'].map(L_CORE_STBL_ranks)

L_BP_STBL_ranks = {'unstable': 1, 'mod-stable': 2, 'stable': 3}
data['BP-STBL'] = data['BP-STBL'].map(L_BP_STBL_ranks)

L_DECISION_ranks = {'I': 1, 'A': 2, 'S': 3}
data['DECISION'] = data['DECISION'].map(L_DECISION_ranks)

data = data.dropna(how='any')

data['COMFORT'] = data['COMFORT'].astype('float')

features = ['L-CORE', 'L-SURF', 'L-O2', 'L-BP', 'SURF-STBL', 'CORE-STBL', 'BP-STBL', 'COMFORT']
target = 'DECISION'
model = LogisticRegression(solver='newton-cg')
train, test = train_test_split(data, shuffle=False)
model.fit(train[features], train[target])

tup1 = (1, 2, 3, 2, 1, 3, 2, 12, 3)
leave = model.predict([list(tup1)[:-1]])
leave = np.round(leave)
leave = leave.astype(float)
leave = leave[0]
print("The models prediction says", leave, ". The doctors prediction says", tup1[8])
tup1 = pd.DataFrame(tup1)
new_row = tup1
new_row = new_row.T
new_row.columns = columns
data = data.append(new_row, ignore_index=True)
data.to_csv(r'C:\Users\Snick\Documents\Personal_Sync\Personal\Python\Python_Projects\post-operative1.data', index=True)
