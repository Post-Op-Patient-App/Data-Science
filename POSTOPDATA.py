import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


columns = ['L-CORE', 'L-SURF', 'L-O2', 'L-BP', 'SURF-STBL', 'CORE-STBL', 'BP-STBL', 'COMFORT', 'DECISION']
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/postoperative-patient-data/post-operative.data', names=columns, na_values="?")

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

train, test = train_test_split(data, random_state=0)

features = ['L-CORE', 'L-SURF', 'L-O2', 'L-BP', 'SURF-STBL', 'CORE-STBL', 'BP-STBL', 'COMFORT']
target = ['DECISION']
model = LinearRegression()
model.fit(train[features], train[target])
leave = model.predict([[1, 1, 1, 1, 1, 1, 1, 1]])
print(model.coef_, model.intercept_)


