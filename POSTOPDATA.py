import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


columns = ['L-CORE', ' L-SURF', 'L-O2', 'L-BP', 'SURF-STBL', 'CORE-STBL', 'BP-STBL', 'COMFORT', 'DECISION']
data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/postoperative-patient-data/post-operative.data', names=columns)
print(data.head())
train, test = train_test_split(data.copy(), random_state=0)
features = ['L-CORE', ' L-SURF', 'L-O2', 'L-BP', 'SURF-STBL', 'CORE-STBL', 'BP-STBL', 'COMFORT']
target = ['DECISION']
model = LinearRegression()
model.fit(train[features], train[target])
