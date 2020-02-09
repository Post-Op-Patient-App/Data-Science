import pandas as pd
import sqlite3
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine

engine = create_engine("postgres://hmtkaddpmvzihq:32053e9be16914fddb77094492892768be24e5c6661de44d2fc646e6dc6c33ef@ec2-184-72-235-80.compute-1.amazonaws.com:5432/d9sfvg60m4mcgs")

run = True

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

my_cursor = engine.connection.cursor()

data.to_sql('data', con=engine)
print(data.dtypes)
'''
features = ['L-CORE', 'L-SURF', 'L-O2', 'L-BP', 'SURF-STBL', 'CORE-STBL', 'BP-STBL', 'COMFORT']
target = ['DECISION']
model = LinearRegression()
while run:
    train, test = train_test_split(data, random_state=0)
    model.fit(train[features], train[target])
    tup1 = (1, 2, 3, 2, 1, 3, 2, 12, 3)
    leave = model.predict([[tup1[0], tup1[1], tup1[2], tup1[3], tup1[4], tup1[5], tup1[6], tup1[7]]])
    leave = int(leave)
    print("The models prediction says", leave, ". The doctors prediction says", tup1[8])
    tup1 = pd.DataFrame(tup1)
    new_row = tup1
    new_row = new_row.T
    new_row.columns = columns
    print(data.shape)
    data = data.append(new_row, ignore_index=True)
    print(data.shape)
    sqlite3()
    run = False


'''