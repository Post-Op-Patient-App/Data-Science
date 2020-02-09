import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np


def run(one, two, three, four, five, six, seven, eight, nine):
    columns = ['L-CORE', 'L-SURF', 'L-O2', 'L-BP', 'SURF-STBL', 'CORE-STBL', 'BP-STBL', 'COMFORT', 'DECISION']
    data = pd.read_csv('post-operative1.data', index_col=0)
    if data.columns[0] != 'L-CORE':
        data = data.drop(data.columns[0], axis=1)
    features = ['L-CORE', 'L-SURF', 'L-O2', 'L-BP', 'SURF-STBL', 'CORE-STBL', 'BP-STBL', 'COMFORT']
    target = 'DECISION'
    model = LogisticRegression(solver='newton-cg')
    train, test = train_test_split(data)
    model.fit(train[features], train[target])
    lineList = list()
    with open('accuracy.data') as f:
        for line in f:
            lineList.append(line)
    lineList = lineList[0]
    lineList = str(lineList).strip(r",'")
    lineListNew = []
    for i in lineList:
        if i == '1':
            lineListNew.append(i)
        elif i == '2':
            lineListNew.append(i)
        elif i == '3':
            lineListNew.append(i)
        elif i == '4':
            lineListNew.append(i)
        else:
            pass
    tup1 = [one, two, three, four, five, six, seven, eight, nine]
    tup1 = np.asarray(tup1, dtype='float64')
    leave = model.predict([tup1[:-1]])
    tup1 = pd.DataFrame(tup1)
    yhat = model.predict(test[features])
    acc = sum(yhat == test['DECISION']) / len(yhat)
    leave = np.round(leave)
    leave = leave.astype(float)
    leave = leave[0]
    leave = str(leave)
    tup1 = pd.DataFrame(tup1)
    new_row = tup1
    new_row = new_row.T
    new_row.columns = columns
    data = data.append(new_row, ignore_index=True)
    data.to_csv('post-operative1.data', index=True)
    return leave
