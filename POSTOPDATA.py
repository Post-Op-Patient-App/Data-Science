# importing the required modules import pandas as pd
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np


# reading in the data set with a index column of 0

def run(one, two, three, four, five, six, seven, eight, nine):
    columns = ['L-CORE', 'L-SURF', 'L-O2', 'L-BP', 'SURF-STBL', 'CORE-STBL', 'BP-STBL', 'COMFORT', 'DECISION']
    data = pd.read_csv('post-operative1.data', index_col=0)

    # if the far left column (not the row index), is not equal to 'L-CORE' I drop that column

    if data.columns[0] != 'L-CORE':
        data = data.drop(data.columns[0], axis=1)

    # Setting the features and the target

    features = ['L-CORE', 'L-SURF', 'L-O2', 'L-BP', 'SURF-STBL', 'CORE-STBL', 'BP-STBL', 'COMFORT']
    target = 'DECISION'

    # setting the model

    model = LogisticRegression(solver='newton-cg')

    # splitting the data into a test train split

    train, test = train_test_split(data)

    # fitting the model

    model.fit(train[features], train[target])

    # setting the function inputs to be equal to a tuple

    tup1 = [one, two, three, four, five, six, seven, eight, nine]

    # assigning that to be an array with dtype float64

    tup1 = np.asarray(tup1, dtype='float64')

    # setting the models prediction equal to leave

    leave = model.predict([tup1[:-1]])

    # making the tuple into a data frame

    tup1 = pd.DataFrame(tup1)

    # finding the models accuracy

    yhat = model.predict(test[features])
    acc = sum(yhat == test['DECISION']) / len(yhat)

    # getting the correct format for the leave output

    leave = np.round(leave)
    leave = leave.astype(float)
    leave = leave[0]
    leave = str(leave)

    # making the new tuple a new transposed data frame and appending that to the current data frame

    new_row = tup1
    new_row = new_row.T
    new_row.columns = columns
    data = data.append(new_row, ignore_index=True)
    data.to_csv('post-operative1.data', index=True)
    print(data.tail())
    return leave


run(1, 1, 1, 1, 1, 1, 1, 1, 1)
