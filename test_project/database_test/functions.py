from django_pandas.managers import DataFrameManager
import pandas as pd
from .models import employees
import psycopg2
from sqlalchemy import create_engine

def to_dataframe(new_info):
    alchemyengine = create_engine('postgres://hmtkaddpmvzihq:32053e9be16914fddb77094492892768be24e5c6661de44d2fc646e6dc6c33ef@ec2-184-72-235-80.compute-1.amazonaws.com:5432/d9sfvg60m4mcgs')
    dbConnection = alchemyengine.connect();

    data = pd.read_sql("select * from \"database_test_employees\"", dbConnection);
    print(data.head(1))
