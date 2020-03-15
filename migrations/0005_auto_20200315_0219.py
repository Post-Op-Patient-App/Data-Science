# Generated by Django 3.0.4 on 2020-03-15 07:19

import datetime
from django.db import migrations, models
from django.utils.timezone import utc


class Migration(migrations.Migration):

    dependencies = [
        ('database_test', '0004_auto_20200315_0149'),
    ]

    operations = [
        migrations.AlterField(
            model_name='database',
            name='pub_date',
            field=models.DateField(default=datetime.datetime(2020, 3, 15, 7, 18, 59, 58079, tzinfo=utc)),
        ),
        migrations.AlterField(
            model_name='employees',
            name='pub_date',
            field=models.DateField(default=datetime.datetime(2020, 3, 15, 7, 18, 59, 113083, tzinfo=utc)),
        ),
    ]