from django.contrib import admin

# Register your models here.
from .models import database, employees, registrationdata

admin.site.register(database)
admin.site.register(employees)
admin.site.register(registrationdata)