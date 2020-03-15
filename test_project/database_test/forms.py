from django import forms

class registration_forms(forms.Form):
    username = forms.CharField(max_length=100)
    password = forms.CharField(max_length=100)
    email = forms.CharField(max_length=100) 
    phone = forms.CharField(max_length=100)