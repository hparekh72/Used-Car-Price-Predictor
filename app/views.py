from django.shortcuts import render, redirect
from app.predictor import Predictor

# Create your views here.



def index(request):
    p = Predictor()
    cars = list(p.cars_set)
    for i in cars:
        if isinstance(i, float):
            cars.remove(i)
    cars.sort()
    m = {'cars': cars, 'name': 'Arpan'}
    return render(request, 'index/index.html', context=m)


def predict(request):
    p = Predictor()
    name = request.POST['name']
    km = int(request.POST['km'])
    year = int(request.POST['year'])
    print(name, year, km)
    val = p.predict_single(name, km, year)
    m = {'name': name, 'val': val}
    return render(request, 'index/result.html', m)
