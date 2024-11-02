print(type(34/2))
print(type(int(34/2)))
print(type('this is string'))

var1 = [2,3,4,5]
print(var1)
var2 = type(var1)
print(var2)
var3= {2,3,4,5,6,'this is str'}

var4 = {'one':1,
        'country':'India',
        'capital':'New Dehli',
        'array':[1,2,3,4]
        }

a = 43
b=23

##conditional statements
if(a<b):
    print('A is less tha b')
elif(b>a):
    print('b  is greater than a')
elif(b==a):
    print('b and a are equal')
else:
    print('A is greater than b')

for value in var3:
    if(type(value) ==str):
        print(value)
    else:
        print('no str value found')

a1= 10
while a>0:
    print('i just do {} operation'.format(a))
    a=a-1

def add_number(a,b):
    c= a+b
    return c
d = add_number(2,7)

import numpy as np
from numpy import random

rand_value=random.randint(1,1000)

print(rand_value)