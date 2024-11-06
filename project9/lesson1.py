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

'''
Numpy introduction
'''
import numpy as np
rand_value=np.random.randint(1,1000)
print(rand_value)
print(np.random.randint(-10,10))
print(np.array(range(1,10,2)))

a_zeros = np.zeros([5,5])
a_ones = np.ones([5,5])

print(a_zeros.reshape(1,-1))
print(a_ones.reshape(1,-1))
print(a_ones.ravel())
print(a_ones.flatten())
print(np.arange(1,11,3,dtype='int'))


'''
Pandas introductions
'''
import pandas as pd
df = pd.read_csv('data/Restaurants-Customers.csv.txt')
print(df.head(5))
print(df.tail(5))
print(df.sample(5))
index = df['Gender']=='Male'
print(df['Gender'].value_counts())