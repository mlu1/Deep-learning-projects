import tensorflow as tf
print(tf.__version__)

#create a constant tensor
scaler = tf.constant(7)
print(scaler)
print(scaler.ndim)

vector = tf.constant([10,10])
print(vector)
print(vector.ndim)

#create a matrix
matrix  = tf.constant(([10,7],[7,10]))
print(matrix)
print(matrix.ndim)

#create another matrix
another_matrix = tf.constant([[10.,7.],
                              [3.,2.],
                              [8.,9]],dtype = tf.float16)

print(another_matrix)
print(another_matrix.ndim)

#Create a tensor now

tensor = tf.constant([
                     [[1,2,3,],
                      [4,5,6]],
                     [[7,8,9],
                      [10,11,12]],
                     [[13,14,15],
                      [16,17,18]]]
                     )
print(tensor)

"""
*Scaler : a single number
*Vector a number with direction for example (wind speed and direction)
*Matrix : a 2 dimensional array of numbers (that means n can be any number, 0 dimenasional tensor is a scaler, 1-dimeansional tensor is a vector)
"""

#Creating the same tensor with tf.Variable

v1 = tf.Variable([10,7]) 
c1 = tf.constant([10,7])


print(v1)
print(c1)

##Expectaton is that the tf.Variable is the only one that cant be changed!
v1[0].assign(7)
print(v1)

#seed the tensor 
random_tensor1 = tf.random.Generator.from_seed(42)
random_tensor1 = random_tensor1.normal(shape=(3,2))
random_tensor2 = tf.random.Generator.from_seed(42)
random_tensor2 = random_tensor2.normal(shape=(3, 2))

print(random_tensor1 == random_tensor2)

"""
ofcourse they are equal! The seed was set to the same value
The below code will change just the seed and lets see what happens!
"""

# Create two random (and different) tensors
random_3 = tf.random.Generator.from_seed(42)
random_3 = random_3.normal(shape=(3, 2))
random_4 = tf.random.Generator.from_seed(11)
random_4 = random_4.normal(shape=(3, 2))

# Check the tensors and see if they are equal when the seed is changed
print(random_tensor1 == random_3)#These have used the same seed and the answer will be True!
print(random_3 == random_4)#Different seeds have been used as such the answer will be False!

