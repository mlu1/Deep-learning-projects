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
                     [
                      [1,2,3,],
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





