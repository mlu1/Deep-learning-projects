import tensorflow as tf
print(tf.__version__)

#create a constant tensor
scaler = tf.constant(7)
print(scaler)
print(scaler.ndim)

vector = tf.constant([10,10])
print(vector)
print(vector.ndim)

matrix  = tf.constant(([10,7],[7,10]))
print(matrix)
print(matrix.ndim)




