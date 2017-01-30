import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#tensor represents the input we'll give. 784 is flattened pixel array
#None means we don't know how many inputs we'll give
x = tf.placeholder(tf.float32, [None, 784])
#784x10 tensor because we want to multiply this by x and end up with
#10 dimensional vectors of evidence for the different classes
W = tf.Variable(tf.zeros([784, 10]))
#bias. Has a shape of [10] so we can add it to the output matrix
b = tf.Variable(tf.zeros([10]))
#output is the dot product of x and W matrix added to b
y = tf.nn.softmax(tf.matmul(x, W) + b)

#correct answers
y_ = tf.placeholder(tf.float32, [None, 10])
#cross-entropy function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for in in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys}
