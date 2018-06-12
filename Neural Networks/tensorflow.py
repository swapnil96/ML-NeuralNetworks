import tensorflow as tf
import numpy as np
import os
from sklearn.utils import shuffle

final = {}
def read_data(link):

    files = os.listdir(link)
    idx = 0
    for file1 in files:
        now = link + file1
        final[idx] = file1.split(".")[0]
        if idx == 0:
            train = np.load(now)
            main = np.array([final[idx]]*train.shape[0])
            temp = np.zeros(20)
            temp[idx] = 1
            m_lbl = np.array([temp]*train.shape[0])

        else:
            temp1 = np.load(now)
            temp2 = np.array([final[idx]]*temp1.shape[0])
            temp = np.zeros(20)
            temp[idx] = 1
            temp3 = np.array([temp]*temp1.shape[0])
            train = np.vstack([train, temp1])
            main = np.hstack([main, temp2])
            m_lbl = np.vstack([m_lbl, temp3])

        idx += 1

    print(final)
    print(train.shape)
    return train, main, m_lbl

test = np.load("col-774-spring-2018/test/test.npy")
train, main, m_lbl = read_data("col-774-spring-2018/train/")

# train, m_lbl = shuffle(train, m_lbl, random_state=0)
# train /= np.max(train)
# test /= np.max(test)

learning_rate = 0.01
num_steps = 10000
batch_size = 5000
display_step = 100

n_hidden_1 = 1024  # 1st layer number of neurons
num_input = 784  # data input (img shape: 28*28)
num_classes = 20  # total classes (0-19 class)

X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_hidden_1, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer

# def neural_net(x, weights, biases, keep_prob):
#     layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
#     layer_1 = tf.nn.sigmoid(layer_1)
#     layer_1 = tf.nn.dropout(layer_1, keep_prob)
#     out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
#     # out_layer = tf.nn.softma
#     return out_layer

logits = neural_net(X)
prediction = tf.nn.softmax(logits)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps + 1):
        # batch_x, batch_y = mnist.train.next_batch(batch_size)
        idx = np.random.randint(0, train.shape[0], batch_size)
        batch_x = train[idx]
        batch_y = m_lbl[idx]
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run(
                [loss_op, accuracy], feed_dict={
                    X: batch_x,
                    Y: batch_y
                })
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for MNIST test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: train,
                                      Y: m_lbl}))

    predic = tf.argmax(prediction, 1)
    y_test = sess.run(predic, feed_dict={X: test})
    with open("submit_NN.csv", "w") as outfile:
        outfile.writelines("ID,CATEGORY\n")
        for i in range(y_test.shape[0]):
            #         print(y[i])
            outfile.writelines(str(i) + ',' + final[y_test[i]]+ '\n')

    print("done")
