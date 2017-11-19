# reading mnist from local
mnist = input_data.read_data_sets("Where you saved mnist", one_hot=True)

# constructing nn using tensorflow
num_units1 = 784
num_units2 = 4096
num_units3 = 2048
num_units4 = 2048
num_units5 = 1024
num_units6 = 10
init_const_num = 0.09
num_filters = 4

stddev01 = mt.sqrt(2 / 10)
stddev02 = mt.sqrt(2 / num_units1)
stddev03 = mt.sqrt(2 / num_units2)
stddev04 = mt.sqrt(2 / num_units3)
stddev05 = mt.sqrt(2 / num_units4)
stddev06 = mt.sqrt(2 / num_units5)

x_image = tf.placeholder(tf.float32, [None, 784])
x_imageInUse = tf.reshape(x_image, [-1,28,28,1])

W_conv = tf.Variable(tf.truncated_normal([5,5,1,num_filters],stddev=stddev01))
h_conv = tf.nn.conv2d(x_imageInUse, W_conv, strides=[1,1,1,1], padding='SAME')
h_pool = tf.nn.max_pool(h_conv, ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')

h_pool_flat = tf.reshape(h_pool, [-1,num_units1])

w4 = tf.Variable(tf.truncated_normal([num_units1, num_units1],stddev=stddev01))
b4 = tf.Variable(tf.constant(init_const_num, shape=[num_units1]))
hidden4 = tf.nn.relu(tf.matmul(h_pool_flat, w4) + b4)

w3 = tf.Variable(tf.truncated_normal([num_units1, num_units2],stddev=stddev02))
b3 = tf.Variable(tf.constant(init_const_num, shape=[num_units2]))
hidden3 = tf.nn.relu(tf.matmul(hidden4, w3) + b3)

w2 = tf.Variable(tf.truncated_normal([num_units2, num_units3],stddev=stddev03))
b2 = tf.Variable(tf.constant(init_const_num, shape=[num_units3]))
hidden2 = tf.nn.relu(tf.matmul(hidden3, w2) + b2)

w1_5 = tf.Variable(tf.truncated_normal([num_units3, num_units4],stddev=stddev04))
b1_5 = tf.Variable(tf.constant(init_const_num, shape=[num_units4]))
hidden1_5 = tf.nn.relu(tf.matmul(hidden2, w1_5) + b1_5)

w1 = tf.Variable(tf.truncated_normal([num_units4, num_units5],stddev=stddev05))
b1 = tf.Variable(tf.constant(init_const_num, shape=[num_units5]))
hidden1 = tf.nn.relu(tf.matmul(hidden1_5, w1) + b1)

w0 = tf.Variable(tf.truncated_normal([num_units5, num_units6],stddev=stddev06))
b0 = tf.Variable(tf.constant(init_const_num, shape=[num_units6]))
p = tf.nn.softmax(tf.matmul(hidden1, w0) + b0)

# constructing opimizers, loss, accracy and more
t = tf.placeholder(tf.float32, [None, 10])
loss = -tf.reduce_sum(t * tf.log(p))
train_step = tf.train.AdamOptimizer(0.0002).minimize(loss)
correct_prediction = tf.equal(tf.argmax(p,1), tf.argmax(t,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# trying to use tensor board
tf.summary.scalar("loss", loss)
tf.summary.scalar("accuracy", accuracy)
tf.summary.histogram("weights_hidden",w1)
tf.summary.histogram("biases_hidden",b1)
tf.summary.histogram("weights_hidden2",w2)
tf.summary.histogram("biases_hidden2",b2)
tf.summary.histogram("weights_hidden3",w3)
tf.summary.histogram("biases_hidden3",b3)
summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter("tf_log2017-07-31", graph=sess.graph)

# instanciate sess
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

# training
i = 2000
for _ in range(2000):
    i += 1
    xs, ts = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={x_image:xs, t:ts})
    if i % 50 == 0:
        loss_val, acc_val,summary = sess.run([loss,accuracy, summary_op],
                                    feed_dict={x_image:mnist.test.images, 
                                               t:mnist.test.labels})
        summary_writer.add_summary(summary,i)
        print ('Step: %d, Loss: %f, Accuracy: %f' % (i, loss_val, acc_val))
        saver.save(sess, 'mfc_sessionAdam2017-08-07', global_step=i)
