
from util import  *
import numpy as np
import matplotlib as mlp

mlp.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

print '########this is an experiment on ssl with ac-gan using n+1 classes######'
training_data, training_labels, validation_data, validation_labels, test_data, test_labels = rap_mnist()

training_data = np.array(training_data)
training_data = ((training_data.astype(np.float32) - 127.5) / 127.5).tolist()
training_labels = [vectorized_result(l) for l in training_labels]

#limiting the number of labels used in training
label_idx = group_labels(validation_labels, 1000)
gr_labels =[]
gr_data = []

for l in label_idx:
    for k in l:
        gr_data.append(validation_data[k])
        gr_labels.append(validation_labels[k])


validation_data = np.array(gr_data)
validation_data = ((validation_data.astype(np.float32) - 127.5) / 127.5)
validation_labels = [vectorized_result(l) for l in gr_labels]

training_data.extend(validation_data)
training_labels.extend(validation_labels)

test_data = np.array(test_data)
test_data = ((test_data.astype(np.float32) - 127.5) / 127.5)
test_labels =[vectorized_result(l) for l in test_labels]


mb_size = 32
X_dim = [28, 28, 1]
y_dim = 10
z_dim = 110
h_dim = 512
eps = 1e-8
G_lr = 1e-4
D_lr = 1e-3

X = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
y = tf.placeholder(tf.float32, shape=[None, y_dim])
z = tf.placeholder(tf.float32, shape=[None, z_dim])
condition = tf.placeholder(tf.int32, shape=[], name="condition")

G_W0 = tf.Variable(xavier_init([z_dim + y_dim, 1024]), name='gw0')
G_b0 = tf.Variable(tf.zeros(shape=[1024]), name='gb0')
G_W1 = tf.Variable(xavier_init([1024, 128 * 7 * 7]), name='gw1')
G_b1 = tf.Variable(tf.zeros(shape=[128 * 7 * 7]), name='gb1')
G_W2 = tf.Variable(xavier_init([5, 5, 256, 128]), name='gw2')
G_b2 = tf.Variable(tf.zeros([256]), name='gb2')
G_W3 = tf.Variable(xavier_init([5, 5, 128, 256]), name='gw3')
G_b3 = tf.Variable(tf.zeros([128]), name='gb3')
G_W4 = tf.Variable(xavier_init([2, 2, 1, 128]), name='gw4')
G_b4 = tf.Variable(tf.zeros(shape=[1]), name='gb4')


def generator(z, c):
    inputs = tf.concat(axis=1, values=[z, c])
    G_h0 = lrelu(tf.matmul(inputs, G_W0) + G_b0)
    G_h1 = lrelu(tf.matmul(G_h0, G_W1), G_b1)
    print 'shape of G_h1 before reshape:', G_h1.get_shape()
    G_h1 = lrelu(tf.reshape(G_h1, [-1, 7, 7, 128]))
    G_h1 = tf.contrib.layers.batch_norm(G_h1)
    print 'shape of G_h1 after reshape:', G_h1.get_shape()

    G_h2 = lrelu(tf.nn.bias_add(
        tf.nn.conv2d_transpose(G_h1, G_W2, output_shape=[mb_size, 7, 7, 256], strides=[1, 1, 1, 1], padding='SAME'),
        G_b2))
    print 'the shape of G_h2 :', G_h2.get_shape()
    G_h2 = tf.contrib.layers.batch_norm(G_h2)
    G_h3 = lrelu(tf.nn.bias_add(
        tf.nn.conv2d_transpose(G_h2, G_W3, output_shape=[mb_size, 14, 14, 128], strides=[1, 2, 2, 1], padding='SAME'),
        G_b3))
    print 'the shape of G_h3 :', G_h3.get_shape()
    G_h3 = tf.contrib.layers.batch_norm(G_h3)

    G_log_prob = tf.nn.bias_add(
        tf.nn.conv2d_transpose(G_h3, G_W4, output_shape=[mb_size, 28, 28, 1], strides=[1, 2, 2, 1], padding='SAME'),
        G_b4)
    G_prob = tf.nn.sigmoid(G_log_prob)
    return G_prob


D_W0 = tf.Variable(xavier_init([5, 5, 1, 32]), name = 'dw0')
D_b0 = tf.Variable(tf.zeros(shape=[32]), name='db0')
D_W1 = tf.Variable(xavier_init([5, 5, 32, 64]), name = 'dw1')
D_b1 = tf.Variable(tf.zeros(shape=[64]), name = 'db1')
D_W2 = tf.Variable(xavier_init([5, 5, 64, 128]), name = 'dw2')
D_b2 = tf.Variable(tf.zeros(shape=[128]), name = 'db2')
D_W3 = tf.Variable(xavier_init([5, 5, 128, 256]), name = 'dw3')
D_b3 = tf.Variable(tf.zeros([256]), name = 'db3')

D_W1_gan = tf.Variable(xavier_init([1024, 1]), name = 'dwgan')
D_b1_gan = tf.Variable(tf.zeros(shape=[1]), name = 'dbgan')
D_W1_aux = tf.Variable(xavier_init([1024, y_dim]), name = 'dwaux')
D_b1_aux = tf.Variable(tf.zeros(shape=[y_dim]), name ='dbaux')


def discriminator(X):
    D_h0 = lrelu(tf.nn.conv2d(X, D_W0, strides=[1, 2, 2, 1], padding='SAME') + D_b0)
    print 'shape of D_h0 :', D_h0.get_shape()
    D_h0 = tf.contrib.layers.batch_norm(D_h0)
    D_h1 = lrelu(tf.nn.conv2d(D_h0, D_W1, strides=[1, 2, 2, 1], padding='SAME') + D_b1)
    print 'shape of D_h1 :', D_h1.get_shape()
    D_h1 = tf.contrib.layers.batch_norm(D_h1)
    D_h2 = lrelu(tf.nn.conv2d(D_h1, D_W2, strides=[1, 2, 2, 1], padding='SAME') + D_b2)
    print 'shape of D_h2 :', D_h2.get_shape()
    D_h2 = tf.contrib.layers.batch_norm(D_h2)
    D_h3 = lrelu(tf.nn.conv2d(D_h2, D_W3, strides=[1, 2, 2, 1], padding='SAME') + D_b3)
    print 'shape of d_h3 :', D_h3.get_shape()
    D_h3 = tf.reshape(D_h3, [mb_size, -1])

    out_gan = tf.nn.sigmoid(tf.matmul(D_h3, D_W1_gan) + D_b1_gan)
    print 'shape of out_gan :', out_gan.get_shape()
    out_aux = tf.matmul(D_h3, D_W1_aux) + D_b1_aux
    print 'shape of out_aux :', out_aux.get_shape()
    return out_gan, out_aux


theta_G = [G_W0, G_W1, G_W2, G_W3, G_W4, G_b0, G_b1, G_b2, G_b3, G_b4]
theta_D = [D_W0, D_W1, D_W2, D_W3, D_W1_gan, D_W1_aux, D_b0, D_b1, D_b2, D_b3, D_b1_gan, D_b1_aux]


def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


def cross_entropy(logit, y):
    return -tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=y))

G_take = generator(z, y)
G_sample = ((G_take - 127.5) / 127.5)

print 'shape of generated images ', G_sample.get_shape()
D_real, C_real = discriminator(X)
D_fake, C_fake = discriminator(G_sample)

###################################################################### alternate gan function for nsupervise learning 

#D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=C_real,labels=tf.ones_like(C_real)))
#D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=C_fake,labels=tf.zeros_like(C_fake)))
#D_loss_gan = D_loss_real + D_loss_fake

#G_loss_gan = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=C_fake,labels=tf.ones_like(C_fake)))

#####################################################################


# Cross entropy aux loss
#C_loss = cross_entropy(C_real, y) + cross_entropy(C_fake, y)

# GAN D loss
D_loss = tf.reduce_mean(tf.log(D_real + eps) + tf.log(1. - D_fake + eps))

#DC_loss = -(D_loss + C_loss)

DC_loss = tf.cond(condition > 0, lambda: -(D_loss +(cross_entropy(C_real, y) + cross_entropy(C_fake, y))), lambda:D_loss)


tf.summary.scalar('DC_loss', DC_loss)

# GAN's G loss
G_loss = tf.reduce_mean(tf.log(D_fake + eps))

#GC_loss = -(G_loss + C_loss)

GC_loss = tf.cond(condition > 0, lambda: -(G_loss +(cross_entropy(C_real, y) + cross_entropy(C_fake, y))), lambda:G_loss)
tf.summary.scalar('GC_loss', GC_loss)

# Classification accuracy

correct_prediction = tf.equal(tf.argmax(C_real, 1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('DC_AUX accuracy',accuracy)


D_solver = (tf.train.AdamOptimizer(learning_rate=D_lr)
            .minimize(DC_loss, var_list=theta_D)) 
G_solver = (tf.train.AdamOptimizer(learning_rate=G_lr)
            .minimize(GC_loss, var_list=theta_G))



if not os.path.exists('output_ac_dcgan_MNIST/'):
    os.makedirs('output_ac_dcgan_MNIST/')

sess = tf.Session()
summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter('output_ac_dcgan_MNIST/logs', sess.graph)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=1)
saver.save(sess, 'output_ac_dcgan_MNIST/ac_dcgan_MNIST_model')

#for k in range(10):
 #    for v in range(64):
  #       cls = [k + 10 for p in range(mb_size)]
   #      cls = [vectorized_result(l,20) for l in cls]
    #     cls = np.array(cls)
     #    spls = sess.run(G_sample, feed_dict={z: sample_z(mb_size, z_dim), y: cls})
      #   training_labels.extend(cls)
       #  training_data.extend(spls)
#
i = 0
training_labels = np.array(training_labels)
training_data = np.array(training_data)
#training_data = ((training_data.astype(np.float32) - 127.5) / 127.5)
#validation_data= np.array(validation_data)
#validation_labels = np.array(validation_labels)
test_data = np.array(test_data)
test_labels = np.array(test_labels)

print 'the number of training samples : ', training_data.shape[0]
print 'the number of labels in ssl : ', validation_labels.shape[0]

for it in range(1000000):
    ind = np.random.choice(training_data.shape[0], mb_size)
    X_mb = np.array(training_data[ind])
    y_mb = np.array(training_labels[ind])#sample_z(mb_size,y_dim)#
    z_mb = sample_z(mb_size, z_dim)

    _, DC_loss_curr, acc, sum_op1 = sess.run([D_solver, DC_loss, accuracy, summary_op], feed_dict={X: X_mb, y: y_mb, z: z_mb, condition:1})
    summary_writer.add_summary(sum_op1, it)

    _, GC_loss_curr, sum_op2 = sess.run([G_solver, GC_loss, summary_op], feed_dict={X: X_mb, y: y_mb, z: z_mb, condition:1})
    summary_writer.add_summary(sum_op2, it)

    if it % 1000 == 0:
        idx = np.random.randint(0, 10)
        c = np.zeros([mb_size, y_dim])
        c[range(mb_size), idx] = 1

        samples = []
        for index in range(y_dim):
          s_level = np.zeros([mb_size, y_dim])
          s_level[range(mb_size), index] = 1
          samples.extend(sess.run(G_sample, feed_dict={z: sample_z(mb_size, z_dim), y: c, condition:1})[:10])

        print('Iter: {}; DC_loss: {:0.4}; GC_loss: {:0.4}; accuracy: {:0.4};'.format(it,DC_loss_curr, GC_loss_curr,acc))


        fig = plot(samples[:100])
        plt.savefig('output_ac_dcgan_MNIST/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)




accr = 0
k = 0
start_el = 0
for j in range(312):
    end_el = start_el + mb_size
    X_tstmb = np.array(test_data[start_el:end_el])
    y_tstmb = np.array(test_labels[start_el:end_el])
    z_tstmb = sample_z(mb_size, z_dim)
    start_el = end_el    

    accr += sess.run(accuracy, feed_dict={X:X_tstmb, y:y_tstmb, z:z_tstmb, condition:1})
    k +=1

print 'this is the final accuracy over 10k test samples : ', accr/312
f = open('output_ac_dcgan_MNIST/accuracy_1_fake_10k.txt','w')
f.write('finalaccuracy on 10k test samples with 1 fake classes : '+str(accr/312) )
f.close()