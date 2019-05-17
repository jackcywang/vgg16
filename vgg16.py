import tensorflow as tf
import numpy as np
import pdb
import os
from datetime import datetime
from create_tf_record import *
import tensorflow.contrib.slim as slim

print("Tensorflow version:{}".format(tf.__version__))
labels_nums = 7  # 类别个数
batch_size = 1  #
resize_height = 224  # 指定存储图片高度
resize_width = 224  # 指定存储图片宽度
depths = 3
data_shape = [batch_size, resize_height, resize_width, depths]

# 定义input_images为图片数据
input_images = tf.placeholder(dtype=tf.float32, shape=[None, resize_height, resize_width, depths], name='input')
# 定义input_labels为labels数据
# input_labels = tf.placeholder(dtype=tf.int32, shape=[None], name='label')
input_labels = tf.placeholder(dtype=tf.int32, shape=[None, labels_nums], name='label')

# 定义dropout的概率
keep_prob = tf.placeholder(tf.float32,name='keep_prob')
is_training = tf.placeholder(tf.bool, name='is_training')
def net_evaluation(sess,loss,accuracy,val_images_batch,val_labels_batch,val_nums):
    val_max_steps = int(val_nums / batch_size)
    val_losses = []
    val_accs = []
    for _ in range(val_max_steps):
        val_x, val_y = sess.run([val_images_batch, val_labels_batch])
        # print('labels:',val_y)
        # val_loss = sess.run(loss, feed_dict={x: val_x, y: val_y, keep_prob: 1.0})
        # val_acc = sess.run(accuracy,feed_dict={x: val_x, y: val_y, keep_prob: 1.0})
        val_loss,val_acc = sess.run([loss,accuracy], feed_dict={input_images: val_x, input_labels: val_y, keep_prob:1.0, is_training: False})
        val_losses.append(val_loss)
        val_accs.append(val_acc)
    mean_loss = np.array(val_losses, dtype=np.float32).mean()
    mean_acc = np.array(val_accs, dtype=np.float32).mean()
    return mean_loss, mean_acc

class Vgg16:
    vgg_mean = [103.939, 116.779, 123.68]

    def __init__(self, vgg16_npy_path=None,input=None, restore_from=None):
        # pre-trained parameters
        try:
            self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        except FileNotFoundError:
            print('Please download VGG16 parameters from here https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM\nOr from my Baidu Cloud: https://pan.baidu.com/s/1Spps1Wy0bvrQHH2IMkRfpg')

        # self.tfx = tf.placeholder(tf.float32, [None, 224, 224, 3])
        self.sess = tf.Session()
        self.tfx = input
        self.tfy = tf.placeholder(tf.float32, [None, 1])

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=self.tfx * 255.0)
        bgr = tf.concat(axis=3, values=[
            blue - self.vgg_mean[0],
            green - self.vgg_mean[1],
            red - self.vgg_mean[2],
        ])

        # pre-trained VGG layers are fixed in fine-tune
        conv1_1 = self.conv_layer(bgr, "conv1_1")
        conv1_2 = self.conv_layer(conv1_1, "conv1_2")
        pool1 = self.max_pool(conv1_2, 'pool1')

        conv2_1 = self.conv_layer(pool1, "conv2_1")
        conv2_2 = self.conv_layer(conv2_1, "conv2_2")
        pool2 = self.max_pool(conv2_2, 'pool2')

        conv3_1 = self.conv_layer(pool2, "conv3_1")
        conv3_2 = self.conv_layer(conv3_1, "conv3_2")
        conv3_3 = self.conv_layer(conv3_2, "conv3_3")
        pool3 = self.max_pool(conv3_3, 'pool3')

        conv4_1 = self.conv_layer(pool3, "conv4_1")
        conv4_2 = self.conv_layer(conv4_1, "conv4_2")
        conv4_3 = self.conv_layer(conv4_2, "conv4_3")
        pool4 = self.max_pool(conv4_3, 'pool4')

        conv5_1 = self.conv_layer(pool4, "conv5_1")
        conv5_2 = self.conv_layer(conv5_1, "conv5_2")
        conv5_3 = self.conv_layer(conv5_2, "conv5_3")
        pool5 = self.max_pool(conv5_3, 'pool5')

        # detach original VGG fc layers and
        # reconstruct your own fc layers serve for your own purpose
        pool5_shape = pool5.get_shape().as_list()
        nodes = pool5_shape[1] * pool5_shape[2] * pool5_shape[3]
        self.flatten = tf.reshape(pool5, [-1, nodes])
        self.fc6 = tf.layers.dense(self.flatten, 256, tf.nn.relu, name='fc6')
        self.out = tf.layers.dense(self.fc6, 7, name='out')

        # self.sess = tf.Session()
        # if restore_from:
        #     saver = tf.train.Saver()
        #     saver.restore(self.sess, restore_from)
        # else:   # training graph
        #     self.loss = tf.losses.mean_squared_error(labels=self.tfy, predictions=self.out)
        #     self.train_op = tf.train.RMSPropOptimizer(0.001).minimize(self.loss)
        #     self.sess.run(tf.global_variables_initializer(

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):   # CNN's filter is constant, NOT Variable that can be trained
            conv = tf.nn.conv2d(bottom, self.data_dict[name][0], [1, 1, 1, 1], padding='SAME')
            lout = tf.nn.relu(tf.nn.bias_add(conv, self.data_dict[name][1]))
            return lout

    def train(self, x, y):
        loss, _ = self.sess.run([self.loss, self.train_op], {self.tfx: x, self.tfy: y})
        return loss

    # def predict(self, paths):
    #     fig, axs = plt.subplots(1, 2)
    #     for i, path in enumerate(paths):
    #         x = load_img(path)
    #         length = self.sess.run(self.out, {self.tfx: x})
    #         axs[i].imshow(x[0])
    #         axs[i].set_title('Len: %.1f cm' % length)
    #         axs[i].set_xticks(()); axs[i].set_yticks(())
    #     plt.show()

    def save(self, path='./model/'):
        saver = tf.train.Saver()
        saver.save(self.sess, path, write_meta_graph=False)

def train(train_record_file,
          train_log_step,
          train_param,
          val_record_file,
          val_log_step,
          labels_nums,
          data_shape,
          snapshot,
          snapshot_prefix):
    '''
    :param train_record_file: 训练的tfrecord文件
    :param train_log_step: 显示训练过程log信息间隔
    :param train_param: train参数
    :param val_record_file: 验证的tfrecord文件
    :param val_log_step: 显示验证过程log信息间隔
    :param val_param: val参数
    :param labels_nums: labels数
    :param data_shape: 输入数据shape
    :param snapshot: 保存模型间隔
    :param snapshot_prefix: 保存模型文件的前缀名
    :return:
    '''
    [base_lr,max_steps]=train_param
    [batch_size,resize_height,resize_width,depths]=data_shape

    # 获得训练和测试的样本数
    train_nums=get_example_nums(train_record_file)
    val_nums=get_example_nums(val_record_file)
    print('train nums:%d,val nums:%d'%(train_nums,val_nums))

    # 从record中读取图片和labels数据
    # train数据,训练数据一般要求打乱顺序shuffle=True
    train_images, train_labels = read_records(train_record_file, resize_height, resize_width, type='normalization')
    train_images_batch, train_labels_batch = get_batch_images(train_images, train_labels,
                                                              batch_size=batch_size, labels_nums=labels_nums,
                                                              one_hot=True, shuffle=False)
    # val数据,验证数据可以不需要打乱数据
    val_images, val_labels = read_records(val_record_file, resize_height, resize_width, type='normalization')
    val_images_batch, val_labels_batch = get_batch_images(val_images, val_labels,
                                                          batch_size=batch_size, labels_nums=labels_nums,
                                                          one_hot=True, shuffle=False)

    # Define the model:
    # with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
    #     out, end_points = inception_v3.inception_v3(inputs=input_images, num_classes=labels_nums, dropout_keep_prob=keep_prob, is_training=is_training)
    vgg = Vgg16(vgg16_npy_path='./vgg16.npy',input=input_images)
    out = vgg.out
    # Specify the loss function: tf.losses定义的loss函数都会自动添加到loss函数,不需要add_loss()了
    tf.losses.softmax_cross_entropy(onehot_labels=input_labels, logits=out)#添加交叉熵损失loss=1.6
    # slim.losses.add_loss(my_loss)
    loss = tf.losses.get_total_loss(add_regularization_losses=True)#添加正则化损失loss=2.2
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(input_labels, 1)), tf.float32))
    # Specify the optimization scheme:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=base_lr)
    train_op = slim.learning.create_train_op(total_loss=loss,optimizer=optimizer)

    # global_step = tf.Variable(0, trainable=False)
    # learning_rate = tf.train.exponential_decay(0.05, global_step, 150, 0.9)
    #
    # optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    # # train_op = optimizer.minimize(loss, global_step)
    # train_op = slim.learning.create_train_op(loss, optimizer,global_step=global_step)



    saver = tf.train.Saver()
    max_acc=0.0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(max_steps+1):
            batch_input_images, batch_input_labels = sess.run([train_images_batch, train_labels_batch])
            _, train_loss = sess.run([train_op, loss], feed_dict={input_images:batch_input_images,
                                                                      input_labels:batch_input_labels,
                                                                      keep_prob:0.5, is_training:True})
            # train测试(这里仅测试训练集的一个batch)
            if i%train_log_step == 0:
                train_acc = sess.run(accuracy, feed_dict={input_images:batch_input_images,
                                                          input_labels: batch_input_labels,
                                                          keep_prob:1.0, is_training: False})
                print("%s: Step [%d]  train Loss : %f, training accuracy :  %g" % (datetime.now(), i, train_loss, train_acc))

            # val测试(测试全部val数据)
            if i%val_log_step == 0:
                mean_loss, mean_acc=net_evaluation(sess, loss, accuracy, val_images_batch, val_labels_batch,val_nums)
                print("%s: Step [%d]  val Loss : %f, val accuracy :  %g" % (datetime.now(), i, mean_loss, mean_acc))

            # 模型保存:每迭代snapshot次或者最后一次保存模型
            if (i %snapshot == 0 and i >0)or i == max_steps:
                print('-----save:{}-{}'.format(snapshot_prefix,i))
                saver.save(sess, snapshot_prefix, global_step=i)
            # 保存val准确率最高的模型
            if mean_acc>max_acc and mean_acc>0.5:
                max_acc=mean_acc
                path = os.path.dirname(snapshot_prefix)
                best_models=os.path.join(path,'best_models_{}_{:.4f}.ckpt'.format(i,max_acc))
                print('------save:{}'.format(best_models))
                saver.save(sess, best_models)

        coord.request_stop()
        coord.join(threads)



if __name__ == '__main__':
    train_record_file='dataset/record/train224.tfrecords'
    val_record_file='dataset/record/val224.tfrecords'

    train_log_step=100
    base_lr = 0.01  # 学习率
    max_steps = 10000  # 迭代次数
    train_param=[base_lr,max_steps]

    val_log_step=200
    snapshot=2000#保存文件间隔
    snapshot_prefix='./models/model.ckpt'
    train(train_record_file=train_record_file,
          train_log_step=train_log_step,
          train_param=train_param,
          val_record_file=val_record_file,
          val_log_step=val_log_step,
          labels_nums=labels_nums,
          data_shape=data_shape,
          snapshot=snapshot,
          snapshot_prefix=snapshot_prefix)
