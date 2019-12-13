import tensorflow  as tf
from tensorflow.examples.tutorials.mnist import input_data

#variables setting
INPUT_NODE = 784
OUTPUT_NODE = 10

LAYEAR1_NODE = 500
BATCH_SIZE = 100

LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99

REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

def inference( input_tensor , avg_class , weights1 , biases1 , weights2 , biases2 ):
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights1)+biases1)
        return tf.matmul(layer1,weights2)+biases2 #没有softmax层
    else:
        layer1=tf.nn.relu( tf.matmul(input_tensor,avg_class.average(weights1))+ avg_class.average(biases1))
        return tf.matmul(layer1,avg_class.average(weights2))+ avg_class.average(biases2) #没有softmax层

def train(mnist):
    x = tf.placeholder(tf.float32,[None,INPUT_NODE],name='x-input')
    y_ = tf.placeholder(dtype=tf.float32,shape=[None,OUTPUT_NODE],name='y-input')

    #hiden-layer weights
    weights1=tf.Variable(tf.truncated_normal([INPUT_NODE,LAYEAR1_NODE],stddev=0.1))
    biases1=tf.Variable(tf.constant(0.1,shape=[LAYEAR1_NODE]))

    weights2=tf.Variable(tf.truncated_normal([LAYEAR1_NODE,OUTPUT_NODE],stddev=0.1))
    biases2=tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))

    #前向传播
    y=inference(x,None,weights1,biases1,weights2,biases2)

    #训练次数
    global_step=tf.Variable(0,trainable=False)

    #滑动平均 与 应用
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variables_average_op = variable_averages.apply(tf.trainable_variables())
    #滑动平均应用在前向传播
    average_y = inference(x,variable_averages,weights1,biases1,weights2,biases2)
    #y:预测值 y_:真值
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.arg_max(y_,1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # L2正则化
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    regularization = regularizer(weights1)+regularizer(weights2)
    loss = cross_entropy_mean+ regularization

    #learning rate
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,mnist.train.num_examples/BATCH_SIZE,LEARNING_RATE_DECAY)

    #训练模式
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    #上下文管理器 同时更新两个参数train_step,variables_average_op
    with tf.control_dependencies([train_step,variables_average_op]):
        train_op = tf.no_op(name='train')  #什么都不做

    #准确率判别
    correct_prediction = tf.equal(tf.argmax(average_y,1),tf.arg_max(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    #创建会话并开始训练
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        #验证
        validate_feed = {x:mnist.validation.images,y_:mnist.validation.labels}
        #测试
        test_feed = {x:mnist.test.images,y_:mnist.test.labels}

        for i in range(TRAINING_STEPS):
            if i%1000 == 0:
                validate_acc = sess.run(accuracy,feed_dict=validate_feed)
                print("train steps= %d,average accuracy= %g"%(i,validate_acc))

            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op,feed_dict={x:xs,y_:ys})

        test_acc= sess.run(accuracy,feed_dict=test_feed)
        print("final accuracy= %d"%(test_acc))

def main(argv=None):
        mnist = input_data.read_data_sets("/tmp/data",one_hot=True)
        train(mnist)

if __name__ == '__main__':
    tf.app.run()
































