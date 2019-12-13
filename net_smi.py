import tensorflow  as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import xlrd
import matplotlib.pyplot as plt

my_test_train='/home/dd/PycharmProjects/sim_arm_net/testexcel/test_trian.xlsx'
my_test_result='/home/dd/PycharmProjects/sim_arm_net/testexcel/test_result.xlsx'

train_ang_data_path='/home/dd/PycharmProjects/sim_arm_net/train-data/train_ang_data.xls'
train_pos_data_path='/home/dd/PycharmProjects/sim_arm_net/train-data/train_pos_data.xls'

val_ang_data_path='/home/dd/PycharmProjects/sim_arm_net/train-data/val_ang_data.xls'
val_pos_data_path='/home/dd/PycharmProjects/sim_arm_net/train-data/val_pos_data.xls'

test_ang_data_path='/home/dd/PycharmProjects/sim_arm_net/train-data/test_ang_data.xls'
test_pos_data_path='/home/dd/PycharmProjects/sim_arm_net/train-data/test_pos_data.xls'

# train_ang_data=xlrd.open_workbook(train_ang_data_path,encoding_override="utf-8").sheets()
# train_pos_data=xlrd.open_workbook(train_pos_data_path,encoding_override="utf-8").sheets()
#
# val_ang_data=xlrd.open_workbook(val_ang_data_path,encoding_override="utf-8").sheets()
# val_pos_data=xlrd.open_workbook(val_pos_data_path,encoding_override="utf-8").sheets()
#
# test_ang_data=xlrd.open_workbook(test_ang_data_path,encoding_override="utf-8").sheets()
# test_pos_data=xlrd.open_workbook(test_pos_data_path,encoding_override="utf-8").sheets()


# excel = xlrd.open_workbook(test_pos_data_path,encoding_override="utf-8")
# all_sheet = excel.sheets()

# sheet_cell_value_1 = test_pos_data[0].cell_value(10, 0)  # 根据位置获取单元值Cell对象的值
# print(sheet_cell_value_1)


def excel_to_matrix(path):
    table = xlrd.open_workbook(path).sheets()[0]#获取第一个sheet表
    row = table.nrows  # 行数
    col = table.ncols  # 列数
    datamatrix = np.zeros((row, col))#生成一个nrows行ncols列，且元素均为0的初始矩阵
    for x in range(col):
        cols = np.matrix(table.col_values(x))  # 把list转换为矩阵进行矩阵操作
        datamatrix[:, x] = cols
    return datamatrix







# for each_col in range(all_sheet[0].ncols):
    #print("当前为%s列：" % each_col)
    #print(all_sheet[0].col_values(each_col ),type(all_sheet[0].col_values(each_col )))
# sheet_cell = all_sheet[0].cell(0, 0)  # 根据位置获取Cell对象
# print(sheet_cell)
# sheet_cell_value = sheet_cell.value  # 返回单元格的值
# print(sheet_cell_value)


#variables setting
INPUT_NODE = 1
OUTPUT_NODE = 1

LAYEAR1_NODE = 3
LAYEAR2_NODE = 2


BATCH_SIZE = 50

LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99

REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 3000
MOVING_AVERAGE_DECAY = 0.99

def inference( input_tensor , avg_class , weights1 , biases1 , weights2 , biases2,weights3 , biases3 ):
    if avg_class == None:
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights1)+biases1)
        layer2=(tf.matmul(layer1,weights2)+biases2)
        resu=tf.matmul(layer2,weights3)+biases3#没有softmax层

    else:
        # layer1=tf.nn.relu( tf.matmul(input_tensor,avg_class.average(weights1))+ avg_class.average(biases1))
        # layer2=tf.nn.relu( tf.matmul(layer1,avg_class.average(weights2))+ avg_class.average(biases2))
        # resu=tf.matmul(layer2,avg_class.average(weights3))+ avg_class.average(biases3) #没有softmax层
        layer1 = (tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        layer2 = (tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2))
        resu = tf.matmul(layer2, avg_class.average(weights3)) + avg_class.average(biases3)  # 没有softmax层
    return resu

def train(mnist):
    x = tf.placeholder(tf.float32,[None,INPUT_NODE],name='x-input')
    y_ = tf.placeholder(dtype=tf.float32,shape=[None,OUTPUT_NODE],name='y-input')

    #hiden-layer weights
    weights1=tf.Variable(tf.truncated_normal([INPUT_NODE,LAYEAR1_NODE],stddev=0.1))
    biases1=tf.Variable(tf.constant(0.1,shape=[LAYEAR1_NODE]))

    weights2=tf.Variable(tf.truncated_normal([LAYEAR1_NODE,LAYEAR2_NODE],stddev=0.1))
    biases2=tf.Variable(tf.constant(0.1,shape=[LAYEAR2_NODE]))

    weights3 = tf.Variable(tf.truncated_normal([LAYEAR2_NODE, OUTPUT_NODE], stddev=0.1))
    biases3 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))


    #前向传播
    y=inference(x,None,weights1,biases1,weights2,biases2,weights3 , biases3 )

    #训练次数
    global_step=tf.Variable(0,trainable=False)

    #滑动平均 与 应用
    # variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    # variables_average_op = variable_averages.apply(tf.trainable_variables())
    #滑动平均应用在前向传播
    # average_y = inference(x,variable_averages,weights1,biases1,weights2,biases2,weights3 , biases3 )
    #y:预测值 y_:真值
    cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.square(y_-y), reduction_indices=[1]))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # L2正则化WWWW
    # regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # regularization = regularizer(weights1)+regularizer(weights2)
    # loss = cross_entropy_mean+ regularization

    #learning rate
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,mnist.train.num_examples/BATCH_SIZE,LEARNING_RATE_DECAY)

    #训练模式
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy,global_step=global_step)
    #上下文管理器 同时更新两个参数train_step,variables_average_op
    # with tf.control_dependencies([train_step,variables_average_op]):
    #     train_op = tf.no_op(name='train')  #什么都不做

    #准确率判别
    # correct_prediction = tf.equal(tf.argmax(average_y,1),tf.arg_max(y_,1))
    correct_prediction = tf.reduce_mean(tf.reduce_sum(tf.square(y_-y), reduction_indices=[1]))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    #创建会话并开始训练
    with tf.Session() as sess:
        train_ang_data = excel_to_matrix(my_test_train)
        train_pos_data = excel_to_matrix(my_test_result)

        val_ang_data = excel_to_matrix(val_ang_data_path)
        val_pos_data = excel_to_matrix(val_pos_data_path)

        test_ang_data = excel_to_matrix(test_ang_data_path)
        test_pos_data = excel_to_matrix(test_pos_data_path)
        # plot the real data
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(train_ang_data, train_pos_data)
        plt.ion()
        plt.show()
        tf.global_variables_initializer().run()
        #验证
        validate_feed = {x:train_ang_data,y_:train_pos_data}
        #测试
        test_feed = {x:test_ang_data,y_:test_pos_data}

        for i in range(TRAINING_STEPS):
            sess.run(train_step, feed_dict={x: train_ang_data, y_: train_pos_data})
            if i%50 == 0:
                validate_acc = sess.run(y,feed_dict=validate_feed)
                print(i)
                print(sess.run(weights1))
                # print("train steps= %d,average accuracy= %d"%(i,validate_acc))
                # plot the prediction
                # print(validate_acc)
                # lines = ax.plot(train_ang_data, validate_acc, 'r-', lw=10)

                fig2 = plt.figure()
                ax = fig2.add_subplot(1, 1, 1)
                ax.scatter(train_ang_data, validate_acc)
                plt.ion()
                plt.show()

                # plt.pause(1)

            # start=i*BATCH_SIZE%100
            # end=min(start+BATCH_SIZE,100)


        # test_acc= sess.run(accuracy,feed_dict=test_feed)
        # print("final accuracy= %g"%test_acc)

def main(argv=None):
        mnist = input_data.read_data_sets("/tmp/data",one_hot=True)
        train(mnist)

if __name__ == '__main__':
    tf.app.run()































