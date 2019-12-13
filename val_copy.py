import tensorflow  as tf
import numpy as np
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


def excel_to_matrix(path,flag):
    table = xlrd.open_workbook(path).sheets()[0]#获取第一个sheet表
    row = table.nrows  # 行数
    col = table.ncols  # 列数
    datamatrix = np.zeros((row, col))#生成一个nrows行ncols列，且元素均为0的初始矩阵
    for x in range(col):
        cols = np.matrix(table.col_values(x))  # 把list转换为矩阵进行矩阵操作
        datamatrix[:, x] = cols
    if flag:
        return datamatrix/1000
    else:
        return datamatrix







# for each_col in range(all_sheet[0].ncols):
    #print("当前为%s列：" % each_col)
    #print(all_sheet[0].col_values(each_col ),type(all_sheet[0].col_values(each_col )))
# sheet_cell = all_sheet[0].cell(0, 0)  # 根据位置获取Cell对象
# print(sheet_cell)
# sheet_cell_value = sheet_cell.value  # 返回单元格的值
# print(sheet_cell_value)



val_ang_data = excel_to_matrix(val_ang_data_path,0)
val_pos_data = excel_to_matrix(val_pos_data_path,1)

train_ang_data = excel_to_matrix(train_ang_data_path,0)
train_pos_data = excel_to_matrix(train_pos_data_path,1)

# train_ang_data = excel_to_matrix(my_test_train,0)
# train_pos_data = excel_to_matrix(my_test_result,0)

#variables setting
INPUT_NODE = 24
OUTPUT_NODE = 3

LAYEAR1_NODE = 100
LAYEAR2_NODE = 300
LAYEAR3_NODE = 100
LAYEAR4_NODE = 100

BATCH_SIZE = 50

LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99

REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 50000
MOVING_AVERAGE_DECAY = 0.99



def inference( input_tensor , avg_class , weights1 , biases1 , weights2 , biases2,weights3 , biases3,weights4,biases4,weights5,biases5 ):
    layer1 = tf.nn.leaky_relu(tf.matmul(input_tensor,weights1)+biases1)
    layer2=tf.nn.leaky_relu(tf.matmul(layer1,weights2)+biases2)
    layer3=tf.nn.leaky_relu(tf.matmul(layer2,weights3)+biases3)
    layer4 = tf.nn.leaky_relu(tf.matmul(layer3, weights4) + biases4)
    resu=tf.matmul(layer4,weights5)+biases5#没有softmax层
    return resu

def train():
    x = tf.placeholder(tf.float32,[None,INPUT_NODE],name='x-input')
    y_ = tf.placeholder(dtype=tf.float32,shape=[None,OUTPUT_NODE],name='y-input')

    #hiden-layer weights
    weights1=tf.Variable(tf.truncated_normal([INPUT_NODE,LAYEAR1_NODE],stddev=0.1))
    biases1=tf.Variable(tf.constant(0.1,shape=[LAYEAR1_NODE]))

    weights2=tf.Variable(tf.truncated_normal([LAYEAR1_NODE,LAYEAR2_NODE],stddev=0.1))
    biases2=tf.Variable(tf.constant(0.1,shape=[LAYEAR2_NODE]))

    weights3 = tf.Variable(tf.truncated_normal([LAYEAR2_NODE, LAYEAR3_NODE], stddev=0.1))
    biases3 = tf.Variable(tf.constant(0.1, shape=[LAYEAR3_NODE]))

    weights4 = tf.Variable(tf.truncated_normal([LAYEAR3_NODE, LAYEAR4_NODE], stddev=0.1))
    biases4 = tf.Variable(tf.constant(0.1, shape=[LAYEAR4_NODE]))

    weights5 = tf.Variable(tf.truncated_normal([LAYEAR4_NODE, OUTPUT_NODE], stddev=0.1))
    biases5 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))


    #前向传播
    y=inference(x,None,weights1,biases1,weights2,biases2,weights3 , biases3,weights4,biases4,weights5,biases5 )
    # y = test_inference( x , weights1 , biases1 , weights2 , biases2 )
    cross_entropy = tf.reduce_mean(tf.reduce_sum(tf.square(y_-y), reduction_indices=[1]))


    #learning rate
    # learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,mnist.train.num_examples/BATCH_SIZE,LEARNING_RATE_DECAY)
    learning_rate=0.1
    #训练模式
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)


    #创建会话并开始训练
    with tf.Session() as sess:


        # # plot the real data
        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1)
        # ax.scatter(train_ang_data, train_pos_data)
        # plt.ion()
        # plt.show()

        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            sess.run(train_step, feed_dict={x: train_ang_data, y_: train_pos_data})
            if i%1000== 0:
                prediction = sess.run(y,feed_dict={x: train_ang_data})
                print(i)
                cc=[[y_[70],y[70],y_[150],y[150],y_[45],y[45]]]
                print(sess.run(cc,feed_dict={x: train_ang_data, y_: train_pos_data}))
                print(sess.run(cross_entropy,feed_dict={x: val_ang_data, y_: val_pos_data}))
                # print("train steps= %d,average accuracy= %d"%(i,validate_acc))
                # plot the prediction
                # print(validate_acc)
                # lines = ax.plot(train_ang_data, validate_acc, 'r-', lw=10)

                # fig2 = plt.figure()
                # ax = fig2.add_subplot(1, 1, 1)
                # ax.scatter(train_ang_data, prediction)
                # plt.ion()
                # plt.show()

                # plt.pause(1)
            # start=i*BATCH_SIZE%100
            # end=min(start+BATCH_SIZE,100)

        # test_acc= sess.run(accuracy,feed_dict=test_feed)
        # print("final accuracy= %g"%test_acc)

def main(argv=None):
        train()

if __name__ == '__main__':
    tf.app.run()































