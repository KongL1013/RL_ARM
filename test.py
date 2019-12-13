import tensorflow as tf
import numpy as np
# import xlsxwriter
# import xlrd
# def excel_to_matrix(path):
#     table = xlrd.open_workbook(path).sheets()[0]#获取第一个sheet表
#     row = table.nrows  # 行数
#     col = table.ncols  # 列数
#     datamatrix = np.zeros((row, col))#生成一个nrows行ncols列，且元素均为0的初始矩阵
#     for x in range(col):
#         cols = np.matrix(table.col_values(x))  # 把list转换为矩阵进行矩阵操作
#         datamatrix[:, x] = cols
#     return datamatrix/100
# test_pos_data_path='/home/dd/PycharmProjects/sim_arm_net/train-data/test_pos_data.xls'
#
#
# data=excel_to_matrix(test_pos_data_path)
# print(data[1:5])



# workbook = xlsxwriter.Workbook('/home/dd/PycharmProjects/sim_arm_net/testexcel/test_trian.xlsx')  #创建一个Excel文件
# worksheet = workbook.add_worksheet()               #创建一个sheet
# for i in range(0,300):
#     worksheet.write(i,0,2/300*i-1)
#
#
#
# workbook1 = xlsxwriter.Workbook('/home/dd/PycharmProjects/sim_arm_net/testexcel/test_result.xlsx')  #创建一个Excel文件
# worksheet1 = workbook1.add_worksheet()               #创建一个sheet
# for i in range(0,300):
#     dd=(2/300*i-1)*(2/300*i-1)
#     worksheet1.write(i,0,dd)
# workbook.close()
# workbook1.close()

#
# with tf.variable_scope("foo", reuse=None):
#     tf.get_variable("v", [1], initializer=tf.constant_initializer(1.0))
#     tf.get_variable("v2", [1,6], initializer=tf.ones_initializer())
#
#
# with tf.variable_scope("foo",reuse=True):
#     v1=tf.get_variable("v",[1])
#     print(tf.get_variable_scope().initializer)
#
# saver=tf.train.Saver()
# init_op=tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init_op)
#     saver.save(sess,'/home/dd/PycharmProjects/sim_arm_net/xjbtry/1.ckpt')
#

import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
# for i in range(1):
# fig=plt.figure()
# dc = np.random.RandomState(11)
# cc = dc.rand(5)
# plt.plot(cc)
# plt.show()
# print(cc)
def my_calss():
    class fa:
        def __init__(self):
            self.a = 1
            self.b = 2
            print("init fa")
        def do_fa(self):
            print("do_fa")

    class son(fa):
        def __init__(self,a =5,b =6):
            super(son,self).__init__()
            self.c = a
            self.d = b
            print("son_init")

    ee = son(7,8)
    print(ee.a)


def test1():
    dd = np.argmax([1,2,3])
    e = np.random.choice(5,size=2)
    print(e)
    cc =np.random.normal(1,2)
    print(cc)

if __name__ == '__main__':
    test1()








