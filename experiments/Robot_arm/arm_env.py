"""
Environment for Robot Arm.
You can customize this script in a way you want.

View more on [莫烦Python] : https://morvanzhou.github.io/tutorials/


Requirement:
pyglet >= 1.2.4
numpy >= 1.12.1
"""
import numpy as np
import pyglet
import time
pyglet.clock.set_fps_limit(10000)


width = 1100
height = width

arm_length=100
class ArmEnv(object):
    action_bound = [-1, 1]
    action_dim = 8  #两个关节
    state_dim = 1+ action_dim*2+2 # 7个观测值
    dt = .1  # refresh rate  转动时间
    arm1l = arm_length
    arm2l = arm_length
    viewer = None
    viewer_xy = (width, height)
    get_point = False
    mouse_in = np.array([False])
    point_l = 30

    def __init__(self, mode='easy'):
        # node1 (l, d_rad, x, y),
        # node2 (l, d_rad, x, y)
        self.mode = mode
        self.arm_info = np.zeros((self.action_dim, 4))
        # self.arm_info[0, 0] = self.arm1l
        # self.arm_info[1, 0] = self.arm2l
        self.point_info = np.array([250, 303])
        self.point_info_init = self.point_info.copy()
        self.center_coord = np.array(self.viewer_xy)/2
        self.arm_num = self.action_dim
        self.grab_counter = 0
        self.ep = 0

    def step(self, action):

        #所有关节转角
        # action = (node1 angular v, node2 angular v)
        action = np.clip(action, *self.action_bound)  #np.clip 截到一点范围
        self.arm_info[:, 1] += action * self.dt  #转多少  转速×时间
        self.arm_info[:, 1] %= np.pi * 2 #超出360°，取余数

        arm1rad = self.arm_info[0, 1] #1的转角
        # arm2rad = self.arm_info[1, 1] #2的转角
        arm1dx_dy = np.array([self.arm_info[0, 0] * np.cos(arm1rad), self.arm_info[0, 0] * np.sin(arm1rad)]) #相对坐标
        # arm2dx_dy = np.array([self.arm_info[1, 0] * np.cos(arm2rad), self.arm_info[1, 0] * np.sin(arm2rad)])
        self.arm_info[0, 2:4] = self.center_coord + arm1dx_dy  # (x1, y1) ，末段绝对坐标
        # self.arm_info[1, 2:4] = self.arm_info[0, 2:4] + arm2dx_dy  # (x2, y2)

        for i in range(1,self.arm_num):
            self.cal_arm_info(i)
        s, arm_lat_distance = self._get_state()
        r = self._r_func(arm_lat_distance)

        return s, r, self.get_point


    def cal_arm_info(self,i):

        arm_i_rad = self.arm_info[i,1]
        self.arm_info[i, 0]= arm_length
        arm_i_dx_dy = np.array([self.arm_info[i, 0] * np.cos(arm_i_rad), self.arm_info[i, 0] * np.sin(arm_i_rad)])
        self.arm_info[i,2:4] = self.arm_info[i-1,2:4]+arm_i_dx_dy




    def reset(self,ep): #每次产生随机转角
        self.ep = ep
        self.get_point = False
        self.grab_counter = 0

        if self.mode == 'hard':
            flip = 10
            aa = int(width/flip) + int(width/flip*2)*np.random.rand(2)
            bb = int(width/flip*5+width/flip)+int(width/flip*2)*np.random.rand(2)
            cc = [*aa,*bb]
            pxy = np.random.choice(cc,2)
            self.point_info[:] = pxy
        else:
            arm_n_rad= np.random.rand(arm_num) * np.pi * self.arm_num   #两个初始随机转角
            for i in range(arm_num):
                self.arm_info[i,1] = arm_n_rad[i]
            # self.arm_info[0, 1] = arm1rad
            # self.arm_info[1, 1] = arm2rad
            arm1dx_dy = np.array([self.arm_info[0, 0] * np.cos(arm1rad), self.arm_info[0, 0] * np.sin(arm1rad)]) #末段x，y相对于初始位置坐标
            # arm2dx_dy = np.array([self.arm_info[1, 0] * np.cos(arm2rad), self.arm_info[1, 0] * np.sin(arm2rad)])
            self.arm_info[0, 2:4] = self.center_coord + arm1dx_dy  # (x1, y1) L1 末端

            for i in range(1,self.arm_num):
                self.cal_arm_info(i)


            self.point_info[:] = self.point_info_init
        return self._get_state()[0]

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(*self.viewer_xy, self.arm_info, self.point_info, self.point_l, self.mouse_in,self.arm_num)
        self.viewer.render()

    def sample_action(self):
        return np.random.uniform(*self.action_bound, size=self.action_dim)
    def set_fps(self, fps=30):
        pyglet.clock.set_fps_limit(fps)

    def _get_state(self): #手臂和目标点的距离
        # return the distance (dx, dy) between arm finger point with blue point
        arm_end = self.arm_info[:, 2:4]

        t_arms = np.ravel(arm_end - self.point_info)  #self.point_info 目标点，即蓝色中心点,这个算的是每一段的末段点距离目标点的位置
        center_dis = (self.center_coord - self.point_info)/(width/2)
        in_point = 1 if self.grab_counter > 0 else 0   #是否到达目标点
        current_state = np.hstack([in_point, t_arms/(width/2), center_dis, #
                          # arm1_distance_p, arm1_distance_b,
                          ])
        # print('current_state',current_state.shape)   27
        return current_state, t_arms[-2:]  #最后一个点的距离

    def _r_func(self, distance):
        if self.ep >700 :
            t = 50  #停留次数，防止学会甩过去
        elif self.ep<=700 and self.ep>400:
            t = 35
        else:
            t = 20
        abs_distance = np.sqrt(np.sum(np.square(distance)))
        r = -abs_distance/(width/2)
        if abs_distance < self.point_l and (not self.get_point):
            r += 1.
            self.grab_counter += 1
            if self.grab_counter > t:
                r += 10.
                self.get_point = True
        elif abs_distance > self.point_l: #此时counter重新赋值为0，也就是说必须要求连续到达目标点次数>t，才算
            self.grab_counter = 0
            self.get_point = False
        return r


class Viewer(pyglet.window.Window):  #单独可视化
    color = {
        'background': [1]*3 + [1]
    }
    fps_display = pyglet.clock.ClockDisplay()
    bar_thc = 5

    def __init__(self, width, height, arm_info, point_info, point_l, mouse_in ,arm_num):
        super(Viewer, self).__init__(width, height, resizable=False, caption='Arm', vsync=False)  # vsync=False to not use the monitor FPS
        self.set_location(x=200, y=100)
        pyglet.gl.glClearColor(*self.color['background'])
        self.arm_num = arm_num


        self.point_info = point_info #目标点



        self.arm_info =arm_info
        self.arm_info[0, 1]= 0
        self.arm_info[0, 0] = arm_length
        arm1rad = self.arm_info[0, 1]  # 1的转角
        arm1dx_dy = np.array([self.arm_info[0, 0] * np.cos(arm1rad), self.arm_info[0, 0] * np.sin(arm1rad)])  # 相对坐标
        self.arm_info[0, 2:4] = [height/2,width/2] + arm1dx_dy  # (x1, y1) ，末段绝对坐标

        for i in range(1,arm_num):
            self.arm_info[i,0] = arm_length
            self.arm_info[i,1] = np.random.rand(1)*np.pi/2
            self.cal_arm_info(i)


        self.mouse_in = mouse_in
        self.point_l = point_l

        self.center_coord = np.array((min(width, height)/2, ) * 2)
        self.batch = pyglet.graphics.Batch()  #构件  全部加到render
        self.arm_box = []
        # arm1_box, arm2_box, point_box = [0]*8, [0]*8, [0]*8
        point_box = [0]*8
        for i in range(self.arm_num):
            self.arm_box.append(np.ravel(([0]*4,[1]*4)))

        # self.arm_box.append( [50, 50,                # x1, y1
        #              50, 100,               # x2, y2
        #              100, 100,              # x3, y3
        #              100, 50])
        #
        # self.arm_box.append([250, 250,              # 同上, 点信息
        #              250, 300,
        #              260, 300,
        #              260, 250])


        c1, c2, c3 = (249, 86, 86)*4, (86, 109, 249)*4, (249, 39, 65)*4  #rgb color
        self.arm = []
        self.point = self.batch.add(4, pyglet.gl.GL_QUADS, None, ('v2f', point_box), ('c3B', c2))  #机械臂要达到的点   'c3B', c2颜色

        for i in range(self.arm_num):
            self.arm.append(self.batch.add(4, pyglet.gl.GL_QUADS, None, ('v2f', self.arm_box[i]), ('c3B', c1)))

        # self.arm.append(self.batch.add(4, pyglet.gl.GL_QUADS, None, ('v2f', arm1_box), ('c3B', c1)))
        # self.arm.append(self.batch.add(4, pyglet.gl.GL_QUADS, None, ('v2f', arm2_box), ('c3B', c1)) ) #4 corners  v2f 是机械臂（长方形）每个点的 x, y 信息 4*2=8

    def cal_arm_info(self,i):

        arm_i_rad = self.arm_info[i,1]
        arm_i_dx_dy = np.array([self.arm_info[0, 0] * np.cos(arm_i_rad), self.arm_info[0, 0] * np.sin(arm_i_rad)])
        self.arm_info[i,2:4] = self.arm_info[i-1,2:4]+arm_i_dx_dy


    def render(self):  #必备
        pyglet.clock.tick()
        self._update_arm()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):  #Viewer必备  # 刷新手臂等位置
        self.clear()
        self.batch.draw()
        # self.fps_display.draw()

    def _update_arm(self):  # 更新手臂的位置信息
        point_l = self.point_l  #把point_l 继承过来的点的信息转化为画图的信息  ，目标点矩形宽度的一半
        point_box = (self.point_info[0] - point_l, self.point_info[1] - point_l,
                     self.point_info[0] + point_l, self.point_info[1] - point_l,
                     self.point_info[0] + point_l, self.point_info[1] + point_l,
                     self.point_info[0] - point_l, self.point_info[1] + point_l)  #构造目标点矩形
        self.point.vertices = point_box

        arm1_coord = (*self.center_coord, *(self.arm_info[0, 2:4]))  # (x0, y0, x1, y1)  机械臂首末两点   机械臂每个坐标点都计算出来了
        # arm2_coord = (*(self.arm_info[0, 2:4]), *(self.arm_info[1, 2:4]))  # (x1, y1, x2, y2)
        arm1_thick_rad = np.pi / 2 - self.arm_info[0, 1]
        x01, y01 = arm1_coord[0] - np.cos(arm1_thick_rad) * self.bar_thc, arm1_coord[1] + np.sin(
            arm1_thick_rad) * self.bar_thc
        x02, y02 = arm1_coord[0] + np.cos(arm1_thick_rad) * self.bar_thc, arm1_coord[1] - np.sin(
            arm1_thick_rad) * self.bar_thc
        x11, y11 = arm1_coord[2] + np.cos(arm1_thick_rad) * self.bar_thc, arm1_coord[3] - np.sin(
            arm1_thick_rad) * self.bar_thc
        x12, y12 = arm1_coord[2] - np.cos(arm1_thick_rad) * self.bar_thc, arm1_coord[3] + np.sin(
            arm1_thick_rad) * self.bar_thc
        self.arm_box[0] = (x01, y01, x02, y02, x11, y11, x12, y12)

        # arm2_thick_rad = np.pi / 2 - self.arm_info[1, 1]
        # x11_, y11_ = arm2_coord[0] + np.cos(arm2_thick_rad) * self.bar_thc, arm2_coord[1] - np.sin(
        #     arm2_thick_rad) * self.bar_thc
        # x12_, y12_ = arm2_coord[0] - np.cos(arm2_thick_rad) * self.bar_thc, arm2_coord[1] + np.sin(
        #     arm2_thick_rad) * self.bar_thc
        # x21, y21 = arm2_coord[2] - np.cos(arm2_thick_rad) * self.bar_thc, arm2_coord[3] + np.sin(
        #     arm2_thick_rad) * self.bar_thc
        # x22, y22 = arm2_coord[2] + np.cos(arm2_thick_rad) * self.bar_thc, arm2_coord[3] - np.sin(
        #     arm2_thick_rad) * self.bar_thc  # self.bar_thc 机械臂一半宽度
        # arm2_box = (x11_, y11_, x12_, y12_, x21, y21, x22, y22)
        for i in range(1,self.arm_num):
            self.cal_update_armbox(i)

        for i in range(self.arm_num):
            self.arm[i].vertices = self.arm_box[i]
        # self.arm[0].vertices = arm1_box
        # self.arm[1].vertices = arm2_box

    def cal_update_armbox(self,i):
        armi_thick_rad = np.pi / 2 - self.arm_info[i, 1]

        # armi_thick_rad = np.pi / 2 - np.random.rand(1)*np.pi

        arm2_coord =  (*(self.arm_info[i-1, 2:4]), *(self.arm_info[i, 2:4]))
        x11_, y11_ = arm2_coord[0] + np.cos(armi_thick_rad) * self.bar_thc, arm2_coord[1] - np.sin(
            armi_thick_rad) * self.bar_thc
        x12_, y12_ = arm2_coord[0] - np.cos(armi_thick_rad) * self.bar_thc, arm2_coord[1] + np.sin(
            armi_thick_rad) * self.bar_thc
        x21, y21 = arm2_coord[2] - np.cos(armi_thick_rad) * self.bar_thc, arm2_coord[3] + np.sin(
            armi_thick_rad) * self.bar_thc
        x22, y22 = arm2_coord[2] + np.cos(armi_thick_rad) * self.bar_thc, arm2_coord[3] - np.sin(
            armi_thick_rad) * self.bar_thc  # self.bar_thc 机械臂一半宽度
        self.arm_box[i] = (x11_, y11_, x12_, y12_, x21, y21, x22, y22)



    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.UP:
            self.arm_info[0, 1] += .1
            print(self.arm_info[:, 2:4] - self.point_info)
        elif symbol == pyglet.window.key.DOWN:
            self.arm_info[0, 1] -= .1
            print(self.arm_info[:, 2:4] - self.point_info)
        elif symbol == pyglet.window.key.LEFT:
            self.arm_info[1, 1] += .1
            print(self.arm_info[:, 2:4] - self.point_info)
        elif symbol == pyglet.window.key.RIGHT:
            self.arm_info[1, 1] -= .1
            print(self.arm_info[:, 2:4] - self.point_info)
        elif symbol == pyglet.window.key.Q:
            pyglet.clock.set_fps_limit(1000)
        elif symbol == pyglet.window.key.A:
            pyglet.clock.set_fps_limit(30)

    def on_mouse_motion(self, x, y, dx, dy):
        self.point_info[:] = [x, y]

    def on_mouse_enter(self, x, y):
        self.mouse_in[0] = True

    def on_mouse_leave(self, x, y):
        self.mouse_in[0] = False

if __name__ == '__main__':
    env=ArmEnv()
    while True:
        env.render()
        a = env.sample_action()
        env.step(a)
        # time.sleep(0.5)

        # print('arm_end', env.arm_info[:, 2:4])
        # print("self.point_info", env.point_info)

