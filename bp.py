#BP NetWork
import random
import math
import numpy as np
from sklearn.datasets import make_moons

#tool function
def rand(a, b):
    return (b - a) * random.random() + a

#创建一个指定大小的矩阵
def make_matrix(m, n, fill=0.0):
    mat = []
    for i in range(m):
        mat.append([fill] * n)
    return mat

def make_vector(n):
    mat = []
    for i in range(n):
        mat.append(random.random())
    return mat

#定义sigmoid函数和它的导数
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_derivate(x):
    return x * (1 - x)


#定义NeuralBPNetwork类
class BPNeuralNetwork:
    def __init__(self):
        self.input_number       = 0
        self.hidden_number      = 0
        self.output_number      = 0
        self.hidden_bias        = [] #隐含层权值,1-d
        self.output_bias        = [] #输出层权值,1-d
        self.input_cells        = [] #等于一个输入样本，1-d
        self.hidden_cells_i     = [] #隐含层输入,1-d
        self.output_cells_i     = [] #输出层输入,1-d
        self.hidden_cells_o     = [] #隐含层输出,1-d
        self.output_cells_o     = [] #输出层输出,1-d
        self.hidden_weights     = [] #根据习惯输出在前，是一个h*i的矩阵
        self.output_weights     = [] #根据习惯输出在前，是一个o*h的矩阵
        self.delta_o_w          = [] #梯度
        self.delta_h_w          = []
        self.delta_o_b          = [] #梯度
        self.delta_h_b          = []


    def setup(self, ni, nh, no):    #定义初始化函数
        self.input_n    = ni
        self.hidden_n   = nh
        self.output_n   = no
        #init cells
        self.input_cells      = np.zeros(self.input_n)
        self.hidden_cells_i   = np.zeros(self.hidden_n)
        self.output_cells_i   = np.zeros(self.output_n)
        self.hidden_cells_o   = np.zeros(self.hidden_n)
        self.output_cells_o   = np.zeros(self.output_n)
        #中间结果
        self.gj               = np.zeros(self.output_n)
        #init weights
        self.hidden_weights      = make_matrix(self.hidden_n, self.input_n,random.random())
        self.output_weights      = make_matrix(self.output_n, self.hidden_n,random.random())
        #init bias
#        np.random.seed(0)
        self.hidden_bias        = make_vector(self.hidden_n)
        self.output_bias        = make_vector(self.output_n)
        #init gradient
        self.delta_o_w          = make_matrix(self.output_n, self.hidden_n) #梯度
        self.delta_h_w          = make_matrix(self.hidden_n, self.input_n)
        self.delta_o_b          = [0.0] * self.output_n #梯度
        self.delta_h_b          = [0.0] * self.hidden_n
        #random activate
        for h in range(self.hidden_n):
            for i in range(self.input_n):
                self.hidden_weights[h][i] = rand(-0.2,0.2)
        for o in range(self.output_n):
            for h in range(self.hidden_n):
                self.output_weights[o][h] = rand(-0.2,0.2)
#        #init correction matrix
#        self.input_correction   = make_matrix(self.input_n, self.hidden_n)
#        self.output_correction  = make_matrix(self.hidden_n, self.output_n)

    #一次输入一个样本进行前馈传播，xi是一个1-d（维数等于特征数）
    def predict(self, xi):
        #开始前馈
        #activate input layer
        #输入层的神经元数等于xi的维数，输出就是xi
        for i in range(self.input_n):
            self.input_cells[i]     = xi[i]
        #activate hidden layer
        #隐含层的神经元数不确定是一个可调的超参数
        #每个神经元的输入等于输入层xi点乘权值
        #每个神经元的输出等于激活函数的输出，激活函数输入为神经元输入减去偏置
        for h in range(self.hidden_n):
            tatol = 0.0
            for i in range(self.input_n):
                tatol += self.input_cells[i] * self.hidden_weights[h][i]
            self.hidden_cells_i[h] = tatol
            self.hidden_cells_o[h] = sigmoid(tatol - self.hidden_bias[h])
        #activate output layer
        for o in range(self.output_n):
            tatol = 0.0
            for h in range(self.hidden_n):
                tatol += self.hidden_cells_o[h] * self.output_weights[o][h]
            self.output_cells_i[o] = tatol
            self.output_cells_o[o] = sigmoid(tatol - self.output_bias[o])
        return self.output_cells_o[:]
    #误差计算，可以选择单样本或批量样本误差计算
    #一个样本计算一次，每次输入xi对应的输出标签yi,是一个1-d,维数等于输出层神经元数
    def compute_error(self,yi):
        error = 0.0;
        for o in range(self.output_n):
            if (self.output_n == 1):
                error += np.power((yi - self.output_cells_o[o]),2)
            else:
                error += np.power((yi[o] - self.output_cells_o[o]),2)
        error = error * 0.5  #一次批量前馈产生的误差
        return error

    def gradient(self, learn, yi):
        #compute output gradient,先计算输出层梯度，权值和阈值
        #先计算权值，用误差对输出层权值求偏导
        for o in range(self.output_n):
            #计算误差对输出层神经元输入的偏导，该偏导需方便储供隐含层使用
            if (self.output_n == 1):
                delta2 = sigmoid_derivate(self.output_cells_o[o]) * (yi - self.output_cells_o[o])
            else:
                delta2 = sigmoid_derivate(self.output_cells_o[o]) * (yi[o] - self.output_cells_o[o])
            #保存中间结果
            self.gj[o] = delta2
            #更新偏置：一个样本，偏置更新一次
            #self.output_bias[o] += -1 * learn * delta2
            self.delta_o_b += -1 *learn * delta2
            for h in range(self.hidden_n):
            #计算输出层神经元输入对权值的偏导,即隐含层的输出
                delta1 = self.hidden_cells_o[h]
                #更新权值
                #self.output_weights[o][h] += learn * delta1 * delta2
                self.delta_o_w[o][h] += learn * delta1 * delta2
        #return self.output_weights[:],self.hidden_weights[:]
        #更新隐含层
        for h in range(self.hidden_n):
            #计算误差对隐含层神经元输入的偏导，
            totol = 0.0
            for o in range(self.output_n):
                totol +=  self.gj[o] * self.output_weights[o][h]
            delta2 = sigmoid_derivate(self.hidden_cells_o[o]) * totol
            #更新偏置：一个样本，偏置更新一次
            #self.hidden_bias[h] += -1 * learn * delta2
            self.delta_h_b[h] += -1 * learn * delta2
            for i in range(self.input_n):
                #计算隐含层神经元输入对权值的偏导,即输入层的输出
                delta1 = self.input_cells[i]
                #开始更新
                #self.hidden_weights[h][i] += learn * delta1 * delta2
                self.delta_h_w[h][i] += learn * delta1 * delta2

    #反向传播，更新权值，返回预测误差
    def back_propagate(self, patch_numbers):
        #compute output gradient,先计算输出层梯度，权值和阈值
        #先计算权值，用误差对输出层权值求偏导
        for o in range(self.output_n):
            self.output_bias[o] += self.delta_o_b[o] / (1.0 * patch_numbers)
            for h in range(self.hidden_n):
                self.output_weights[o][h] += self.delta_o_w[o][h] / (1.0 * patch_numbers)
        #更新隐含层
        for h in range(self.hidden_n):
            self.hidden_bias[h] += self.delta_h_b[h] / (1.0 * patch_numbers)
            for i in range(self.input_n):
                self.hidden_weights[h][i] += self.delta_h_w[h][i] / (1.0 * patch_numbers)
        #反馈一次清空批量调节的梯度值
        self.delta_o_w          = np.zeros((self.output_n, self.hidden_n)) #梯度
        self.delta_h_w          = np.zeros((self.hidden_n, self.input_n))
        self.delta_o_b          = np.zeros(self.output_n) #梯度
        self.delta_h_b          = np.zeros(self.hidden_n)

    def result(self, data_X, label_Y):
        number = len(label_Y)
        count = 0
        for i in range(number):
            sum = 0.0
            #y = self.output_cells_o
            y = self.predict(data_X[i])
            for j in range(self.output_n):
                if (self.output_n == 1):
                    sum +=  y[j] - label_Y[i]
                else:
                    sum +=  y[j] - label_Y[i][j]
            if (np.abs(sum) < 0.5):
                count += 1
        correct_rate = count * 1.0 / number
        return correct_rate

    def make_data(self):
        x, y = make_moons(600, noise=0.25)
        return x,y


#np.random.seed(5)

learn = 0.02
bpnet = BPNeuralNetwork()
data_X, data_Y = bpnet.make_data()
x_train = data_X[:500]
y_train = data_Y[:500]

x_pre = data_X[500:]
y_pre = data_Y[500:]

bpnet.setup(2,8,1)
#a = bpnet.hidden_weights
#b = bpnet.output_weights
#c = bpnet.hidden_bias
#d = bpnet.output_bias
length = len(data_Y)
print("没开始之前测试一下识别率 ：",bpnet.result(x_pre, y_pre))
for t in range(5000):
    for times in range(25):
        i = (times + 1) * 20
        j = times * 20
        patch_x = x_train[j:i]
        patch_y = y_train[j:i]
        patch_error = 0.0
        for index in range(20):
            bpnet.predict(patch_x[index])
            bpnet.gradient(learn,patch_y[index])
            patch_error += bpnet.compute_error(patch_y[index])
#        print(patch_error)
#        print(bpnet.hidden_weights)
#        print(bpnet.delta_h_w)
        bpnet.back_propagate(10)
#        print(bpnet.delta_h_w)

#        q = bpnet.delta_o_w
#        print(q)
    print("迭代",t,"次后的识别率 ：",bpnet.result(x_pre, y_pre))
print("最终的识别率识别率 ：",bpnet.result(x_pre, y_pre))
