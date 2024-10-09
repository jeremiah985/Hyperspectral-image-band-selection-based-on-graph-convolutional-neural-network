
"""
@author: danfeng
"""
#import library
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.io as scio 
import scipy.io as sio
from tf_utils import random_mini_batches, convert_to_one_hot
from tensorflow.python.framework import ops

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def create_placeholders(n_x, n_y):
#占位符，为后续模型计算准备接口
    isTraining = tf.placeholder_with_default(True, shape=())
    x_in = tf.placeholder(tf.float32,  [None, n_x], name = "x_in")
    y_in = tf.placeholder(tf.float32, [None, n_y], name = "y_in")
    mask_train = tf.placeholder(tf.float32, name = "mask_train")
    mask_test = tf.placeholder(tf.float32, name = "mask_test")
    lap = tf.placeholder(tf.float32, [None, None], name = "lap")
    
    return x_in, y_in, lap, mask_train, mask_test, isTraining

def initialize_parameters():
   #初始化网络参数，
    tf.set_random_seed(1)
    #随机种子，确保每次实验的一致性
    #第一层，200个输入，128个输出，大小为【200，128】                           #这里是初始化权重的策略
    x_w1 = tf.get_variable("x_w1", [200,128], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    #偏置初始化为0
    x_b1 = tf.get_variable("x_b1", [128], initializer = tf.zeros_initializer())
    #第二个权重矩阵输入128，输出16
    
    x_w2 = tf.get_variable("x_w2", [128,16], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    #偏置初始化为0
    x_b2 = tf.get_variable("x_b2", [16], initializer = tf.zeros_initializer())
    
    #存储了所有的参数
    parameters = {"x_w1": x_w1,
                  "x_b1": x_b1,
                  "x_w2": x_w2,
                  "x_b2": x_b2}
                  
    return parameters

#图卷积层，接受 输入特征矩阵，拉普拉斯矩阵，权重矩阵
#              输出的是 经过图卷积处理后的特征矩阵，每个节点不仅包含了自身的信息，还包含了邻居节点的信息
def GCN_layer(x_in, L_, weights):

    x_mid = tf.matmul(x_in, weights)
    x_out = tf.matmul(L_, x_mid)
    
    return x_out


# 接受的是输入特征，参数，拉普拉斯矩阵，训练状态和动量参数（用于批量归一化）
#        输出的是输出特征和L2正则化损失

def mynetwork(x, parameters, Lap, isTraining, momentums = 0.9):

    with tf.name_scope("x_layer_1"):
        #归一化，标准化着每个特征维度，momentums参数控制着历史均值的影响
         x_z1_bn = tf.layers.batch_normalization(x, momentum = momentums, training = isTraining)  
         #图卷积，加上偏置        
         x_z1 = GCN_layer(x_z1_bn, Lap, parameters['x_w1']) + parameters['x_b1']
         #再次归一化
         x_z1_bn = tf.layers.batch_normalization(x_z1, momentum = momentums, training = isTraining)  
         #激活函数，进行下一次映射 
         x_a1 = tf.nn.relu(x_z1_bn)     

    with tf.name_scope("x_layer_2"):
         #同上：
         x_z2_bn = tf.layers.batch_normalization(x_a1, momentum = momentums, training = isTraining)  
         #得到第二层的输出特征          
         x_z2 = GCN_layer(x_z2_bn, Lap, parameters['x_w2']) + parameters['x_b2']         
    #l2损失函数是两个权重矩阵的最小值，
    #为什么要学习最小的呢？怕过拟合么？
    l2_loss =  tf.nn.l2_loss(parameters['x_w1']) + tf.nn.l2_loss(parameters['x_w2'])
                
    return x_z2, l2_loss


#                       预测输出，实际标签，L2正则化损失，掩码，正则化强度系数，学习率，跟踪全局训练步骤的变量
def mynetwork_optimaization(y_est,  y_re, l2_loss,      mask, reg, learning_rate,  global_step):
    
    with tf.name_scope("cost"):
        #交叉熵损失
         cost = (tf.nn.softmax_cross_entropy_with_logits(logits = y_est, labels = y_re)) +  reg * l2_loss
         #使损失只关注有效样本
         mask = tf.cast(mask, dtype = tf.float32)
         mask /= tf.reduce_mean(mask)
         cost *= mask
         cost = tf.reduce_mean(cost) +  reg * l2_loss
         #更新操作
    with tf.name_scope("optimization"):
         update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
         optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost,  global_step=global_step)
         optimizer = tf.group([optimizer, update_ops])
         
    return cost, optimizer

#考虑掩码的情况下保持计算的准确率
def masked_accuracy(preds, labels, mask):

      correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
      accuracy = tf.cast(correct_prediction, "float")
      mask = tf.cast(mask, dtype = tf.float32)
      mask /= tf.reduce_mean(mask)
      accuracy *= mask
      
      return tf.reduce_mean(accuracy)

#训练数据的特征，标签，拉普拉斯矩阵，训练掩码，测试掩码，学习率，正则化参数，训练轮数，是否打印过程
def train_mynetwork(x_all, y_all, L_all, mask_in, mask_out, learning_rate = 0.001, /
                    beta_reg = 0.001, num_epochs = 200, print_cost = True):
    #重置图
    ops.reset_default_graph()  
    #获取输入数据的形状   
    # m样本数量，n_x特征数量 n_y标签数量                                                    
    (m, n_x) = x_all.shape
    (m, n_y) = y_all.shape
    
    #初始化数据存储列表
    costs = []                                        
    costs_dev = []
    train_acc = []
    val_acc = []
    #创建占位符
    x_in, y_in, lap, mask_train, mask_test, isTraining = create_placeholders(n_x, n_y) 
    #初始化参数
    parameters = initialize_parameters()
    #构建网络
    with tf.name_scope("network"):
         x_out, l2_loss = mynetwork(x_in, parameters, lap, isTraining)

    global_step = tf.Variable(0, trainable=False)
    
    with tf.name_scope("optimization"):
         cost, optimizer = mynetwork_optimaization(x_out, y_in, l2_loss, mask_train, beta_reg, learning_rate, global_step)

    with tf.name_scope("metrics"):
         accuracy_train = masked_accuracy(x_out, y_in, mask_train)
         accuracy_test= masked_accuracy(x_out, y_in, mask_test)
         
    init = tf.global_variables_initializer()
    #创建会话并训练
    with tf.Session() as sess:
        
        sess.run(init)
        # Do the training loop
        for epoch in range(num_epochs + 1):

            _, epoch_cost, epoch_acc = sess.run([optimizer, cost, accuracy_train], feed_dict={x_in: x_all, y_in: y_all, lap: L_all, mask_train: mask_in, mask_test: mask_out, isTraining: True})
            
            if print_cost == True and epoch % 50 == 0:
                features, overall_cost_dev, overall_acc_dev = sess.run([x_out, cost, accuracy_test], feed_dict={x_in: x_all, y_in: y_all, lap: L_all, mask_train: mask_in, mask_test: mask_out, isTraining: False})
                print ("epoch %i: Train_loss: %f, Val_loss: %f, Train_acc: %f, Val_acc: %f" % (epoch, epoch_cost, overall_cost_dev, epoch_acc, overall_acc_dev))
            
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                train_acc.append(epoch_acc)
                costs_dev.append(overall_cost_dev)
                val_acc.append(overall_acc_dev)
      
        # plot the cost         
        plt.plot(np.squeeze(costs))
        plt.plot(np.squeeze(costs_dev))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        # plot the accuracy 
        plt.plot(np.squeeze(train_acc))
        plt.plot(np.squeeze(val_acc))
        plt.ylabel('accuracy')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
    
        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")
    
        return parameters , val_acc, features


ALL_X = scio.loadmat('HSI_GCN/ALL_X.mat')
ALL_Y = scio.loadmat('HSI_GCN/ALL_Y.mat')
ALL_L = scio.loadmat('HSI_GCN/ALL_L.mat')

ALL_L = ALL_L['ALL_L']
ALL_X = ALL_X['ALL_X']
ALL_Y = ALL_Y['ALL_Y']

GCN_mask_TR = sample_mask(np.arange(0,695), ALL_Y.shape[0])
GCN_mask_TE = sample_mask(np.arange(696,10366), ALL_Y.shape[0])

ALL_Y = convert_to_one_hot(ALL_Y - 1, 16)
ALL_Y = ALL_Y.T


parameters, val_acc, features = train_mynetwork(ALL_X, ALL_Y, ALL_L.todense(), GCN_mask_TR, GCN_mask_TE)
sio.savemat('features.mat', {'features': features})