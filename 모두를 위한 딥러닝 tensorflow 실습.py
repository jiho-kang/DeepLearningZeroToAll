# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # 모두를 위한 딥러닝 tensorflow

# +
##Hello Tensorflow
# -

import tensorflow as tf
hello = tf.constant("Hello,Tensorflow!")
sess = tf.Session()
print(sess.run(hello))

# # 01) Computational Graph

# (1) Build graph(tensors) using TensorFlow operations

node1 = tf.constant(3.0, tf.float32) #넣을내용,데이터형식
node2 = tf.constant(4.0) #also tf.float32 implicitly
node3 = tf.add(node1,node2) #node3 = node1 + node2도 가능

print("node1:", node2, "node2:", node2)
print("node3:", node3)

# (2) feed data and run graph (operation)
#     sess.run(op)

# (3) update variables in the graph
#     (and return values)

sess = tf.Session()
print("sess.run([node1,node2]): ",sess.run([node1,node2]))
print("sess.run(node3): ",sess.run(node3))

# ### Placeholder (like input)

# placeholder로 미지수로 남기고, feed_dict로 넣어줌. n개 가능

# +
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a+b #tf.add(a,b)도 가능

print(sess.run(adder_node, feed_dict={a:3,b:4.5}))
print(sess.run(adder_node, feed_dict={a:[1,3], b:[2,4]}))
# -

# ### Tensor Ranks, Shpaes and Types

# Rank / Mateh entity / Python example

#     0 / Scalar(magnitude only) / s = 483
#     1 / Vector(magnitude and directioni) / v = [1.2, 3.3, 4.3]
#     2 / Matrix(table of numbers) / m = [[1,2], [5,6], [7,3]]
#     3 / 3-Tensor(cube of numbers) t = [[[2],[3],[4]],[[5],[6],[7]],[[8],[9],[10]]]
#     n / n-Tensor

# # 02) Linear regression 구현

# ### (1) Build graph using TF operations
# H(x) = Wx+b

# +
x = [1,2,3]
y = [1,2,3]

#tt.Variable = TF가 학습하는 과정에서 변경시키는 v이다.
w = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
#our hypothesis Wx+b
hypothesis = w*x + b
# -

# cost(W,b) = 1/m (시그마 i=1~m : (H(xi)-yi)^2)

#tf.reduce_mean: 평균 값을 구해줌
#tf.square: 제곱
cost = tf.reduce_mean(tf.square(hypothesis - y))

# GradientDescent (경사하강법)로 cost 최소값 구하기

optimizer =  tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# ### (2),(3) Run/Update graph and get results

# +
sess =tf.Session()
#tf.variable을 사용할 경우 run전에 tf.global_variables_initializer()를 사용해야함
sess.run(tf.global_variables_initializer())

for step in range(2001):
    sess.run(train)
    if step % 100 == 0:
        print(step, sess.run(cost), sess.run(w), sess.run(b))
# -

# ### Full code with placeholders

# +
import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

w = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
x = tf.placeholder(tf.float32, shape = [None])
y = tf.placeholder(tf.float32, shape = [None])

hy = w*x+b
cost = tf.reduce_mean(tf.square(hy-y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, w_val, b_val, _ = sess.run([cost, w, b, train], feed_dict={x:[1,2,3,4], y:[2.1,3.1,4.1,5.1]})
    if step %200==0:
        print(step,cost_val,w_val,b_val)
        
print(sess.run(hy, feed_dict={x: [5]}))
print(sess.run(hy, feed_dict={x: [2.5]}))
# -

# # 03) Linear Regression의 cost 최소화 구현

# ### cost함수 그리기

# +
import tensorflow as tf
import matplotlib.pyplot as plt

x=[1,2,3]
y=[1,2,3]
w=tf.placeholder(tf.float32)
hypothesis = w*x

cost = tf.reduce_mean(tf.square(hypothesis-y))
sess=tf.Session()
sess.run(tf.global_variables_initializer())

w_val=[]
cost_val=[]
for i in range(-30,50):
    feed_w = i*0.1
    curr_cost, curr_w = sess.run([cost,w], feed_dict = {w: feed_w})
    w_val.append(curr_w)
    cost_val.append(curr_cost)
    
#show the cost function
plt.plot(w_val, cost_val)
plt.show()
# -

# ### cost함수 미분하기
#     위의 cost function에서 gradient descent를 활용하여 미분하기
#     cost(w)를 미분했을 때 양/음 값이 나오면 w값을 -/+조절하는 함수 만들기

#Minimize: Gradient Descent using derivative(미분):
#w -= Learning_rate * derivative
learning_rate = 0.1
gradient = tf.reduce_mean((w*x-y)*x)
descent = w - learning_rate * gradient
update = w.assign(descent)

# ## full code

# +
import tensorflow as tf
x_data=[1,2,3]
y_data=[1,2,3]

w=tf.Variable(tf.random_normal([1]), name = 'weight')
x=tf.placeholder(tf.float32, shape=[None])
y=tf.placeholder(tf.float32, shape=[None])
hy=w*x
cost = tf.reduce_mean(tf.square(hy-y))

#이 부분이
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
#train = optimizer.minimize(cost)
#로 대체할 수 있는 것
learning_rate = 0.1
gradient = tf.reduce_mean((w*x-y)*x)
descent = w - learning_rate * gradient
update = w.assign(descent)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(21):
    sess.run(update, feed_dict={x:x_data,y:y_data})
    print(step, sess.run(cost,feed_dict={x:x_data,y:y_data}), sess.run(w))
# -

# # 04-1) multi-variable linear regression 구현

# ### Matrix (X)

# +
import tensorflow as tf

#여기부터
x1_data = [73,93, 89, 96, 73]
x2_data = [80, 88, 91, 98, 66]
x3_data = [75, 93, 90, 100, 70]
y_data = [152, 185, 180, 196, 142]

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
#여기까지 매트리스로 구현 가능

w1 = tf.Variable(tf.random_normal([1]), name='weight1')
w2 = tf.Variable(tf.random_normal([1]), name='weight2')
w3 = tf.Variable(tf.random_normal([1]), name='weight3')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = x1*w1 + x2*w2 + x3*w3 +b

cost = tf.reduce_mean(tf.square(hypothesis-y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5)
train = optimizer.minimize(cost)

sess=tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                        feed_dict={x1: x1_data, x2:x2_data, x3: x3_data, y: y_data})
    if step % 1000 == 0:
        print(step, "cost:", cost_val, "\nprediction:\n", hy_val, "\n")
# -

# ### Matrix (0)

# +
import tensorflow as tf

x_data = [[73,80,75],[93,88,93],[89,91,90],[96,98,100],[73,66,70]]
y_data = [[152], [185], [180], [196], [142]]

x = tf.placeholder(tf.float32, shape=[None,3])
y = tf.placeholder(tf.float32, shape=[None,1])

#w는 normal([들어오는 값,나가는 값])
#b는 nomal([나가는 값])
w = tf.Variable(tf.random_normal([3,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(x,w)+b

cost = tf.reduce_mean(tf.square(hypothesis-y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5)
train = optimizer.minimize(cost)

sess=tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                        feed_dict={x: x_data, y: y_data})
    if step % 1000 == 0:
        print(step, "cost:", cost_val, "\nprediction:\n", hy_val, "\n")
# -

# # 04-2) 파일에서 데이터 읽어오기

# +
import tensorflow as tf
import numpy as np

xy = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# Make sure the shape and data are OK
#print(x_data.shape, "\n", x_data, len(x_data))
#print(y_data.shape, "\n", y_data, len(y_data))

# +
import tensorflow as tf
import numpy as np

xy = np.loadtxt('data-01-test-score.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

x = tf.placeholder(tf.float32, shape=[None,3])
y = tf.placeholder(tf.float32, shape=[None,1])

#w는 normal([들어오는 값,나가는 값])
#b는 nomal([나가는 값])
w = tf.Variable(tf.random_normal([3,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(x,w)+b

cost = tf.reduce_mean(tf.square(hypothesis-y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5)
train = optimizer.minimize(cost)

sess=tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                        feed_dict={x: x_data, y: y_data})
    #if step % 1000 == 0:
    #    print(step, "cost:", cost_val, "\nprediction:\n", hy_val, "\n")

print("Your Score will be", sess.run(hypothesis, feed_dict={x: [[100,70,101]]}))
print("Your Score will be", sess.run(hypothesis, feed_dict={x: [[60,70,110],[90,100,80]]}))

# -

# ### Queue Runners 이용 - 이해 부족

#     파일의 양, 크기가 너무 클 경우 용량이 부족해진다.
#     tensorflow의 queue runner를 이용하여 해결

# +
import tensorflow as tf
filename_queue = tf.train.string_input_producer(
    ['data-01-test-scroe.csv'], shuffle=False, name='filename_queue')
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

record_defaults = [[0.],[0.],[0.],[0.]]
xy = tf.decode_csv(value, record_defaults=record_defaults)

train_x_batch, train_y_batch = \
    tf.train.batch([xy[0:-1], xy[-1]], batch_size=10)

x = tf.placeholder(tf.float32, shape=[None,3])
y = tf.placeholder(tf.float32, shape=[None,1])

#w는 normal([들어오는 값,나가는 값])
#b는 nomal([나가는 값])
w = tf.Variable(tf.random_normal([3,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(x,w)+b

cost = tf.reduce_mean(tf.square(hypothesis-y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-5)
train = optimizer.minimize(cost)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for step in range(2001):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                        feed_dict={x: x_data, y: y_data})
    if step % 1000 == 0:
        print(step, "cost:", cost_val, "\nprediction:\n", hy_val, "\n")
        
coord.request_stop()
coord.join(threads)
# -

# # 05) Logistic Classification 구현

# +
import tensorflow as tf

x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]
y_data = [[0],[0],[0],[1],[1],[1]]

x = tf.placeholder(tf.float32, shape=[None,2])
y = tf.placeholder(tf.float32, shape=[None,1])

#w는 normal([들어오는 값,나가는 값])
#b는 nomal([나가는 값])
w = tf.Variable(tf.random_normal([2,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(x,w)+b)

cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

#if문 같은 것.
# true if hypothesis>0.5 and return(TorF) in float. T=1, F=0
predicted = tf.cast(hypothesis>0.5, dtype = tf.float32)

#predict값과 y의 값이 같으면 true, false를 float으로 반환하여 평균을 냄
#값이 높을수록 정확도가 높은 것
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,y),dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={x: x_data, y:y_data})
        if step % 1000 == 0:
            print(step,cost_val)
            
    h,c,a = sess.run([hypothesis,predicted,accuracy], feed_dict={x: x_data, y:y_data})
    print("\nHypothesis:",h,"\ncorrect(y):",c,"\nAccuracy:",a)
# -

# ### Classifying diabetes (실제 데이터로 당뇨병 예측하기)

# +
import tensorflow as tf
import numpy as np

xy = np.loadtxt('data-03-diabetes.csv', delimiter = ',', dtype=np.float32)
x_data=xy[:,0:-1]#여기 한번 수정해보기
y_data=xy[:,[-1]]

x = tf.placeholder(tf.float32, shape=[None,8])
y = tf.placeholder(tf.float32, shape=[None,1])

#w는 normal([들어오는 값,나가는 값])
#b는 nomal([나가는 값])
w = tf.Variable(tf.random_normal([8,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(x,w)+b)

cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

#if문 같은 것.
# true if hypothesis>0.5 and return(TorF) in float. T=1, F=0
predicted = tf.cast(hypothesis>0.5, dtype = tf.float32)

#predict값과 y의 값이 같으면 true, false를 float으로 반환하여 평균을 냄
#값이 높을수록 정확도가 높은 것
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,y),dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={x: x_data, y:y_data})
        if step % 1000 == 0:
            print(step,cost_val)
            
    h,c,a = sess.run([hypothesis,predicted,accuracy], feed_dict={x: x_data, y:y_data})
    print("\nHypothesis:",h,"\ncorrect(y):",c,"\nAccuracy:",a)
# -
# # 06-1) Softmax Classification 구현


# ### 배운 이론을 식으로 정리

# +
#먼저 XW=Y를 만들어주기 위해
Z = tf.matmul(x,w)+b

#hypothesis에 기존에는 sigmoid함수를 사용했다면
hypothesis = tf.nn.softmax(Z)

#cost function으로 나타내기
cost= tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis),axis=1))

#cost function 최소화하기
optimizer = tf.train.GradienDescentOptimizer(learning_rate=0.1).minimize(cost)

#arg_max
#훈련시킨 다음, 데이터를 넣어서 hypothesis로 나온 값을 one-hot codind된 수로 출력하기
a = sess.run(hypothesis, feed_dict={x:[넣을 데이터 값]})
print(a, sess.run(tf.arg_max(a,1)))
# -

# ### 실제 구현

# +
import tensorflow as tf
import numpy as np

x_data = [[1,2,1,1,],[2,1,3,2,],[3,1,3,4,],[4,1,5,5,],[1,7,5,5],[1,2,5,6],[1,6,6,6,],[1,7,7,7,]]
#one-hot encoding
y_data = [[0,0,1],[0,0,1],[0,0,1],[0,1,0],[0,1,0],[0,1,0],[1,0,0],[1,0,0]]

x=tf.placeholder("float",[None, 4])
y=tf.placeholder("float",[None, 3])
nb_classes = 3

#w는 normal([들어오는 값,나가는 값])
#b는 nomal([나가는 값])
w = tf.Variable(tf.random_normal([4, nb_classes]), name = 'weight')
b = tf.Variable(tf.random_normal([nb_classes]), name = 'bias')

hypothesis = tf.nn.softmax(tf.matmul(x,w)+b)

#cross entropy cost/loss
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2000):
        sess.run(optimizer, feed_dict={x:x_data, y:y_data})
    
    a = sess.run(hypothesis, feed_dict={x: [[1,11,7,9],[1,3,4,3],[1,1,0,1]]})
    print(a, sess.run(tf.math.argmax(a,1)))

# -

# # 06-2) Fancy한 Softmax Classifier

# cross_entropy, one_hot, reshape 함수를 사용

# ### tf.nn.softmax_cross_entropy_with_logits

# +
logits_x = tf.matual(x,w)+b
hypothesis = tf.nn.softmax(logits_x)

#1 기존
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hpyothesis), axis=1))

#2 함수 사용
cost_i = tf.nn.softmax_cross_entropy_with_logits(logits = logits_x,
                                                labels = y)
cost = tf.reduce_mean(cost_i)
# -

# ### tf.one_hot() & tf.reshape()
# one_hot으로 인해 실제 데이터에 한 차원이 더 생김. reshape으로 되돌려줌

nb_classes = 7 #y값의 범위. 해당예시) 0~6
y = tf.placholder(tf.int32,[None,1]) #shape=(?,1)
y_one_hot = tf.one_hot(y,nb_classes) #shape=(?,1,7)
y_one_hot = tf.reshape(Y_one_hot, [-1,nb_classes]) #shape=(?,7)

# ### 실제 데이터로 classification
# TF버전이 바뀌면서 돌아가지 않음
#
# TF2 버전 코드
#
# https://github.com/hunkim/DeepLearningZeroToAll/tree/master/tf2

# +
import tensorflow as tf
import numpy as np

#predicting animal type based on various features
xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:,:-1]
y_data = xy[:, [-1]]

nb_classes = 7

x = tf.placeholder(tf.float32, [None,16])
y = tf.placeholder(tf.int32, [None,1])

y_one_hot = tf.one_hot(y,nb_classes)
y_one_hot = tf.reshape(y_one_hot, [-1, nb_classes])

w = tf.Variable(tf.random_normal([16,nb_classes]), name = 'weight')
b = tf.Variable(tf.random_normal([nb_classes]), name = 'weight')

logits = tf.matmul(x,w)+b
hypothesis = tf.nn.softmax(logits)

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y_one_hot)
cost = tf.reduce_mean(cost_i)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

prediction = tf.argmax(hypothesis,1)
answer = tf.equal(prediction, tf.argmax(y_one_hot,1))
accuracy = tf.reduce_mean(tf.cast(answer,prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        sess.run(optimizer, feed_dict={x:x_data, y:y_data})
        if step %100 == 0:
            lass, acc = sess.run([cost, accuracy], feed_dict={x:x_data, y:y_data})
            print("step:{:5}\tLoss: {:.3f}\t.Acc: {:.2%}".format(step,loss,acc))
    
    predict = sess.run(prediction, feed_data={x:x_data})
    
    for p,y in zip(predict,y_data.flatten()):
        print("[{}] Prediction: {} Real Y: {}".format(p==int(y),p,int(y)))
# -

# # 07-1) training/test dataset, learning rate, normalization

# ### Min_max normalization

# +
import tensorflow as tf
import numpy as np


def min_max_scaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)


xy = np.array(
    [
        [828.659973, 833.450012, 908100, 828.349976, 831.659973],
        [823.02002, 828.070007, 1828100, 821.655029, 828.070007],
        [819.929993, 824.400024, 1438100, 818.97998, 824.159973],
        [816, 820.958984, 1008100, 815.48999, 819.23999],
        [819.359985, 823, 1188100, 818.469971, 818.97998],
        [819, 823, 1198100, 816, 820.450012],
        [811.700012, 815.25, 1098100, 809.780029, 813.669983],
        [809.51001, 816.659973, 1398100, 804.539978, 809.559998],
    ]
)

# very important. It does not work without it.
xy = min_max_scaler(xy)
print(xy)

'''
[[0.99999999 0.99999999 0.         1.         1.        ]
 [0.70548491 0.70439552 1.         0.71881782 0.83755791]
 [0.54412549 0.50274824 0.57608696 0.606468   0.6606331 ]
 [0.33890353 0.31368023 0.10869565 0.45989134 0.43800918]
 [0.51436    0.42582389 0.30434783 0.58504805 0.42624401]
 [0.49556179 0.42582389 0.31521739 0.48131134 0.49276137]
 [0.11436064 0.         0.20652174 0.22007776 0.18597238]
 [0.         0.07747099 0.5326087  0.         0.        ]]
'''

x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.Dense(units=1, input_dim=4))
tf.model.add(tf.keras.layers.Activation('linear'))
tf.model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(lr=1e-5))
tf.model.summary()

history = tf.model.fit(x_data, y_data, epochs=1000)

predictions = tf.model.predict(x_data)
score = tf.model.evaluate(x_data, y_data)

print('Prediction: \n', predictions)
print('Cost: ', score)
# -

# ### Learning rate and Evaluation

# +
# Lab 7 Learning rate and Evaluation
import tensorflow as tf

x_data = [[1, 2, 1],
          [1, 3, 2],
          [1, 3, 4],
          [1, 5, 5],
          [1, 7, 5],
          [1, 2, 5],
          [1, 6, 6],
          [1, 7, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

# Evaluation our model using this test dataset
x_test = [[2, 1, 1],
          [3, 1, 2],
          [3, 3, 4]]
y_test = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1]]

# try different learning_rate
# learning_rate = 65535  # ? it works too hahaha
learning_rate = 0.1
# learning_rate = 1e-10  # small learning rate won't work either

tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.Dense(units=3, input_dim=3, activation='softmax'))
tf.model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(lr=learning_rate), metrics=['accuracy'])

tf.model.fit(x_data, y_data, epochs=1000)

# predict
print("Prediction: ", tf.model.predict_classes(x_test))

# Calculate the accuracy
print("Accuracy: ", tf.model.evaluate(x_test, y_test)[1])
