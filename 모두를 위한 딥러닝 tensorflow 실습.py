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


