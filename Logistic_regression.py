
# coding: utf-8

# In[9]:

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# In[10]:

#予測式(モデル)を記述
#入力変数と出力変数のプレースホルダを生成
x = tf.placeholder(tf.float32, shape=(None,2),name="x")
y_ = tf.placeholder(tf.float32,shape=(None,1),name="y")
#モデルパラメータ
a=tf.Variable(-10*tf.ones((2,1)),name="a")
b=tf.Variable(200.,name="b")
#モデル式
u=tf.matmul(x,a)+b
y=tf.sigmoid(u)


# In[11]:

#誤差関数と最適化手法を記述
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=u,labels= y_))
#最適化手段(最急降下法)
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)


# In[14]:

sess=tf.Session()
init=tf.global_variables_initializer()
sess.run(init)

#最急降下法でパラメータ更新
for i in range(100):
    _, l, a_, b_ = sess.run([train_step, loss, a, b], feed_dict={x:  train_x, y_:  train_y})
    if (i+1)%10 ==0:
        print("step=%3d, a1=%6.2f, a2=%6.2f, loss=%.2f" %(i+1,a_[0],a_[1],b_,1))

#学習結果出力
est_a,est_b = sess.run([a,b], feed_dict={x: train_x, y_: train_y})
print("Estimated:a1=%6.2f, a2=%6.2f, b=%6.2f"%(est_a[0],est_a[1],est_b))


# In[ ]:




# In[ ]:



