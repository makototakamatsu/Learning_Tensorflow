
# coding: utf-8

# In[1]:

import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf


# In[2]:

#入力変数と出力変数のプレースホルダを生成
#値を格納する箱を作成
x = tf.placeholder(tf.float32,shape=(None,2),name="x")
y_ = tf.placeholder(tf.float32, shape=(None,1),name="y")
#以降は予測式(モデル)を記述
#モデルパラメータ
a = tf.Variable(tf.zeros((2,1)), name="a")
#モデル式
y = tf.matmul(x,a)


# In[4]:

#誤差関数と最適化手法を記述
#誤差関数loss
loss = tf.reduce_mean(tf.square(y_ - y))
#最適化手段を選ぶ(最急降下法)
train_step = tf.train.GradientDescentOptimizer(0.02).minimize(loss)


# In[6]:

#訓練データ作成
train_x = np.array([[1.,3.],[3.,1.],[5.,7.]])
train_y = np.array([190.,330.,660.]).reshape(3,1)
print("x=",train_x)
print ("y=",train_y)


# In[9]:

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

#最急降下法でパラメータ更新
for i in range(100):
    _, l, a_ = sess.run([train_step, loss, a], feed_dict={x: train_x, y_: train_y})
    if (i + 1) % 10 == 0:
        print("step=%3d, a1=%6.2f, a2=%6.2f, loss=%.2f" % (i + 1, a_[0], a_[1], l))
        
#学習結果を出力
est_a = sess.run(a,feed_dict={x: train_x,y_:train_y})
print("Estimated: a1=%6.2f, a2=%6.2f" %(est_a[0],est_a[1]))


# In[12]:

#予測
#新しいデータを用意
#りんご2個、みかん4個購入した例
new_x = np.array([2.,4.]).reshape(1,2)
#学習結果を使って予測実施
new_y = sess.run(y,feed_dict={x: new_x})
print(new_y)


# In[13]:

#セッション終了
sess.close()


# In[ ]:



