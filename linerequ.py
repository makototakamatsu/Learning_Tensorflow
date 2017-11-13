
# coding: utf-8

# In[13]:


import tensorflow as tf
import numpy as np

## 学習データ作成（x座標をランダムに作成(0.0〜1.0を100個)
x_data = np.random.rand(100).astype(np.float32)
# y座標を生成 (y = 0.5x + 10)
y_data = 0.5 * x_data + 10

## モデルを作成（y_data = W * x_data + b となる W と b の適正値を見つけます。
W = tf.Variable(tf.zeros([1]))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

## 損失関数を作成（最小二乗法を使用
loss = tf.reduce_mean(tf.square(y - y_data))

## 最適化アルゴリズムを指定（勾配降下法で損失関数を最小化
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

## パラメータを初期化（Variableを使用する場合必要らしい
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

## 学習する
for step in range(200):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))

