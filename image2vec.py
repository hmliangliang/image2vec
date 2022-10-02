#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2022/7/13 15:10
# @Author  : Liangliang
# @File    : test.py
# @Software: PyCharm

import numpy as np
import tensorflow as tf
from tensorflow import keras
import requests
import os
import time
import datetime
import s3fs
import math
import argparse
import pandas as pd
import base64

from multiprocessing.dummy import Pool
e = 0.0000001

result = 0

def multiprocessingWrite(file_number,data,output_path,count, ids):
    #print("开始写第{}个文件 {}".format(file_number,datetime.datetime.now()))
    n = len(data)  # 列表的长度
    #s3fs.S3FileSystem = S3FileSystemPatched
    #fs = s3fs.S3FileSystem()
    with open(os.path.join(output_path, 'pred_{}_{}_{}.csv'.format(count,ids, int(file_number))), mode="a") as resultfile:
        if n > 1:#说明此时的data是[[],[],...]的二级list形式
            for i in range(n):
                line = ",".join(map(str, data[i])) + "\n"
                resultfile.write(line)
        else:#说明此时的data是[x,x,...]的list形式
            line = ",".join(map(str, data)) + "\n"
            resultfile.write(line)
    print("第{}个大数据文件的第{}个分文件第{}个子文件已经写入完成,写入数据的行数{} {}".format(count,ids, file_number,n,datetime.datetime.now()))

class S3FileSystemPatched(s3fs.S3FileSystem):
    def __init__(self, *k, **kw):
        super(S3FileSystemPatched, self).__init__(*k,
                                                  key=os.environ['AWS_ACCESS_KEY_ID'],
                                                  secret=os.environ['AWS_SECRET_ACCESS_KEY'],
                                                  client_kwargs={'endpoint_url': 'http://' + os.environ['S3_ENDPOINT']},
                                                  **kw
                                                  )


class S3Filewrite:
    def __init__(self, args):
        super(S3Filewrite, self).__init__()
        self.output_path = args.data_output


def write(data, count, ids, args):
    #注意在此业务中data是一个二维list
    n_data = len(data) #数据的数量
    n = math.ceil(n_data/args.file_max_num) #列表的长度
    start = time.time()
    for i in range(0,n):
        multiprocessingWrite(i, data[i * args.file_max_num:min((i + 1) * args.file_max_num, n_data)],
                                 args.data_output, count,ids)
    cost = time.time() - start
    print("write is finish. write {} lines with {:.2f}s".format(n_data, cost))


class imageNet_teacher(keras.Model):
    def __init__(self, dim=64):
        super(imageNet_teacher, self).__init__()
        self.dim = dim
        self.bach_normalization1 = keras.layers.BatchNormalization()
        self.cov1 = keras.layers.Dense(self.dim, use_bias=True)
        self.cov2 = keras.layers.Dense(self.dim, use_bias=True)
        self.bach_normalization2 = keras.layers.BatchNormalization()
        # 定义预训练模型
        self.model = tf.keras.applications.densenet.DenseNet121(include_top=True, weights='imagenet', input_tensor=None,
                                                                input_shape=(224, 224, 3), pooling="avg", classes=1000)
        self.model = keras.models.Model(inputs=self.model.input, outputs=self.model.get_layer('avg_pool').output)
        self.model.trainable = False  # 冻结预训练模型参数

    def call(self, inputs, training=None, mask=None):
        inputs = self.model(inputs) #输出1024维的向量
        h = self.cov1(inputs)
        h = self.bach_normalization1(h, training=training)
        h = tf.nn.relu(h)
        h = self.cov2(h)
        h = self.bach_normalization2(h, training=training)
        h = tf.nn.relu(h)
        return h


class imageNet_student(keras.Model):
    def __init__(self, dim=64):
        super(imageNet_student, self).__init__()
        self.dim = dim
        self.bach_normalization1 = keras.layers.BatchNormalization()
        self.cov1 = keras.layers.Dense(self.dim, use_bias=True)
        #定义预训练模型
        self.model = tf.keras.applications.densenet.DenseNet121(include_top=True, weights='imagenet', input_tensor=None,
                                                       input_shape=(224, 224, 3), pooling="avg", classes=1000)
        self.model = keras.models.Model(inputs=self.model.input, outputs=self.model.get_layer('avg_pool').output)
        self.model.trainable = False #冻结预训练模型参数

    def call(self, inputs, training=None, mask=None):
        inputs = self.model(inputs) #输出1024维的向量
        h = self.cov1(inputs)
        h = self.bach_normalization1(h, training=training)
        h = tf.nn.relu(h)
        return h

def Loss(z_t, z_s, args):
    n = z_t.shape[0]
    loss_mse = tf.norm(z_t - z_s)
    #负采样过程
    z_neg = tf.gather(z_t,np.random.randint(0,n,n*args.neg_num),axis=0)
    z_t = tf.repeat(z_t, args.neg_num, axis=0)
    #融入正样本信息,以提高判别难度,防止产生塌陷解
    z_neg = (1 - args.alpha)*z_neg + args.alpha*z_t
    loss_neg = tf.norm(z_t-z_neg)
    loss = 1/n*loss_mse - args.beta*loss_neg/(args.neg_num*n)
    return loss

def get_image(url, num):
    # 将url地址从base64编码进行解码
    img_url = str(base64.b64decode(url), 'utf-8')
    #print("开始下载第{}个文件:{} {}".format(num, datetime.datetime.now(), img_url))
    # 下载数据
    r = requests.get(img_url, stream=True)
    # 返回状态码
    if r.status_code == 200:
        with open('tmp.jpg', 'wb') as f:
            f.write(r.content)  # 将内容写入图片
        img = tf.io.read_file('tmp.jpg')
        img = tf.io.decode_image(img, channels=3)
        if len(img.shape) == 4:  # 读取的图片为gif格式
            img = img[0]
        # https://blog.csdn.net/weixin_43239842/article/details/88585714
        # 数据增广采用剪切与颜色变换  simCLR算法证明了采用剪切与颜色变换对表征效果更好
        if len(img) < 3:
            print("异常图片的url=", img_url)
            img = tf.zeros((509, 484, 3))
        img_aug1 = tf.image.random_crop(img, [int(img.shape[0] * args.rate), int(img.shape[1] * args.rate), 3])
        img_aug1 = tf.image.random_hue(img_aug1, np.random.rand() / 2.2)
        img_aug2 = tf.image.random_crop(img, [int(img.shape[0] * args.rate), int(img.shape[1] * args.rate), 3])
        img_aug2 = tf.image.random_hue(img_aug2, np.random.rand() / 2.2)
        img_aug1 = tf.image.resize(img_aug1, (224, 224))
        img_aug2 = tf.image.resize(img_aug2, (224, 224))
        img_aug1 = tf.cast(img_aug1, tf.float32)
        img_aug2 = tf.cast(img_aug2, tf.float32)
        img_aug1 = tf.reshape(img_aug1, (1, 224, 224, 3))
        img_aug2 = tf.reshape(img_aug2, (1, 224, 224, 3))
        global batch_data
        global batch_data_agu
        batch_data[num] = img_aug1.numpy()
        batch_data_agu[num] = img_aug2.numpy()
    else:
        print("读取图像地址{}异常!".format(batch_files.iloc[num]))


if __name__ == "__main__":
    # 配置参数
    parser = argparse.ArgumentParser(description='算法的参数')
    parser.add_argument("--env", help="运行的环境(train or incremental_train)", type=str, default="incremental_train")
    parser.add_argument("--dim", help="数据的输出维数", type=int, default=128)
    parser.add_argument("--epoch", help="训练的epoch数目", type=int, default=150)
    parser.add_argument("--rate", help="图片剪切的比例", type=float, default=0.6)
    parser.add_argument("--tau", help="梯度动量更新的比例", type=float, default=0.9999)
    parser.add_argument("--alpha", help="负样本中正样本信息占比", type=float, default=0.15)
    parser.add_argument("--beta", help="负样本loss占比", type=float, default=0.5)
    parser.add_argument("--lr", help="学习率", type=float, default=0.00001)
    parser.add_argument("--neg_num", help="负采样的数目", type=int, default=4)
    parser.add_argument("--batch_size", help="batch_size的大小", type=int, default=100)
    parser.add_argument("--file_max_num", help="单个csv文件中写入数据的最大行数", type=int, default=800000)
    parser.add_argument("--data_input", help="输入数据的位置", type=str, default='')
    parser.add_argument("--data_output", help="数据的输出位置", type=str, default='')
    parser.add_argument("--tb_log_dir", help="日志位置", type=str, default='')
    parser.add_argument("--model_output", help="模型的输出位置", type=str,
                        default='s3://models/imageEmbeddingNet/')
    args = parser.parse_args()
    # 读取数据文件
    path = args.data_input.split(',')[0]
    s3fs.S3FileSystem = S3FileSystemPatched
    fs = s3fs.S3FileSystem()
    input_files = sorted([file for file in fs.ls(path) if file.find("part-") != -1])
    #读取预训练模型
    #model = tf.keras.applications.densenet.DenseNet121(include_top=True, weights='imagenet', input_tensor=None, input_shape=(224, 224, 3), pooling="avg", classes=1000)
    #model = keras.models.Model(inputs=model.input, outputs=model.get_layer('avg_pool').output)
    #model.trainable = False #冻结预训练模型参数

    #读取模型
    if args.env == "train":
        #第一次运行需要定义模型
        net_teacher = imageNet_teacher(args.dim) #teacher网络
        net_student = imageNet_student(args.dim) #student网络
    else:
        #装载已训练好的Teacher模型
        cmd = "s3cmd get -r  " + args.model_output + "net_teacher"
        os.system(cmd)
        net_teacher = keras.models.load_model("./net_teacher", custom_objects={'tf': tf}, compile=False)
        print("Teacher Model is loaded!")

        # 装载已训练好的Student模型
        cmd = "s3cmd get -r  " + args.model_output + "net_student"
        os.system(cmd)
        net_student = keras.models.load_model("./net_student", custom_objects={'tf': tf}, compile=False)
        print("Student Model is loaded!")
    before_net_teacher = net_teacher
    before_net_student = net_student
    before_loss = 2**31 - 1
    loss = 0
    # 定义优化器
    optimizer = keras.optimizers.Adam(learning_rate=args.lr)
    #读取图像数据
    for epoch in range(args.epoch):
        count = 0
        for file in input_files:
            count = count + 1
            print("当前正在处理第{}个文件,文件路径:{}...... {}".format(count, "s3://" + file, datetime.datetime.now()))
            data = pd.read_csv("s3://" + file, sep=',', header=None, usecols=[0, 1]).astype('str')  # 读取数据,第一列为id,第二列为url地址
            n = math.ceil(data.shape[0]/args.batch_size)
            for i in range(n):
                print("正在处理第{}个文件的第{}个子文件 {}".format(count, i+1, datetime.datetime.now()))
                if epoch == args.epoch - 1:  # 保存ID
                    ID = data.iloc[i * args.batch_size:min((i + 1) * args.batch_size, data.shape[0]), 0]  # 保存ID
                batch_files = data.iloc[i * args.batch_size:min((i + 1) * args.batch_size, data.shape[0]), 1]  # 训练图像的url地址
                batch_data = np.zeros((min((i+1)*args.batch_size, data.shape[0]) - i*args.batch_size, 224, 224, 3 ))
                batch_data_agu = np.zeros((min((i+1)*args.batch_size, data.shape[0]) - i*args.batch_size, 224, 224, 3))
                pool = Pool(processes=5)
                for num in range(batch_files.shape[0]):
                    pool.apply_async(func=get_image, args=(batch_files.iloc[num], num,))
                pool.close()
                pool.join()
                if i == 0:
                    print("数据处理完成! {}".format(datetime.datetime.now()))
                #把numpy数组转化成tensor类型
                batch_data = tf.convert_to_tensor(batch_data, dtype=tf.float32)
                batch_data_agu = tf.convert_to_tensor(batch_data_agu, dtype=tf.float32)
                batch_data = tf.keras.applications.densenet.preprocess_input(batch_data)
                batch_data_agu = tf.keras.applications.densenet.preprocess_input(batch_data_agu)
                #batch_data = model(batch_data)
                #batch_data_agu = model(batch_data_agu)
                with tf.GradientTape() as tape:
                    #开始对比学习与表征学习
                    z_t = net_teacher(batch_data, training = True)
                    z_t = tf.math.l2_normalize(z_t + e, axis=1)
                    z_s = net_student(batch_data_agu, training = True)
                    z_s = tf.math.l2_normalize(z_s + e, axis=1)
                    loss1 = Loss(z_t, z_s, args)
                    z_t = net_student(batch_data, training = True)
                    z_t = tf.math.l2_normalize(z_t + e, axis=1)
                    z_s = net_teacher(batch_data_agu, training = True)
                    z_s = tf.math.l2_normalize(z_s + e, axis=1)
                    loss2 = Loss(z_t, z_s, args)
                    loss = (loss1 + loss2)/2
                grads = tape.gradient(loss, net_teacher.trainable_variables)
                optimizer.apply_gradients(zip(grads, net_teacher.trainable_variables))
                #动量更新student网络参数 参考https://github.com/garder14/byol-tensorflow2/blob/main/pretraining.py
                net_student_weights = net_student.cov1.get_weights()
                for layer in range(len(net_student_weights)):
                    net_student_weights[layer] = args.tau*net_student_weights[layer] + (1 - args.tau)*net_teacher.cov1.get_weights()[layer]
                net_student.cov1.set_weights(net_student_weights)
                net_student_batch_weights = net_student.bach_normalization1.get_weights()
                for layer in range(len(net_student_batch_weights)):
                    net_student_batch_weights[layer] = args.tau*net_student_batch_weights[layer] + (1 - args.tau)*net_teacher.bach_normalization1.get_weights()[layer]
                net_student.bach_normalization1.set_weights(net_student_batch_weights)
                if epoch == args.epoch - 1:
                    #该输出embedding vector的时候到了
                    z_t = pd.concat([ID, pd.DataFrame(z_t.numpy()).astype("str")], axis=1).values.tolist()
                    write(z_t, count, i, args)
                print("epoch:{} 第{}个文件{}/{}子文件的loss:{} {}".format(epoch, count, i+1, n, loss, datetime.datetime.now()))
        if loss < before_loss:
            #保存结果
            before_net_teacher = net_teacher
            before_net_student = net_student
            before_loss = loss
        print("epoch:{} 当前的loss{} 当前最佳的loss:{} {}".format(epoch, loss, before_loss, datetime.datetime.now()))
    print("模型训练与embedding vectors均已完成!")
    # 保存teacher网络模型
    before_net_teacher.save("./net_teacher", save_format="tf")
    print("teacher_net已保存!")
    cmd = "s3cmd put -r ./net_teacher " + args.model_output
    os.system(cmd)
    # 保存student网络模型
    before_net_student.save("./net_student", save_format="tf")
    print("net_student已保存!")
    cmd = "s3cmd put -r ./net_student " + args.model_output
    os.system(cmd)
    print("网络模型上传完成! {}".format(datetime.datetime.now()))