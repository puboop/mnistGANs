import tensorflow as tf
from tensorflow import keras
from visual import save_gan, cvt_gif
from utils import set_soft_gpu, binary_accuracy, save_weights
import numpy as np
import time

"""
基于gan网络的二次函数绘制
"""


class FunctionType:
    """声明各种类型的真是数据函数
    可绘制各种各样的函数图像
    """

    def get(self, func_name: str):
        try:
            return getattr(self, func_name)
        except Exception:
            raise ValueError(f"{self}对象中未找到{func_name}函数！！！")

    def square(self, data_dim, batch_size):
        """
        一元二次方程公式：y = a*x^2 + b
        """
        for i in range(300):
            # np.newaxis 在一维的数组中增加一个维度
            a = np.random.uniform(1, 2, size=batch_size)[:, np.newaxis]
            # repeat变换维度，将一维的数据转换为对应的维度，会将原有的一维重复至相应的维度
            # -4, 4 为x的输入范围
            base = np.linspace(-4, 4, data_dim)[np.newaxis, :].repeat(batch_size, axis=0)
            yield a * np.power(base, 2) + (a - 1)

    def sin(self, data_dim, batch_size):
        """
        获取真实数据
        正弦函数公式：sin(x)
        """
        for i in range(300):
            x = np.linspace(-4, 4, data_dim)[np.newaxis, :].repeat(batch_size, axis=0)
            yield np.sin(x)


class GAN(keras.Model):
    def __init__(self, latent_dim, data_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.data_dim = data_dim

        self.g = self._get_generator()
        self.d = self._get_discriminator()
        # beta_1(一阶动量参数) 动量估计的指数衰减率 可以帮助模型更好的收敛
        self.opt = keras.optimizers.Adam(0.0002, beta_1=0.5)
        # 当使用 from_logits=True 时，表示传入的预测值 y_pred 是 logits，而不是概率。logits 是指在经过 sigmoid 激活函数之前的原始预测值
        # 这时，BinaryCrossentropy 会先对 logits 应用 sigmoid 函数，将其转换为概率，然后再计算交叉熵损失。
        self.loss_func = keras.losses.BinaryCrossentropy(from_logits=True)

    def call(self, n, training=None, mask=None):
        """
        normal 正态分布的序列
        tf.random.normal((n, self.latent_dim))表示着随机输入的x，根据随机输入的x，得到一个y值
        n 为输入n维度数量
        """
        return self.g.call(tf.random.normal((n, self.latent_dim)), training=training)

    def _get_generator(self):
        model = keras.Sequential([
            keras.Input([None, self.latent_dim]),
            # 全链接层 输出神经元为32个
            keras.layers.Dense(32, activation=keras.activations.relu),
            # 全链接层
            keras.layers.Dense(self.data_dim),
        ], name="generator")
        model.summary()  # 允许打印模型的概况
        return model

    def _get_discriminator(self):
        model = keras.Sequential([
            keras.Input([None, self.data_dim]),
            keras.layers.Dense(32, activation=keras.activations.relu),
            # 全链接层神经元个数为一个
            keras.layers.Dense(1)
        ], name="discriminator")
        model.summary()
        return model

    def train_d(self, data, label):
        """
        data: 输入给判别器的数据，包括真实数据和生成的数据
        label: 数据的标签，真实数据的标签为1，生成数据的标签为0
        """
        # tf.GradientTape 提供了一个方便的方式来记录和计算梯度，是 TensorFlow 实现自动微分和反向传播的重要工具
        with tf.GradientTape() as tape:
            # 判别器对输入数据的预测结果
            pred = self.d.call(data, training=True)
            # 判别器的损失
            loss = self.loss_func(label, pred)
        # 损失相对于判别器参数的梯度
        grads = tape.gradient(loss, self.d.trainable_variables)
        # 更新判别器的参数
        self.opt.apply_gradients(zip(grads, self.d.trainable_variables))
        return loss, binary_accuracy(label, pred)

    def train_g(self, d_label):
        """
        d_label: 生成器希望判别器认为生成数据为真的标签（全1）
        """
        # tf.GradientTape 提供了一个方便的方式来记录和计算梯度，是 TensorFlow 实现自动微分和反向传播的重要工具
        with tf.GradientTape() as tape:
            # 生成器生成的数据
            g_data = self.call(len(d_label), training=True)
            # 判别器对生成数据的预测结果
            pred = self.d.call(g_data, training=False)
            # 生成器的损失
            loss = self.loss_func(d_label, pred)
        # grads 损失相对于生成器参数的梯度 梯度下降
        # self.g.trainable_variables 生成器的可训练参数（即权重和偏置）
        grads = tape.gradient(loss, self.g.trainable_variables)
        # 更新生成器的参数
        self.opt.apply_gradients(zip(grads, self.g.trainable_variables))
        return loss, g_data, binary_accuracy(d_label, pred)

    def step(self, data):
        # 生成器的目标标签，全为1，因为生成器希望判别器认为生成的数据是真实的
        d_label = tf.ones((len(data) * 2, 1), tf.float32)  # let d think generated are real
        # d_label 调用生成器的训练函数 train_g，传入目标标签 d_label
        # train_g 返回生成器的损失 g_loss，生成的数据 g_data 和生成器的准确率 g_acc
        g_loss, g_data, g_acc = self.train_g(d_label)

        # 判别器的训练 接下来，判别器的目标是区分真实数据和生成的数据。所以判别器需要对一批混合了真实数据和生成数据的样本进行训练
        # 将真实数据与生成数据的标签连接起来
        d_label = tf.concat((tf.ones((len(data), 1), tf.float32), tf.zeros((len(g_data) // 2, 1), tf.float32)), axis=0)
        # 将其数据连接起来 g_data[:len(g_data) // 2] 取前32个
        data = tf.concat((data, g_data[:len(g_data) // 2]), axis=0)
        # 训练判别器 返回判别器的损失与准确率 将取到的数据与标签进行判别器的训练
        d_loss, d_acc = self.train_d(data, d_label)
        return d_loss, d_acc, g_loss, g_acc


def train(gan, epoch):
    t0 = time.time()
    for ep in range(epoch):
        for t, data in enumerate(get_real_data(DATA_DIM, BATCH_SIZE)):
            d_loss, d_acc, g_loss, g_acc = gan.step(data)
            if t % 400 == 0:
                t1 = time.time()
                print(
                    "ep={} | time={:.1f} | t={} | d_acc={:.2f} | g_acc={:.2f} | d_loss={:.2f} | g_loss={:.2f}".format(
                        ep, t1 - t0, t, d_acc.numpy(), g_acc.numpy(), d_loss.numpy(), g_loss.numpy(), ))
                t0 = t1
        save_gan(gan, ep)
    save_weights(gan)
    cvt_gif(gan, shrink=2)


if __name__ == "__main__":
    LATENT_DIM = 10
    DATA_DIM = 16
    BATCH_SIZE = 32
    EPOCH = 400

    FUNCTION = FunctionType()
    get_real_data = FUNCTION.get("sin")

    set_soft_gpu(True)
    m = GAN(LATENT_DIM, DATA_DIM)
    train(m, EPOCH)
