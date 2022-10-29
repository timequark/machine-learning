import os
import logging
import pickle
import numpy as np
from keras import backend as K
from keras.layers import Layer
from keras import activations
from keras import utils
from keras.datasets import cifar10
from keras.models import Model
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

'''
https://keras.io/zh/examples/cifar10_cnn_capsule/
https://github.com/bojone/Capsule/

[重要的参考资源]
https://paperswithcode.com/
https://paperswithcode.com/dataset/cifar-10
https://paperswithcode.com/sota/image-classification-on-cifar-10

'''

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s",
                    datefmt = '%Y-%m-%d  %H:%M:%S %a'    #注意月份和天数不要搞乱了，这里的格式化符与time模块相同
                    )

# 挤压函数
# 我们在此使用 0.5，而不是 Hinton 论文中给出的 1
# 如果为 1，则向量的范数将被缩小。
# 如果为 0.5，则当原始范数小于 0.5 时，范数将被放大，
# 当原始范数大于 0.5 时，范数将被缩小。
def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm) / (0.5 + s_squared_norm)
    return scale * x


# 使用自定义的 softmax 函数，而非 K.softmax，
# 因为 K.softmax 不能指定轴。
def softmax(x, axis=-1):
    ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
    return ex / K.sum(ex, axis=axis, keepdims=True)


# 定义 margin loss，类似于 hinge loss
def margin_loss(y_true, y_pred):
    lamb, margin = 0.5, 0.1
    return K.sum(y_true * K.square(K.relu(1 - margin - y_pred)) + lamb * (
        1 - y_true) * K.square(K.relu(y_pred - margin)), axis=-1)


class Capsule(Layer):
    """一个由纯 Keras 实现的 Capsule 网络。
    总共有两个版本的 Capsule。
    一个类似于全连接层 (用于固定尺寸的输入)，
    另一个类似于时序分布全连接层 (用于变成输入)。

    Capsure 的输入尺寸必须为 (batch_size,
                             input_num_capsule,
                             input_dim_capsule
                            )
    以及输出尺寸必须为 (batch_size,
                      num_capsule,
                      dim_capsule
                     )

    Capsule 实现来自于 https://github.com/bojone/Capsule/
    Capsule 论文: https://arxiv.org/abs/1710.09829
    """

    def __init__(self,
                 num_capsule,
                 dim_capsule,
                 routings=3,
                 share_weights=True,
                 activation='squash',
                 **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'squash':
            self.activation = squash
        else:
            self.activation = activations.get(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        print('input_shape: ', input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.kernel = self.add_weight(name='capsule_kernel',
                                          shape=(1, input_dim_capsule,
                                                 self.num_capsule * self.dim_capsule),
                                          initializer='glorot_uniform',
                                          trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.kernel = self.add_weight(name='capsule_kernel',
                                          shape=(input_num_capsule, input_dim_capsule,
                                                 self.num_capsule * self.dim_capsule),
                                          initializer='glorot_uniform',
                                          trainable=True)

    def call(self, u_vecs):
        """遵循 Hinton 论文中的路由算法，
        但是将 b = b + <u,v> 替换为 b = <u,v>。

        这一改变将提升 Capsule 的特征表示能力。

        然而，你仍可以将
            b = K.batch_dot(outputs, hat_inputs, [2, 3])
        替换为
            b += K.batch_dot(outputs, hat_inputs, [2, 3])
        来实现一个标准的路由。
        """

        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.kernel)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.kernel, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        #final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:, :, :, 0]) # shape = [None, num_capsule, input_num_capsule]
        
        # for i in range(self.routings):
        #     c = softmax(b, 1)
        #     o = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))
        #     print('b shape: ', b.shape, ', c shape: ', c.shape, ', x shape: ', o.shape, ', y shape: ', u_hat_vecs.shape)
        #     if i < self.routings - 1:
        #         b = K.batch_dot(o, u_hat_vecs, [2, 3])
        #         if K.backend() == 'theano':
        #             o = K.sum(o, axis=1)

        # return o
        
        for i in range(self.routings):
            c = softmax(b, 1)
            # o = K.batch_dot(c, u_hat_vecs, [2, 2])
            o = tf.einsum('bin,binj->bij', c, u_hat_vecs)
            if K.backend() == 'theano':
                o = K.sum(o, axis=1)
            if i < self.routings - 1:
                o = K.l2_normalize(o, -1)
                # b = K.batch_dot(o, u_hat_vecs, [2, 3])
                b = tf.einsum('bij,binj->bin', o, u_hat_vecs)
                if K.backend() == 'theano':
                    b = K.sum(b, axis=1)

        return self.activation(o)

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)

def load_cfar10_batch(cifar10Dir, batchfile, gray:bool = False):
    with open(os.path.join(cifar10Dir, batchfile), 'rb') as f:
        batch = pickle.load(f, encoding='latin1')
    features = batch['data'].reshape( (len(batch['data']), 3, 32, 32) ).transpose(0, 2, 3, 1)
    if gray:
        features = features.max(axis=3) # 灰度处理，https://blog.csdn.net/qq_41915623/article/details/124547004
    labels = batch['labels']
    
    return features, labels

def test_model(cifar10Dir):
    gray = False
    X_train, y_train = load_cfar10_batch(cifar10Dir, 'data_batch_1', gray)
    features, labels = load_cfar10_batch(cifar10Dir, 'data_batch_2', gray)
    X_train, y_train = np.concatenate([X_train, features]), np.concatenate([y_train, labels])
    features, labels = load_cfar10_batch(cifar10Dir, 'data_batch_3', gray)
    X_train, y_train = np.concatenate([X_train, features]), np.concatenate([y_train, labels])
    features, labels = load_cfar10_batch(cifar10Dir, 'data_batch_4', gray)
    X_train, y_train = np.concatenate([X_train, features]), np.concatenate([y_train, labels])
    features, labels = load_cfar10_batch(cifar10Dir, 'data_batch_5', gray)
    X_train, y_train = np.concatenate([X_train, features]), np.concatenate([y_train, labels])
    
    X_test, y_test   = load_cfar10_batch(cifar10Dir, 'test_batch', gray)
    
    # 重塑
    if gray:
        X_train_rows = X_train.reshape(X_train.shape[0], 32*32)
        X_test_rows  = X_test.reshape(X_test.shape[0], 32*32)
    else:
        X_train_rows = X_train.reshape(X_train.shape[0], 32*32*3)
        X_test_rows  = X_test.reshape(X_test.shape[0], 32*32*3)
    
    # 归一化
    from sklearn.preprocessing import MinMaxScaler
    scal = MinMaxScaler()
    X_train_rows = scal.fit_transform(X_train_rows)
    X_test_rows = scal.fit_transform(X_test_rows)
    X_train_m = X_train_rows.reshape( X_train_rows.shape[0], 3, 32, 32).transpose(0, 2, 3, 1)
    X_test_m = X_test_rows.reshape( X_test_rows.shape[0], 3, 32, 32).transpose(0, 2, 3, 1)
    y_train_m = to_categorical(y_train, 10) # 将标签转换为One-hot编码
    y_test_m = to_categorical(y_test, 10) # 将标签转换为One-hot编码
    
    batch_size = 128
    num_classes = 10
    epochs = 100
    
    # (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    # x_train = x_train.astype('float32')
    # x_test = x_test.astype('float32')
    # x_train /= 255
    # x_test /= 255
    # y_train = utils.to_categorical(y_train, num_classes)
    # y_test = utils.to_categorical(y_test, num_classes)

    # 一个常规的 Conv2D 模型
    input_image = Input(shape=(None, None, 3))
    cnn = Conv2D(64, (3, 3), activation='relu')(input_image)
    cnn = Conv2D(64, (3, 3), activation='relu')(cnn)
    cnn = AveragePooling2D((2, 2))(cnn)
    cnn = Conv2D(128, (3, 3), activation='relu')(cnn)
    cnn = Conv2D(128, (3, 3), activation='relu')(cnn)


    """现在我们将其尺寸重新调整为 (batch_size, input_num_capsule, input_dim_capsule)，再连接一个 Capsule 网络。

    最终模型的输出为长度为 10 的 Capsure，其 dim=16。

    Capsule 的长度表示为 proba，
    因此问题变成了一个『10个二分类』的问题。
    """

    cnn = Reshape((-1, 128))(cnn)
    capsule = Capsule(10, 16, 3, True)(cnn)
    output = Lambda(lambda z: K.sqrt(K.sum(K.square(z), 2)))(capsule)
    
    model = Model(inputs=input_image, outputs=output)
    model.compile(loss=margin_loss, optimizer='adam', metrics=['accuracy']) # 使用 margin loss
    model.summary()

    # 可以比较有无数据增益对应的性能
    data_augmentation = False

    if not data_augmentation:
        print('Not using data augmentation.')
        history = model.fit(
            X_train_m,
            y_train_m,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_test_m, y_test_m),
            shuffle=True)
    else:
        print('Using real-time data augmentation.')
        # 这一步将进行数据处理和实时数据增益:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # 将整个数据集的均值设为 0
            samplewise_center=False,  # 将每个样本的均值设为 0
            featurewise_std_normalization=False,  # 将输入除以整个数据集的标准差
            samplewise_std_normalization=False,  # 将输入除以其标准差
            zca_whitening=False,  # 运用 ZCA 白化
            zca_epsilon=1e-06,  # ZCA 白化的 epsilon值
            rotation_range=0,  # 随机旋转图像范围 (角度, 0 to 180)
            width_shift_range=0.1,  # 随机水平移动图像 (总宽度的百分比)
            height_shift_range=0.1,  # 随机垂直移动图像 (总高度的百分比)
            shear_range=0.,  # 设置随机裁剪范围
            zoom_range=0.,  # 设置随机放大范围
            channel_shift_range=0.,  # 设置随机通道切换的范围
            # 设置填充输入边界之外的点的模式
            fill_mode='nearest',
            cval=0.,  # 在 fill_mode = "constant" 时使用的值
            horizontal_flip=True,  # 随机水平翻转图像
            vertical_flip=False,  # 随机垂直翻转图像
            # 设置缩放因子 (在其他转换之前使用)
            rescale=None,
            # 设置将应用于每一个输入的函数
            preprocessing_function=None,
            # 图像数据格式，"channels_first" 或 "channels_last" 之一
            data_format=None,
            # 保留用于验证的图像比例（严格在0和1之间）
            validation_split=0.0)

        # 计算特征标准化所需的计算量
        # (如果应用 ZCA 白化，则为 std，mean和主成分)。
        datagen.fit(X_train_m)

        # 利用由 datagen.flow() 生成的批来训练模型。
        history = model.fit_generator(
            datagen.flow(X_train_m, y_train_m, batch_size=batch_size),
            epochs=epochs,
            validation_data=(X_test_m, y_test_m),
            workers=4)
    
    print(history.history['val_acc'])

def test():
    input_shape = (4, 28, 28, 3)
    x = tf.random.normal(input_shape)
    y = tf.keras.layers.Conv2D(5, 3, activation='relu', input_shape=input_shape[1:])(x)
    print(y.shape) # output: (4, 26, 26, 5)
    
if __name__ == '__main__':
    # test()
    test_model('/data-warehouse/machine-learning-dataset/cifar-10-python/cifar-10-batches-py')
    logging.info('over')
