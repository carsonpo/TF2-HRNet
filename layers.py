import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, add, concatenate, Conv2DTranspose


class BasicBlock(tf.keras.layers.Layer):
    def __init__(self, out_filters, strides=(1, 1)):
        super(BasicBlock, self).__init__()
        self.out_filters = out_filters
        self.strides = strides
        self.Conv1 = Conv2D(out_filters, 3, strides, padding='same',
                            use_bias=False, kernel_initializer='he_normal')
        self.Conv2 = Conv2D(out_filters, 3, strides, padding='same',
                            use_bias=False, kernel_initializer='he_normal')
        self.BatchNorm1 = BatchNormalization(axis=3)
        self.BatchNorm2 = BatchNormalization(axis=3)
        self.BatchNorm3 = BatchNormalization(axis=3)
        self.Relu = Activation('relu')

    @tf.function
    def call(self, inputs):
        x = self.Conv1(inputs)
        x = self.BatchNorm1(x)
        x = self.Relu(x)

        x = self.Conv2(x)
        x = self.BatchNorm2(x)

        x = add([x, inputs])

        return x


class BottleNeckBlock(tf.keras.layers.Layer):
    def __init__(self, out_filters, strides=(1, 1), with_conv_shortcut=False):
        super(BottleNeckBlock, self).__init__()
        expansion = 4
        de_filters = out_filters // expansion
        self.out_filters = out_filters
        self.strides = strides
        self.with_conv_shortcut = with_conv_shortcut
        self.Conv3x3 = Conv2D(out_filters, 3, strides, padding='same',
                              use_bias=False, kernel_initializer='he_normal')
        self.BatchNorm1 = BatchNormalization(axis=3)
        self.BatchNorm2 = BatchNormalization(axis=3)
        self.BatchNorm3 = BatchNormalization(axis=3)
        self.BatchNorm4 = BatchNormalization(axis=3)
        self.Relu1 = Activation('relu')
        self.Relu2 = Activation('relu')
        self.Relu3 = Activation('relu')
        self.Conv1 = Conv2D(de_filters, 1, use_bias=False,
                            kernel_initializer='he_normal')
        self.Conv2 = Conv2D(de_filters, 3, strides, padding='same',
                            use_bias=False, kernel_initializer='he_normal')
        self.Conv3 = Conv2D(de_filters, 1, use_bias=False,
                            kernel_initializer='he_normal')
        self.Conv4 = Conv2D(de_filters, 1, use_bias=False,
                            kernel_initializer='he_normal')

    @tf.function
    def call(self, inputs):
        x = self.Conv1(inputs)
        x = self.BatchNorm1(x)
        x = self.Relu1(x)

        x = self.Conv2(x)
        x = self.BatchNorm2(x)
        x = self.Relu2(x)

        x = self.Conv3(x)
        x = self.BatchNorm3(x)

        if self.with_conv_shortcut:
            residual = self.Conv4(inputs)
            residual = self.BatchNorm4(residual)
            x = add([x, residual])
        else:
            x = add([x, inputs])

        x = self.Relu3(x)

        return x


class StemNet(tf.keras.layers.Layer):
    def __init__(self, input_shape):
        super(StemNet, self).__init__()
        self.Conv1 = Conv2D(64, 3, strides=(2, 2), padding='same', use_bias=False,
                            kernel_initializer='he_normal', input_shape=input_shape)
        self.BatchNorm1 = BatchNormalization(axis=3)
        self.Relu1 = Activation('relu')

        self.BottleNeck1 = BottleNeckBlock(256, with_conv_shortcut=True)
        self.BottleNeck2 = BottleNeckBlock(256)
        self.BottleNeck3 = BottleNeckBlock(256)
        self.BottleNeck4 = BottleNeckBlock(256)

    @tf.function
    def call(self, inputs):
        x = self.Conv1(inputs)
        x = self.BatchNorm1(x)
        x = self.Relu1(x)

        x = self.BottleNeck1(x)
        x = self.BottleNeck2(x)
        x = self.BottleNeck3(x)
        x = self.BottleNeck4(x)

        return x


class TransitionLayer1(tf.keras.layers.Layer):
    def __init__(self, out_filters_list=[32, 64]):
        super(TransitionLayer1, self).__init__()
        self.out_filters_list = out_filters_list

        self.Conv0 = Conv2D(
            out_filters_list[0], 3, padding='same', use_bias=False, kernel_initializer='he_normal')
        self.Conv1 = Conv2D(out_filters_list[1], 3, strides=(
            2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')

        self.BatchNorm1 = BatchNormalization(axis=3)
        self.BatchNorm2 = BatchNormalization(axis=3)

        self.Relu1 = Activation('relu')
        self.Relu2 = Activation('relu')

    @tf.function
    def call(self, inputs):
        x0 = self.Conv0(inputs)
        x0 = self.BatchNorm1(x0)
        x0 = self.Relu1(x0)

        x1 = self.Conv1(x0)
        x1 = self.BatchNorm2(x1)
        x1 = self.Relu2(x1)

        return [x0, x1]


class TransitionLayer2(tf.keras.layers.Layer):
    def __init__(self, out_filters_list=[32, 64, 128]):
        super(TransitionLayer2, self).__init__()
        self.out_filters_list = out_filters_list

        self.Conv0 = Conv2D(
            out_filters_list[0], 3, padding='same', use_bias=False, kernel_initializer='he_normal')
        self.Conv1 = Conv2D(
            out_filters_list[1], 3, padding='same', use_bias=False, kernel_initializer='he_normal')
        self.Conv2 = Conv2D(out_filters_list[2], 3, strides=(
            2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')

        self.BatchNorm1 = BatchNormalization(axis=3)
        self.BatchNorm2 = BatchNormalization(axis=3)
        self.BatchNorm3 = BatchNormalization(axis=3)

        self.Relu1 = Activation('relu')
        self.Relu2 = Activation('relu')
        self.Relu3 = Activation('relu')

    @tf.function
    def call(self, inputs):
        x0 = self.Conv0(inputs[0])
        x0 = self.BatchNorm1(x0)
        x0 = self.Relu1(x0)

        x1 = self.Conv1(inputs[1])
        x1 = self.BatchNorm2(x1)
        x1 = self.Relu2(x1)

        x2 = self.Conv2(inputs[1])
        x2 = self.BatchNorm3(x2)
        x2 = self.Relu3(x2)

        return [x0, x1, x2]


class TransitionLayer3(tf.keras.layers.Layer):
    def __init__(self, out_filters_list=[32, 64, 128, 256]):
        super(TransitionLayer3, self).__init__()
        self.out_filters_list = out_filters_list

        self.Conv0 = Conv2D(
            out_filters_list[0], 3, padding='same', use_bias=False, kernel_initializer='he_normal')
        self.Conv1 = Conv2D(
            out_filters_list[1], 3, padding='same', use_bias=False, kernel_initializer='he_normal')
        self.Conv2 = Conv2D(
            out_filters_list[2], 3, padding='same', use_bias=False, kernel_initializer='he_normal')
        self.Conv3 = Conv2D(out_filters_list[3], 3, strides=(
            2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')

        self.BatchNorm1 = BatchNormalization(axis=3)
        self.BatchNorm2 = BatchNormalization(axis=3)
        self.BatchNorm3 = BatchNormalization(axis=3)
        self.BatchNorm4 = BatchNormalization(axis=3)

        self.Relu1 = Activation('relu')
        self.Relu2 = Activation('relu')
        self.Relu3 = Activation('relu')
        self.Relu4 = Activation('relu')

    @tf.function
    def call(self, inputs):
        x0 = self.Conv0(inputs[0])
        x0 = self.BatchNorm1(x0)
        x0 = self.Relu1(x0)

        x1 = self.Conv1(inputs[1])
        x1 = self.BatchNorm2(x1)
        x1 = self.Relu2(x1)

        x2 = self.Conv2(inputs[2])
        x2 = self.BatchNorm3(x2)
        x2 = self.Relu3(x2)

        x3 = self.Conv3(inputs[2])
        x3 = self.BatchNorm4(x3)
        x3 = self.Relu4(x3)

        return [x0, x1, x2, x3]


class FuseLayer1(tf.keras.layers.Layer):

    def __init__(self):
        super(FuseLayer1, self).__init__()
        self.Conv1 = Conv2D(32, 1, use_bias=False,
                            kernel_initializer='he_normal')
        self.Conv2 = Conv2D(64, 3, strides=(2, 2), padding='same',
                            use_bias=False, kernel_initializer='he_normal')

        self.BatchNorm1 = BatchNormalization(3)
        self.BatchNorm2 = BatchNormalization(3)

        self.ConvT1 = Conv2DTranspose(32, 3, strides=(
            2, 2), use_bias=False, kernel_initializer='he_normal', padding='same')

    @tf.function
    def call(self, x):

        x0_0 = x[0]

        x0_1 = self.Conv1(x[1])
        x0_1 = self.BatchNorm1(x0_1)
        x0_1 = self.ConvT1(x0_1)
        x0 = add([x0_0, x0_1])

        x1_0 = self.Conv2(x[0])
        x1_0 = self.BatchNorm2(x1_0)
        x1_1 = x[1]

        x1 = add([x1_0, x1_1])

        return [x0, x1]


class FuseLayer2(tf.keras.layers.Layer):
    def __init__(self):
        super(FuseLayer2, self).__init__()
        self.Conv1 = Conv2D(32, 1, use_bias=False,
                            kernel_initializer='he_normal', padding='same')
        self.Conv2 = Conv2D(32, 1, use_bias=False,
                            kernel_initializer='he_normal', padding='same')
        self.Conv3 = Conv2D(64, 3, strides=(2, 2), use_bias=False,
                            kernel_initializer='he_normal', padding='same')
        self.Conv4 = Conv2D(64, 1, use_bias=False,
                            kernel_initializer='he_normal', padding='same')
        self.Conv5 = Conv2D(32, 3, strides=(2, 2), use_bias=False,
                            kernel_initializer='he_normal', padding='same')
        self.Conv6 = Conv2D(128, 3, strides=(
            2, 2), use_bias=False, kernel_initializer='he_normal', padding='same')
        self.Conv7 = Conv2D(128, 3, strides=(
            2, 2), use_bias=False, kernel_initializer='he_normal', padding='same')

        self.BatchNorm1 = BatchNormalization(3)
        self.BatchNorm2 = BatchNormalization(3)
        self.BatchNorm3 = BatchNormalization(3)
        self.BatchNorm4 = BatchNormalization(3)
        self.BatchNorm5 = BatchNormalization(3)
        self.BatchNorm6 = BatchNormalization(3)
        self.BatchNorm7 = BatchNormalization(3)

        self.Relu1 = Activation('relu')

        self.ConvT1 = Conv2DTranspose(1, (1, 1), strides=(
            2, 2), use_bias=False, kernel_initializer='he_normal', padding='same')
        self.ConvT2 = Conv2DTranspose(1, (1, 1), strides=(
            4, 4), use_bias=False, kernel_initializer='he_normal', padding='same')
        self.ConvT3 = Conv2DTranspose(1, (1, 1), strides=(
            2, 2), use_bias=False, kernel_initializer='he_normal', padding='same')

    @tf.function
    def call(self, x):

        x0_0 = x[0]
        x0_1 = self.Conv1(x[1])
        x0_1 = self.BatchNorm1(x0_1)
        x0_1 = self.ConvT1(x0_1)

        x0_2 = self.Conv2(x[2])
        x0_2 = self.BatchNorm2(x0_2)
        x0_2 = self.ConvT2(x0_2)

        x0 = add([x0_0, x0_1, x0_2])

        x1_0 = self.Conv3(x[0])
        x1_0 = self.BatchNorm3(x1_0)
        x1_1 = x[1]

        x1_2 = self.Conv4(x[2])
        x1_2 = self.BatchNorm4(x1_2)
        x1_2 = self.ConvT3(x1_2)

        x1 = add([x1_0, x1_1, x1_2])

        x2_0 = self.Conv5(x[0])
        x2_0 = self.BatchNorm5(x2_0)
        x2_0 = self.Relu1(x2_0)
        x2_0 = self.Conv6(x2_0)
        x2_0 = self.BatchNorm6(x2_0)

        x2_1 = self.Conv7(x[1])
        x2_1 = self.BatchNorm7(x2_1)

        x2_2 = x[2]
        x2 = add([x2_0, x2_1, x2_2])

        return [x0, x1, x2]


class FuseLayer3(tf.keras.layers.Layer):

    def __init__(self):
        super(FuseLayer3, self).__init__()
        self.Conv1 = Conv2D(32, 1, use_bias=False,
                            kernel_initializer='he_normal')
        self.Conv2 = Conv2D(32, 1, use_bias=False,
                            kernel_initializer='he_normal')
        self.Conv3 = Conv2D(32, 1, use_bias=False,
                            kernel_initializer='he_normal')

        self.BatchNorm1 = BatchNormalization(3)
        self.BatchNorm2 = BatchNormalization(3)
        self.BatchNorm3 = BatchNormalization(3)

        self.ConvT1 = Conv2DTranspose(32, 3, strides=(
            2, 2), use_bias=False, kernel_initializer='he_normal', padding='same')
        self.ConvT2 = Conv2DTranspose(32, 3, strides=(
            4, 4), use_bias=False, kernel_initializer='he_normal', padding='same')
        self.ConvT3 = Conv2DTranspose(32, 3, strides=(
            8, 8), use_bias=False, kernel_initializer='he_normal', padding='same')

    @tf.function
    def call(self, x):

        x0_0 = x[0]
        x0_1 = self.Conv1(x[1])
        x0_1 = self.BatchNorm1(x0_1)
        x0_1 = self.ConvT1(x0_1)

        x0_2 = self.Conv2(x[2])
        x0_2 = self.BatchNorm2(x0_2)
        x0_2 = self.ConvT2(x0_2)

        x0_3 = self.Conv3(x[3])
        x0_3 = self.BatchNorm3(x0_3)
        x0_3 = self.ConvT3(x0_3)

        x0 = concatenate([x0_0, x0_1, x0_2, x0_3])

        return x0


class FinalLayer(tf.keras.layers.Layer):
    def __init__(self, classes=1):
        super(FinalLayer, self).__init__()
        self.ConvT = Conv2DTranspose(16, 2, strides=(
            2, 2), use_bias=False, kernel_initializer='he_normal', padding='same')
        self.Conv = Conv2D(classes, 1, use_bias=False,
                           kernel_initializer='he_normal')
        self.BatchNorm = BatchNormalization(3)
        self.Relu = Activation('relu')

    @tf.function
    def call(self, x):

        x = self.ConvT(x)
        x = self.Conv(x)
        x = self.BatchNorm(x)
        x = self.Relu(x)

        return x


class Branch(tf.keras.layers.Layer):
    def __init__(self, filters=32):
        super(Branch, self).__init__()
        self.BasicBlock1 = BasicBlock(filters)
        self.BasicBlock2 = BasicBlock(filters)
        self.BasicBlock3 = BasicBlock(filters)
        self.BasicBlock4 = BasicBlock(filters)

    @tf.function
    def call(self, x):
        x = self.BasicBlock1(x)
        x = self.BasicBlock2(x)
        x = self.BasicBlock3(x)
        x = self.BasicBlock4(x)

        return x
