import tensorflow as tf
from layers import (
    BasicBlock,
    Branch,
    FinalLayer,
    FuseLayer1,
    FuseLayer2,
    FuseLayer3,
    StemNet,
    TransitionLayer1,
    TransitionLayer2,
    TransitionLayer3
)


class HRNet(tf.keras.models.Model):

    def __init__(self, batch_size=1, height=64, width=64, channel=28, classes=1):

        super(HRNet, self).__init__()
        self.StemNet = StemNet((None, height, width, channel))
        self.TransitionLayer1 = TransitionLayer1()
        self.TransitionLayer2 = TransitionLayer2()
        self.TransitionLayer3 = TransitionLayer3()

        self.FuseLayer1 = FuseLayer1()
        self.FuseLayer2 = FuseLayer2()
        self.FuseLayer3 = FuseLayer3()

        self.FinalLayer = FinalLayer(classes)

        self.Branch1_0 = Branch(32)
        self.Branch1_1 = Branch(64)

        self.Branch2_0 = Branch(32)
        self.Branch2_1 = Branch(64)
        self.Branch2_2 = Branch(128)

        self.Branch3_0 = Branch(32)
        self.Branch3_1 = Branch(64)
        self.Branch3_2 = Branch(128)
        self.Branch3_3 = Branch(256)

    @tf.function
    def call(self, x):
        x = self.StemNet(x)

        x = self.TransitionLayer1(x)
        x0 = self.Branch1_0(x[0])
        x1 = self.Branch1_1(x[1])
        x = self.FuseLayer1([x0, x1])

        x = self.TransitionLayer2(x)
        x0 = self.Branch2_0(x[0])
        x1 = self.Branch2_1(x[1])
        x2 = self.Branch2_2(x[2])
        x = self.FuseLayer2([x0, x1, x2])

        x = self.TransitionLayer3(x)
        x0 = self.Branch3_0(x[0])
        x1 = self.Branch3_1(x[1])
        x2 = self.Branch3_2(x[2])
        x3 = self.Branch3_3(x[3])

        x = self.FuseLayer3([x0, x1, x2, x3])

        out = self.FinalLayer(x)

        return out
