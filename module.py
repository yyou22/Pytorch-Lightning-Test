import pytorch_lightning as pl
import torch
from torchmetrics import Accuracy

from cifar10_models.densenet import densenet121, densenet161, densenet169
from cifar10_models.googlenet import googlenet
from cifar10_models.inception import inception_v3
from cifar10_models.mobilenetv2 import mobilenet_v2
from cifar10_models.resnet import resnet18, resnet34, resnet50
from cifar10_models.vgg import vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn
from schduler import WarmupCosineLR

from trades import trades_loss

all_classifiers = {
    "vgg11_bn": vgg11_bn(),
    "vgg13_bn": vgg13_bn(),
    "vgg16_bn": vgg16_bn(),
    "vgg19_bn": vgg19_bn(),
    "resnet18": resnet18(),
    "resnet34": resnet34(),
    "resnet50": resnet50(),
    "densenet121": densenet121(),
    "densenet161": densenet161(),
    "densenet169": densenet169(),
    "mobilenet_v2": mobilenet_v2(),
    "googlenet": googlenet(),
    "inception_v3": inception_v3(),
}


class CIFAR10Module(pl.LightningModule):
    def __init__(self, hparams_):
        super().__init__()
        self.hparams_ = hparams_

        #self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task='multiclass', num_classes=4)

        self.model = all_classifiers[self.hparams_.classifier]

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.hparams_.learning_rate,
            weight_decay=self.hparams_.weight_decay,
            momentum=0.9,
            nesterov=True,
        )

    def forward(self, batch):
        images, labels = batch
        predictions = self.model(images)
        #loss = self.criterion(predictions, labels)
        loss = trades_loss(model=model,
                           x_natural=images,
                           y=labels,
                           optimizer=optimizer,
                           step_size=0.007,
                           epsilon=0.031,
                           perturb_steps=10,
                           beta=1.0)
        self.accuracy.update(predictions, labels)
        accuracy = self.accuracy
        return loss, accuracy * 100

    def training_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("loss/train", loss)
        self.log("acc/train", accuracy)
        return loss

    def validation_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("loss/val", loss)
        self.log("acc/val", accuracy)

    def test_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        self.log("acc/test", accuracy)

    def configure_optimizers(self):
        #optimizer = torch.optim.SGD(
            #self.model.parameters(),
            #lr=self.hparams.learning_rate,
            #weight_decay=self.hparams.weight_decay,
            #momentum=0.9,
            #nesterov=True,
        #)
        total_steps = self.hparams_.max_epochs * len(self.train_dataloader())
        scheduler = {
            "scheduler": WarmupCosineLR(
                self.optimizer, warmup_epochs=total_steps * 0.3, max_epochs=total_steps
            ),
            "interval": "step",
            "name": "learning_rate",
        }
        return [self.optimizer], [scheduler]
