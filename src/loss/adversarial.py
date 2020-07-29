import torch
import torch.nn as nn

from utils import interact

import torch.cuda.amp as amp

class Adversarial(nn.modules.loss._Loss):
    # pure loss function without saving & loading option
    # but trains deiscriminator
    def __init__(self, args, model, optimizer):
        super(Adversarial, self).__init__()
        self.args = args
        self.model = model.model
        self.optimizer = optimizer
        self.scaler = amp.GradScaler(
            init_scale=self.args.init_scale,
            enabled=self.args.amp
        )

        self.gan_k = 1

        self.BCELoss = nn.BCEWithLogitsLoss()

    def forward(self, fake, real, training=False):
        if training:
            # update discriminator
            fake_detach = fake.detach()
            for _ in range(self.gan_k):
                self.optimizer.D.zero_grad()
                # d: B x 1 tensor
                with amp.autocast(self.args.amp):
                    d_fake = self.model.D(fake_detach)
                    d_real = self.model.D(real)

                    label_fake = torch.zeros_like(d_fake)
                    label_real = torch.ones_like(d_real)

                    loss_d = self.BCELoss(d_fake, label_fake) + self.BCELoss(d_real, label_real)

                self.scaler.scale(loss_d).backward(retain_graph=False)
                self.scaler.step(self.optimizer.D)
                self.scaler.update()
        else:
            d_real = self.model.D(real)
            label_real = torch.ones_like(d_real)

        # update generator (outside here)
        d_fake_bp = self.model.D(fake)
        loss_g = self.BCELoss(d_fake_bp, label_real)

        return loss_g