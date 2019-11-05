import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
from nets import DNet, GNet
from dataset import Datasets


class Trainer:
    def __init__(self, save_net_path, save_image_path, dataset_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_net_path = save_net_path
        self.save_image_path = save_image_path
        self.d_net = DNet().to(self.device)
        self.g_net = GNet().to(self.device)
        self.train_data = DataLoader(Datasets(dataset_path), batch_size=100, shuffle=True)
        self.loss_fn = nn.BCELoss()
        self.g_net_optimizer = torch.optim.Adam(self.g_net.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_net_optimizer = torch.optim.Adam(self.d_net.parameters(), lr=0.0002, betas=(0.5, 0.999))
        if os.path.exists(os.path.join(self.save_net_path, "d_net.pth")):
            self.d_net.load_state_dict(torch.load(os.path.join(self.save_net_path, "d_net.pth")))
            self.g_net.load_state_dict(torch.load(os.path.join(self.save_net_path, "g_net.pth")))
        self.d_net.train()
        self.g_net.train()

    def train(self):
        epoch = 1
        d_loss_new = 100000
        g_loss_new = 100000
        while True:
            for i, real_image in enumerate(self.train_data):
                real_image = real_image.to(self.device)
                real_label = torch.ones(real_image.size(0), 1, 1, 1).to(self.device)
                fake_label = torch.zeros(real_image.size(0), 1, 1, 1).to(self.device)

                d_real_out = self.d_net(real_image)
                d_loss_real = self.loss_fn(d_real_out, real_label)

                z = torch.randn(real_image.size(0), 128, 1, 1).to(self.device)
                d_fake_img = self.g_net(z)
                d_fake_out = self.d_net(d_fake_img)
                d_loss_fake = self.loss_fn(d_fake_out, fake_label)

                # 训练判别器
                d_loss = d_loss_real + d_loss_fake
                self.d_net_optimizer.zero_grad()
                d_loss.backward()
                self.d_net_optimizer.step()

                # 训练生成器,重新从正太分别取数据提高多样性
                z = torch.randn(real_image.size(0), 128, 1, 1).to(self.device)
                g_fake_img = self.g_net(z)
                g_fake_out = self.d_net(g_fake_img)
                g_loss = self.loss_fn(g_fake_out, real_label)
                self.g_net_optimizer.zero_grad()
                g_loss.backward()
                self.g_net_optimizer.step()

                if i % 10 == 0:
                    print("epoch:{},i:{},d_loss{:.6f},g_loss:{:.6f},"
                          "d_real:{:.3f},d_fake:{:.3f}".format(epoch, i, d_loss.item(), g_loss.item(),
                                                               d_real_out.detach().mean(), d_fake_out.detach().mean()))
                    save_image(real_image, "{}/{}-{}-real_img.jpg".format(self.save_image_path, epoch, i), 10,
                               normalize=True, scale_each=True)
                    save_image(g_fake_img, "{}/{}-{}-fake_img.jpg".format(self.save_image_path, epoch, i), 10,
                               normalize=True, scale_each=True)

                if d_loss.item() < d_loss_new:
                    d_loss_new = d_loss.item()
                    torch.save(self.d_net.state_dict(), os.path.join(self.save_net_path, "d_net.pth"))

                if g_loss.item() < g_loss_new:
                    g_loss_new = g_loss.item()
                    torch.save(self.g_net.state_dict(), os.path.join(self.save_net_path, "g_net.pth"))
            epoch += 1


if __name__ == '__main__':
    trainer = Trainer("models/", "images", r"C:\sample\Cartoon_faces\faces")
    trainer.train()
