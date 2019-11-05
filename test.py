import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
from nets import DNet,GNet

batch_size = 100
num_epoch = 50
random_num = 128  # noise dimension

if __name__ == '__main__':

    if not os.path.exists("./cartoon_img"):
        os.mkdir("./cartoon_img")
    if not os.path.exists("./params"):
        os.mkdir("./params")
    def to_img(x):
        out = 0.5 * (x + 1)#[(0,1)+1=(1,2),(1,2)*0.5=(0.5,1)],
        # [(-1,1)+1=(0,2),(0,2)*0.5=(0,1)]
        out = out.clamp(0, 1)#Clamp函数可以将随机变化的数值
        # 限制在一个给定的区间[min, max]内,[0,1]
        return out

    dataloader = DataLoader(sapling_data, batch_size=batch_size,
                shuffle=True,num_workers=4,drop_last=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    d_net = D_Net().to(device)
    g_net = G_Net().to(device)
    d_net.train()
    g_net.train()

    d_net.load_state_dict(torch.load("./params/d_net2.pth"))
    g_net.load_state_dict(torch.load("./params/g_net2.pth"))

    loss_fn = nn.BCELoss()
    d_optimizer = torch.optim.Adam(
        d_net.parameters(), lr=0.0002, betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(
        g_net.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(num_epoch):
        for i, img in enumerate(dataloader):

            real_img = img.to(device)
            real_label = torch.ones(batch_size)\
                .view(-1,1,1,1).to(device)
            fake_label = torch.zeros(batch_size)\
                .view(-1,1,1,1).to(device)
            real_out = d_net(real_img)
            d_loss_real = loss_fn(real_out, real_label)
            real_scores = real_out
            z = torch.randn(batch_size, random_num,1,1).to(device)
            fake_img = g_net(z)
            fake_out = d_net(fake_img)
            d_loss_fake = loss_fn(fake_out, fake_label)
            fake_scores = fake_out
            d_loss = d_loss_real + d_loss_fake
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            z = torch.randn(batch_size, random_num,1,1).to(device)
            fake_img = g_net(z)
            output = d_net(fake_img)
            g_loss = loss_fn(output, real_label)
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()
            if i%100 == 0:
                print('Epoch [{}/{}], d_loss: {:.3f}, g_loss: {:.3f} '
                      'D real: {:.3f}, D fake: {:.3f}'
                      .format(epoch, num_epoch, d_loss, g_loss,
                              real_scores.data.mean(), fake_scores.data.mean()))

        images = to_img(fake_img.cpu().data)
        show_images = images.permute([0,2,3,1])
        # show_images = torch.transpose(images,1,3)
        # plt.imshow(show_images[0])
        # plt.pause(1)

        fake_images = to_img(fake_img.cpu().data)
        # normalize=True ，会将图片的像素值归一化处理。
        # scale_each=True ，每个图片独立归一化，
        # 而不是根据所有图片的像素最大最小值来规范化
        save_image(fake_images, './cartoon_img2/{}-fake_images.png'
        .format(epoch + 1),nrow=10,normalize=True,scale_each=True)

        real_images = to_img(real_img.cpu().data)
        save_image(real_images, './cartoon_img2/{}-real_images.png'
        .format(epoch + 1), nrow=10,normalize=True,scale_each=True)

        torch.save(d_net.state_dict(), "./params/d_net.pth")
        torch.save(g_net.state_dict(), "./params/g_net.pth")


