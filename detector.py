import torch
from nets import GNet, DNet
from torchvision.utils import save_image
import os


class Detector:
    def __init__(self, net_path, save_image_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_image_path = save_image_path
        self.d_net = DNet().to(self.device)
        self.g_net = GNet().to(self.device)
        self.d_net.load_state_dict(torch.load(os.path.join(net_path, "d_net.pth")))
        self.g_net.load_state_dict(torch.load(os.path.join(net_path, "g_net.pth")))
        self.d_net.eval()
        self.g_net.eval()

    def detect(self):
        epoch_num = 1
        batch_size = 100
        random_num = 128
        for i in range(epoch_num):
            z = torch.randn(batch_size, random_num, 1, 1).to(self.device)
            with torch.no_grad():
                fake_img = self.g_net(z)
                fake_out = self.d_net(fake_img)

            save_image(fake_img, "{}/{}-{}-fake_img.jpg".format(self.save_image_path, "detector", 10), 10,
                       normalize=True, scale_each=True)
            print("fake_out:{}".format(fake_out.detach().mean()))


if __name__ == '__main__':
    detector = Detector("models/", r"detector_images")
    detector.detect()
