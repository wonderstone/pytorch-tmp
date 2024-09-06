import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os

# 定义生成器
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 28*28),
            nn.Tanh()  # 输出 [-1, 1] 之间的像素值
        )
    
    def forward(self, x):
        return self.model(x)

# 定义判别器
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28*28, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()  # 输出概率 [0, 1]
        )
    
    def forward(self, x):
        return self.model(x)

# 保存模型到本地
def save_model(generator, discriminator, epoch):
    torch.save(generator.state_dict(), f'GAN/models/generator_{epoch}.pth')
    torch.save(discriminator.state_dict(), f'GAN/models/discriminator_{epoch}.pth')
    print(f"Models saved at epoch {epoch}")
    
# 从本地加载模型
def load_model(generator, discriminator):
    # 如果本地有保存的模型，则加载, using relative path
    if os.path.exists('GAN/generator.pth') and os.path.exists('GAN/discriminator.pth'):
        generator.load_state_dict(torch.load('GAN/generator.pth'))
        discriminator.load_state_dict(torch.load('GAN/discriminator.pth'))
        print("Loaded pre-trained models.")
    else:
        print("No pre-trained models found, starting from scratch.")

if not os.path.exists('images'):
    os.makedirs('images')



if __name__ == '__main__':

    # 加载MNIST数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # 将像素值从 [0, 1] 映射到 [-1, 1]
    ])

    mnist = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    dataloader = DataLoader(mnist, batch_size=64, shuffle=True)

    # 初始化生成器和判别器
    generator = Generator()
    discriminator = Discriminator()

    # 尝试加载本地保存的模型
    load_model(generator, discriminator)

    # 损失函数和优化器
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002)

    # 训练GAN
    num_epochs = 20
    for epoch in range(num_epochs):
        for batch_idx, (real_images, _) in enumerate(dataloader):
            batch_size = real_images.size(0)

            # 训练判别器：最大化 log(D(x)) + log(1 - D(G(z)))
            real_images = real_images.view(batch_size, -1)
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)

            # 判别器在真实图像上的损失
            outputs = discriminator(real_images)
            d_loss_real = criterion(outputs, real_labels)

            # 生成假图像并计算判别器损失
            z = torch.randn(batch_size, 100)  # 随机噪声
            fake_images = generator(z)
            outputs = discriminator(fake_images.detach())
            d_loss_fake = criterion(outputs, fake_labels)

            # 判别器总损失
            d_loss = d_loss_real + d_loss_fake

            # 反向传播和优化
            optimizer_d.zero_grad()
            d_loss.backward()
            optimizer_d.step()

            # 训练生成器：最大化 log(D(G(z)))
            z = torch.randn(batch_size, 100)
            fake_images = generator(z)
            outputs = discriminator(fake_images)
            g_loss = criterion(outputs, real_labels)  # 生成器的目标是让判别器认为它生成的图像是真实的

            # 反向传播和优化
            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}')

        # 可视化生成的图像（保存到本地）
        if (epoch + 1) % 10 == 0:
            fake_images = fake_images.view(fake_images.size(0), 1, 28, 28)
            save_image(fake_images.data[:25], f'GAN/images/{epoch+1}.png', nrow=5, normalize=True)

        # 保存模型
        save_model(generator, discriminator, epoch)

    # 训练完成后保存模型
    torch.save(generator.state_dict(), 'GAN/generator.pth')
    torch.save(discriminator.state_dict(), 'GAN/discriminator.pth')
    print("Final models saved.")