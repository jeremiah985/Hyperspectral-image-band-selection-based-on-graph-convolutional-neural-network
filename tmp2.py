import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import scipy.io as sio
import numpy as np
from sklearn.metrics import accuracy_score as acc, normalized_mutual_info_score as nmi
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

# 1. 定义Conv2dSamePad类
class Conv2dSamePad(nn.Module):
    def __init__(self, kernel_size, stride):
        super(Conv2dSamePad, self).__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, (list, tuple)) else [kernel_size, kernel_size]
        self.stride = stride if isinstance(stride, (list, tuple)) else [stride, stride]

    def forward(self, x):
        in_height, in_width = x.size(2), x.size(3)
        pad_height = self.kernel_size[0] - self.stride[0]
        pad_width = self.kernel_size[1] - self.stride[1]
        pad_top, pad_bottom = pad_height // 2, pad_height - pad_height // 2
        pad_left, pad_right = pad_width // 2, pad_width - pad_width // 2
        return x[:, :, pad_top:in_height - pad_bottom, pad_left:in_width - pad_right]

# 2. 定义ConvTranspose2dSamePad类
class ConvTranspose2dSamePad(nn.Module):
    def __init__(self, kernel_size, stride):
        super(ConvTranspose2dSamePad, self).__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, (list, tuple)) else [kernel_size, kernel_size]
        self.stride = stride if isinstance(stride, (list, tuple)) else [stride, stride]

    def forward(self, x):
        in_height, in_width = x.size(2), x.size(3)
        pad_height = self.kernel_size[0] - self.stride[0]
        pad_width = self.kernel_size[1] - self.stride[1]
        pad_top, pad_bottom = pad_height // 2, pad_height - pad_height // 2
        pad_left, pad_right = pad_width // 2, pad_width - pad_width // 2
        return x[:, :, pad_top:in_height - pad_bottom, pad_left:in_width - pad_right]

# 3. 定义自编码器类
class ConvAE(nn.Module):
    def __init__(self, channels, kernels):
        super(ConvAE, self).__init__()
        self.encoder = nn.Sequential()
        for i in range(1, len(channels)):
            self.encoder.add_module(f'pad{i}', Conv2dSamePad(kernels[i - 1], 2))
            self.encoder.add_module(f'conv{i}', nn.Conv2d(channels[i - 1], channels[i], kernel_size=kernels[i - 1], stride=2))
            self.encoder.add_module(f'relu{i}', nn.ReLU(True))

        self.decoder = nn.Sequential()
        channels = list(reversed(channels))
        kernels = list(reversed(kernels))
        for i in range(len(channels) - 1):
            self.decoder.add_module(f'deconv{i + 1}', nn.ConvTranspose2d(channels[i], channels[i + 1], kernel_size=kernels[i], stride=2))
            self.decoder.add_module(f'padd{i}', ConvTranspose2dSamePad(kernels[i], 2))
            self.decoder.add_module(f'relud{i}', nn.ReLU(True))

    def forward(self, x):
        h = self.encoder(x)
        y = self.decoder(h)
        return y

# 4. 定义SelfExpression类
class SelfExpression(nn.Module):
    def __init__(self, n):
        super(SelfExpression, self).__init__()
        self.Coefficient = nn.Parameter(1.0e-8 * torch.ones(n, n, dtype=torch.float32), requires_grad=True)

    def forward(self, x):
        y = torch.matmul(self.Coefficient, x)
        return y

# 5. 定义DSCNet类
class DSCNet(nn.Module):
    def __init__(self, channels, kernels, num_sample):
        super(DSCNet, self).__init__()
        self.n = num_sample
        self.ae = ConvAE(channels, kernels)
        self.self_expression = SelfExpression(self.n)

    def forward(self, x):
        z = self.ae.encoder(x)
        shape = z.shape
        z = z.view(self.n, -1)
        z_recon = self.self_expression(z)
        z_recon_reshape = z_recon.view(shape)
        x_recon = self.ae.decoder(z_recon_reshape)
        return x_recon, z, z_recon

    def loss_fn(self, x, x_recon, z, z_recon, weight_coef, weight_selfExp):
        loss_ae = F.mse_loss(x_recon, x, reduction='sum')
        loss_coef = torch.sum(torch.pow(self.self_expression.Coefficient, 2))
        loss_selfExp = F.mse_loss(z_recon, z, reduction='sum')
        loss = loss_ae + weight_coef * loss_coef + weight_selfExp * loss_selfExp
        return loss

# 6. 定义光谱聚类函数
def spectral_clustering(C, K, dim_subspace, alpha, ro):
    C = 0.5 * (C + C.T)
    r = ro * K + 1
    U, S, _ = np.linalg.svd(C)
    U = U[:, :r] @ np.diag(np.sqrt(S[:r]))
    U = normalize(U, norm='l2', axis=1)
    Z = U @ U.T
    Z = Z * (Z > alpha)
    spectral = KMeans(n_clusters=K).fit(Z)
    y_pred = spectral.labels_
    return y_pred

# 7. 定义训练函数
def train_with_band_selection(model, x, y, epochs, lr=1e-3, weight_coef=1.0, weight_selfExp=150, device='cuda',
                              alpha=0.04, dim_subspace=12, ro=8, show=10):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32, device=device)
    x = x.to(device)
    if isinstance(y, torch.Tensor):
        y = y.to('cpu').numpy()
    K = len(np.unique(y))

    selected_bands = []

    for epoch in range(epochs):
        x_recon, z, z_recon = model(x)
        loss = model.loss_fn(x, x_recon, z, z_recon, weight_coef=weight_coef, weight_selfExp=weight_selfExp)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % show == 0 or epoch == epochs - 1:
            C = model.self_expression.Coefficient.detach().to('cpu').numpy()
            y_pred = spectral_clustering(C, K, dim_subspace, alpha, ro)
            print('Epoch %02d: loss=%.4f, acc=%.4f, nmi=%.4f' %
                  (epoch, loss.item() / y_pred.shape[0], acc(y, y_pred), nmi(y, y_pred)))

            # 选择波段
            band_indices = np.argmax(np.abs(C), axis=0)
            selected_bands.append(band_indices)

    return selected_bands

# 8. 主程序
if __name__ == "__main__":
    import os

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
    # 加载Indian Pines数据集
    data_mat = sio.loadmat('indianpines_dataset.mat')
    labels_mat = sio.loadmat('indianpines_gt.mat')

    data = data_mat['pixels']
    labels = labels_mat['pixels']

    print("labels的初始维度：",labels.shape)
# 交换标签的第一维度和第二维度
    labels = np.transpose(labels, (1, 0))
    x, y = data.reshape((-1, 1, 145, 145)), labels
    y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]


    # 网络和优化参数
    num_sample = x.shape[0]
    channels = [1, 30, 60]  # 可以调整
    kernels = [7, 5]  # 可以调整
    epochs = 50
    weight_coef = 1.0
    weight_selfExp = 150

    # 聚类后处理参数
    alpha = 0.04  # C的阈值
    dim_subspace = 12  # 每个子空间的维度
    ro = 8  #

    # 构建并训练模型
    dscnet = DSCNet(num_sample=num_sample, channels=channels, kernels=kernels)
    dscnet.to(device)

    selected_bands = train_with_band_selection(dscnet, x, y, epochs=epochs, weight_coef=weight_coef, weight_selfExp=weight_selfExp,
                                               alpha=alpha, dim_subspace=dim_subspace, ro=ro, show=10, device=device)

    # 输出最终选择的波段
    print("Final selected bands:", selected_bands[-1])

    # 保存模型
    if not os.path.exists('results'):
        os.makedirs('results')
    torch.save(dscnet.state_dict(), 'results/dscnet_indian_pines.ckp')
