import torch
import random
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import torchvision.datasets
from PIL import Image

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True


MNIST_train = torchvision.datasets.MNIST('./', download=True, train=True)
MNIST_test = torchvision.datasets.MNIST('./', download=True, train=False)

X_train = MNIST_train.train_data
y_train = MNIST_train.train_labels
X_test = MNIST_test.test_data
y_test = MNIST_test.test_labels

X_train = X_train.unsqueeze(1).float()
X_test = X_test.unsqueeze(1).float()


class LeNet5(torch.nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        self.conv1 = torch.nn.Conv2d(
            in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.act1 = torch.nn.Tanh()
        self.pool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = torch.nn.Conv2d(
            in_channels=6, out_channels=16, kernel_size=5, padding=0)
        self.act2 = torch.nn.Tanh()
        self.pool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)

        self.fc1 = torch.nn.Linear(5 * 5 * 16, 120)
        self.act3 = torch.nn.Tanh()

        self.fc2 = torch.nn.Linear(120, 84)
        self.act4 = torch.nn.Tanh()

        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.act2(x)
        x = self.pool2(x)

        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))

        x = self.fc1(x)
        x = self.act3(x)
        x = self.fc2(x)
        x = self.act4(x)
        x = self.fc3(x)

        return x


def plot_image(pixels: np.array):
    plt.imshow(pixels.reshape((28, 28)), cmap='gray')
    plt.show()


def mlp_digits_predict(image_file):
    image_size = 28
    img = keras.preprocessing.image.load_img(image_file, target_size=(image_size, image_size), color_mode='grayscale')
    img_arr = np.expand_dims(img, axis=0)
    img_arr = img_arr.reshape(img_arr.shape[0], 1, 28, 28).astype('float32')
    # print(img_arr)
    return img_arr


lenet5 = LeNet5()

loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lenet5.parameters(), lr=1.0e-3)

batch_size = 100

test_accuracy_history = []
test_loss_history = []

# for epoch in range(10000):
for epoch in range(50):
    order = np.random.permutation(len(X_train))
    for start_index in range(0, len(X_train), batch_size):
        optimizer.zero_grad()

        batch_indexes = order[start_index:start_index + batch_size]

        X_batch = X_train[batch_indexes]
        y_batch = y_train[batch_indexes]

        preds = lenet5.forward(X_batch)
        loss_value = loss(preds, y_batch)
        loss_value.backward()

        optimizer.step()

    # test_preds = lenet5.forward(X_test)
    # test_loss_history.append(loss(test_preds, y_test).data.cpu())

    # accuracy = (test_preds.argmax(dim=1) == y_test).float().mean()
    # print(accuracy)

print('Тестируем на данных из тестового набора MNIST')
print(50 * '*')
print(X_test[4])
print(lenet5.forward(X_test[4].unsqueeze(1).float()))
print(y_test[4])
plot_image(X_test[4])

print(50 * '*')
print('Тестируем на собственных картинках')
print(50 * '0')
k = Image.open("0.png")
img = mlp_digits_predict("0.png")
# print('img_arr=', img)
img = torch.from_numpy(img).long()
print(lenet5.forward(img.float()).argmax(dim=1))
plot_image(img)

print(50 * '1')
k = Image.open("1.png")
img = mlp_digits_predict("1.png")
# print('img_arr=', img)
img = torch.from_numpy(img).long()
print(lenet5.forward(img.float()).argmax(dim=1))
plot_image(img)

print(50 * '2')
img = mlp_digits_predict("2.png")
k = Image.open("2.png")
# print('img_arr=', img)
img = torch.from_numpy(img).long()
print(lenet5.forward(img.float()).argmax(dim=1))
plot_image(img)

print(50 * '3')
img = mlp_digits_predict("3.png")
k = Image.open("3.png")
# print('img_arr=', img)
img = torch.from_numpy(img).long()
print(lenet5.forward(img.float()).argmax(dim=1))
plot_image(img)

print(50 * '4')
img = mlp_digits_predict("4.png")
k = Image.open("4.png")
# print('img_arr=', img)
img = torch.from_numpy(img).long()
print(lenet5.forward(img.float()).argmax(dim=1))
plot_image(img)

print(50 * '5')
img = mlp_digits_predict("5.png")
k = Image.open("5.png")
# print('img_arr=', img)
img = torch.from_numpy(img).long()
print(lenet5.forward(img.float()).argmax(dim=1))
plot_image(img)

print(50 * '6')
img = mlp_digits_predict("6.png")
k = Image.open("6.png")
# print('img_arr=', img)
img = torch.from_numpy(img).long()
print(lenet5.forward(img.float()).argmax(dim=1))
plot_image(img)

print(50 * '7_1')
img = mlp_digits_predict("7_1.png")
k = Image.open("7_1.png")
# print('img_arr=', img)
img = torch.from_numpy(img).long()
print(lenet5.forward(img.float()).argmax(dim=1))
plot_image(img)

print(50 * '7_2')
img = mlp_digits_predict("7_2.png")
k = Image.open("7_2.png")
# print('img_arr=', img)
img = torch.from_numpy(img).long()
print(lenet5.forward(img.float()).argmax(dim=1))
plot_image(img)

print(50 * '8')
img = mlp_digits_predict("8.png")
k = Image.open("8.png")
# print('img_arr=', img)
img = torch.from_numpy(img).long()
print(lenet5.forward(img.float()).argmax(dim=1))
plot_image(img)

print(50 * '9')
img = mlp_digits_predict("9.png")
k = Image.open("9.png")
# print('img_arr=', img)
img = torch.from_numpy(img).long()
print(lenet5.forward(img.float()).argmax(dim=1))
plot_image(img)
