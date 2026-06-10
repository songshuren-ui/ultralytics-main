from torchvision import datasets, transforms
train_set = datasets.MNIST("data",train=True,download=True,transform=transforms.ToTensor(),)
test_set = datasets.MNIST("data",train=False,download=True,transform=transforms.ToTensor(),)