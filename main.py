import torch
import torchvision
from torchvision import transforms
import torch.nn.functional as F
import torch.optim as optim

import LearningWithoutForgetting as lwf




if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### DATASETS & LOADERS ###

    USPS_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])

    usps = torchvision.datasets.USPS("./data"
                                     , train=True
                                     , download=True
                                     , transform=USPS_transform
                                     )

    mnist = torchvision.datasets.MNIST(
        root='./data'
        , train=True
        , download=True
        , transform=transforms.Compose([transforms.ToTensor()])
    )

    SVHN_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1)
    ])

    svhn = torchvision.datasets.SVHN(
        root='./data'
        , split='train'
        , transform=SVHN_transform
        , download=True)

    svhn_loader = torch.utils.data.DataLoader(svhn, batch_size=100, shuffle=False)

    mnist_loader = torch.utils.data.DataLoader(mnist, batch_size=100, shuffle=False)
    usps_loader = torch.utils.data.DataLoader(usps, batch_size=100, shuffle=False)




    network = lwf.Network()
    optimizer1 = optim.Adam(network.parameters(), 0.01) #per il training su mnist

    print("-----> TRAINING <--------")
    lwf.training(network, mnist_loader, optimizer1, num_epochs=5)
    print("\t\tTesting on MNIST:")
    print(f'\t\t {lwf.testing(network, mnist, mnist_loader)}')
    print("\t\tTesting on SVHN:")
    print(f'\t\t {lwf.testing(network, svhn, svhn_loader)}')
    print(""
          ""
          "")

    torch.save(network.state_dict(), 'PATHS_test/mnist6.pth')

    #fine-tuning
    # print("-----> Fine Tuning <------")
    # opt = optim.Adam(network.parameters(), 1e-3)
    # lwf.training(network, svhn_loader, opt, num_epochs=10)
    # print("\t\tTesting on MNIST:")
    # print(f'\t\t {lwf.testing(network, mnist, mnist_loader)}')
    # print("\t\tTesting on SVHN:")
    # print(f'\t\t {lwf.testing(network, svhn, svhn_loader)}')
    # print(""
    #       "")


    network2 = lwf.Network()
    network2.load_state_dict(torch.load('PATHS_test/mnist6.pth'))
    network.load_state_dict(torch.load('PATHS_test/mnist6.pth'))

    lr = 1e-3
    opt = optim.Adam(network2.parameters(), lr, weight_decay=0.0005) #prede i parametri della rete addestrata su mnist
    for lam in (100, 1):
        print("--------> L w F <------------")
        print(f'lr = {lr}, lambda = {lam}')
        network2.load_state_dict(torch.load('PATHS_test/mnist6.pth'))
        network.load_state_dict(torch.load('PATHS_test/mnist6.pth'))
        lwf.LwF(network2, network, svhn_loader, opt, lam=lam, num_epochs=5)
        print("\t\tTesting on MNIST:")
        print(f'\t\t {lwf.testing(network2, mnist, mnist_loader)}')
        print("\t\tTesting on SVHN:")
        print(f'\t\t {lwf.testing(network2, svhn, svhn_loader)}')
        print(""
              "")




