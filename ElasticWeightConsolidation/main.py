import torch
import torch.optim as optim
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

import Network
import EWC


#### DATASETS ####

mnist = torchvision.datasets.MNIST(root='./data',
                                   train=True, download=True,
                                   transform=transforms.Compose([transforms.ToTensor()]))

USPS_transform = transforms.Compose([transforms.Resize((28, 28)),
                                     transforms.ToTensor(),])

usps = torchvision.datasets.USPS("./data", train=True,
                                 download=True,
                                 transform=USPS_transform)

SVHN_transform = transforms.Compose([transforms.Resize((28, 28)),
                                    transforms.ToTensor(),
                                    transforms.Grayscale(num_output_channels=1) ])

svhn = torchvision.datasets.SVHN(root='./data' ,
                                 split='train' ,
                                 transform=SVHN_transform,
                                 download=True)


svhn_loader = torch.utils.data.DataLoader(svhn, batch_size=100, shuffle=True)
mnist_loader = torch.utils.data.DataLoader(mnist, batch_size=100, shuffle=True)
usps_loader = torch.utils.data.DataLoader(usps, batch_size=100, shuffle=True)



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("--> Sto usando ", device)
    print(" ")


    ### CREATE NETWORK AND TRAIN ON FIRST TASK ###

    network = Network.Network()
    Network.training(network, mnist, mnist_loader, lr=0.01, num_epochs=10)
    torch.save(network.state_dict(), 'PATHS/mnist')
    print("\t\t Testing on MNIST: ", Network.testing(network, mnist, mnist_loader))
    print("\t\t Testing on USPS: ", Network.testing(network, usps, usps_loader))
    print("\t\t Testing on SVHN: ", Network.testing(network, svhn, svhn_loader))


    print(" "
          "----> EWC on USPS <-------"
          " ")

    #### EWC per passare alla seconda task ###
    opt1 = optim.Adam(network.parameters(), lr=1e-3)
    ewc1 = EWC.EWC_obj(network)

    EWC.ewc_train(network, opt1, usps, usps_loader, ewc1, lam=1000, num_epochs=10)
    print("\t\t Testing on MNIST: ", Network.testing(network, mnist, mnist_loader))
    print("\t\t Testing on USPS: ", Network.testing(network, usps, usps_loader))
    print("\t\t Testing on SVHN: ", Network.testing(network, svhn, svhn_loader))

    plt.show()

    ### EWC per passare alla terza task ###

    print(" "
          "----> EWC on SVHN <-------"
          " ")

    opt2 = optim.Adam(network.parameters(), lr=1e-3)
    ewc2 = EWC.EWC_obj(network)

    EWC.ewc_train(network, opt2, svhn, svhn_loader, ewc2, lam=1000, num_epochs=10)
    print("\t\t Testing on MNIST: ", Network.testing(network, mnist, mnist_loader))
    print("\t\t Testing on USPS: ", Network.testing(network, usps, usps_loader))
    print("\t\t Testing on SVHN: ", Network.testing(network, svhn, svhn_loader))


    #### e se invece avessi fatto L2? #####
    network.load_state_dict(torch.load('PATHS/mnist'))

    print(" "
          "----> L2 on USPS <-------"
          " ")
    EWC.ewc_train(network, opt2, usps, usps_loader, ewc1, lam=1000, num_epochs=10, type="L2")
    print("\t\t Testing on MNIST: ", Network.testing(network, mnist, mnist_loader))
    print("\t\t Testing on USPS: ", Network.testing(network, usps, usps_loader))
    print("\t\t Testing on SVHN: ", Network.testing(network, svhn, svhn_loader))

    print(" "
          "----> L2 on SVHN <-------"
          " ")
    EWC.ewc_train(network, opt2, svhn, svhn_loader, ewc2, lam=1000, num_epochs=10, type="L2")
    print("\t\t Testing on MNIST: ", Network.testing(network, mnist, mnist_loader))
    print("\t\t Testing on USPS: ", Network.testing(network, usps, usps_loader))
    print("\t\t Testing on SVHN: ", Network.testing(network, svhn, svhn_loader))


    ### e se invece avessi fatto Fine-Tuning? ###
    network.load_state_dict(torch.load('PATHS/mnist'))

    print(" "
          "----> fine tuning on USPS <-------"
          " ")

    Network.training(network, usps, usps_loader, 1e-3, num_epochs=10)
    print("\t\t Testing on MNIST: ", Network.testing(network, mnist, mnist_loader))
    print("\t\t Testing on USPS: ", Network.testing(network, usps, usps_loader))
    print("\t\t Testing on SVHN: ", Network.testing(network, svhn, svhn_loader))

    print(" "
          "----> fine tuning on SVHN <-------"
          " ")

    Network.training(network, svhn, svhn_loader, 1e-3, num_epochs=10)
    print("\t\t Testing on MNIST: ", Network.testing(network, mnist, mnist_loader))
    print("\t\t Testing on USPS: ", Network.testing(network, usps, usps_loader))
    print("\t\t Testing on SVHN: ", Network.testing(network, svhn, svhn_loader))