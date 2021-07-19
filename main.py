import torch
import torch.optim as optim
import torchvision
from torchvision import transforms

import Network
import EWC


#### DATASETS ####

mnist = torchvision.datasets.MNIST(root='./data',
                                   train=True, download=True,
                                   transform=transforms.Compose([transforms.ToTensor()]))

mnist_test = torchvision.datasets.MNIST(root='./data',
                                   train=False, download=True,
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
mnist_test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=100, shuffle=True)

combination = []
combination.append(mnist)
combination.append(usps)
combination.append(svhn)

pre_combination = []
pre_combination.append(mnist)
pre_combination.append(usps)


joint = torch.utils.data.ConcatDataset(combination)
joint_loader = torch.utils.data.DataLoader(joint, batch_size=100, shuffle=True)

pre_joint = torch.utils.data.ConcatDataset(pre_combination)
prejoint_loader = torch.utils.data.DataLoader(pre_joint, batch_size=100, shuffle=True)

##--------------
lr = 1e-3
lam = 1000
##--------------


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("--> Sto usando ", device)
    print(" ")


    print("---> Joint Training <---")
    network_jt = Network.Network()
    Network.training(network_jt, joint, joint_loader, lr=0.01, num_epochs=10)
    print("\t\t Testing on MNIST: ", Network.testing(network_jt, mnist, mnist_loader))
    print("\t\t Testing on USPS: ", Network.testing(network_jt, usps, usps_loader))
    print("\t\t Testing on SVHN: ", Network.testing(network_jt, svhn, svhn_loader))
    print(" ")


    ## CREATE NETWORK AND TRAIN ON FIRST TASK ###
    print("--> Training on MNIST <--")
    network = Network.Network()
    Network.training(network, mnist, mnist_loader, lr=0.01, num_epochs=10)
    torch.save(network.state_dict(), 'PATHS/mnist')
    print("\t\t Testing on MNIST: ", Network.testing(network, mnist, mnist_loader))
    print("\t\t Testing on USPS: ", Network.testing(network, usps, usps_loader))
    print("\t\t Testing on SVHN: ", Network.testing(network, svhn, svhn_loader))


    #### EWC train on second task ###
    network.load_state_dict(torch.load('PATHS/mnist'))
    opt1 = optim.Adam(network.parameters(), lr=lr)
    ewc1 = EWC.EWC_obj(network)
    test1 = [[mnist, mnist_loader]]

    EWC.ewc_train(network, opt1, usps, usps_loader, ewc1, lam=lam, num_epochs=10, test=test1)
    print("\t\t Testing on MNIST: ", Network.testing(network, mnist, mnist_loader))
    print("\t\t Testing on USPS: ", Network.testing(network, usps, usps_loader))
    #print("\t\t Testing on SVHN: ", Network.testing(network, svhn, svhn_loader))

    ## EWC train on third task ###
    print(" "
          "----> EWC on SVHN with lam = ", lam, " <-------"
          " ")

    opt2 = optim.Adam(network.parameters(), lr=lr)
    ewc2 = EWC.EWC_obj(network)
    test2 = [[mnist, mnist_loader], [usps, usps_loader]]

    EWC.ewc_train(network, opt2, svhn, svhn_loader, ewc2, lam=lam, num_epochs=10, test=test2)
    print("\t\t Testing on MNIST: ", Network.testing(network, mnist, mnist_loader))
    print("\t\t Testing on USPS: ", Network.testing(network, usps, usps_loader))
    print("\t\t Testing on SVHN: ", Network.testing(network, svhn, svhn_loader))

    print(""
          "------------------------------------"
          "")
    #
    #
    #### L2 train on first task #####
    network.load_state_dict(torch.load('PATHS/mnist'))
    opt3 = optim.Adam(network.parameters(), lr=lr)
    ewc3 = EWC.EWC_obj(network)

    print(" "
          "----> L2 on USPS <-------"
          " ")

    EWC.ewc_train(network, opt3, usps, usps_loader, ewc3, lam=lam, num_epochs=10, type="L2", test=test1)
    print("\t\t Testing on MNIST: ", Network.testing(network, mnist, mnist_loader))
    print("\t\t Testing on USPS: ", Network.testing(network, usps, usps_loader))
    print("\t\t Testing on SVHN: ", Network.testing(network, svhn, svhn_loader))

    opt4 = optim.Adam(network.parameters(), lr=lr)
    ewc4 = EWC.EWC_obj(network)

    #### L2 train on third task ###
    print(" "
          "----> L2 on SVHN <-------"
          " ")
    EWC.ewc_train(network, opt4, svhn, svhn_loader, ewc4, lam=lam, num_epochs=10, type="L2", test=test2)
    print("\t\t Testing on MNIST: ", Network.testing(network, mnist, mnist_loader))
    print("\t\t Testing on USPS: ", Network.testing(network, usps, usps_loader))
    print("\t\t Testing on SVHN: ", Network.testing(network, svhn, svhn_loader))
    #
    #

    ### Fine Tuning on second task ###
    network = Network.Network()
    network.load_state_dict(torch.load('PATHS/mnist'))

    print(" "
          "----> fine tuning on USPS <-------"
          " ")

    Network.training(network, usps, usps_loader, lr=lr, num_epochs=10, test=test1)
    print("\t\t Testing on MNIST: ", Network.testing(network, mnist, mnist_loader))
    print("\t\t Testing on USPS: ", Network.testing(network, usps, usps_loader))
    print("\t\t Testing on SVHN: ", Network.testing(network, svhn, svhn_loader))

    ###Fine tuning on third task ###
    print(" "
          "----> fine tuning on SVHN <-------"
          " ")

    Network.training(network, svhn, svhn_loader, lr=lr, num_epochs=10, test=test2)
    print("\t\t Testing on MNIST: ", Network.testing(network, mnist, mnist_loader))
    print("\t\t Testing on USPS: ", Network.testing(network, usps, usps_loader))
    print("\t\t Testing on SVHN: ", Network.testing(network, svhn, svhn_loader))
