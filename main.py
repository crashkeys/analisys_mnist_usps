import torch
import torch.optim as optim
import torchvision
from torchvision import transforms
import lwf


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Using {device}")

    ### DATASETS ###

    USPS_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
    ])

    SVHN_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Grayscale(num_output_channels=1)
    ])

    usps = torchvision.datasets.USPS("./data"
                                     , train=True
                                     , download=True
                                     , transform=USPS_transform
                                     )

    mnist = torchvision.datasets.MNIST(root='./data'
                                       , train=True
                                       , download=True
                                       , transform=transforms.Compose([transforms.ToTensor()])
                                       )

    svhn = torchvision.datasets.SVHN(root='./data'
                                     , split='train'
                                     , transform=SVHN_transform
                                     , download=True)

    combination = []
    combination.append(mnist)
    combination.append(usps)
    combination.append(svhn)

    total_dataset = torch.utils.data.ConcatDataset(combination)

    ### LOADER ###

    mnist_loader = torch.utils.data.DataLoader(mnist, batch_size=100, shuffle=True)
    usps_loader = torch.utils.data.DataLoader(usps, batch_size=100, shuffle=True)
    svhn_loader = torch.utils.data.DataLoader(svhn, batch_size=100, shuffle=True)
    total_loader = torch.utils.data.DataLoader(total_dataset, batch_size=100, shuffle=True)

    ### JOINT TRAINING ###
    network_jt = lwf.Network()
    opt_jt = optim.Adam(network_jt.parameters(), lr=0.01)
    print("----> Joint Training <-------")
    lwf.training(network_jt, total_loader, opt_jt, 10)
    print("\t\t Testing on MNIST: ", lwf.testing(network_jt, mnist, mnist_loader))
    print("\t\t Testing on USPS: ", lwf.testing(network_jt, usps, usps_loader))
    print("\t\t Testing on SVHN: ", lwf.testing(network_jt, svhn, svhn_loader))
    print(""
          "#############################################################"
          "")

    ### TRAINING ON FIRST TASK (MNIST) ###
    network = lwf.Network()
    opt = optim.Adam(network.parameters(), lr=0.01)
    print("----> Testing on MNIST <----- (check forward transfer)")
    lwf.training(network, mnist_loader, opt, 10)
    print("\t\t Testing on MNIST: ", lwf.testing(network, mnist, mnist_loader))
    print("\t\t Testing on USPS: ", lwf.testing(network, usps, usps_loader))
    print("\t\t Testing on SVHN: ", lwf.testing(network, svhn, svhn_loader))
    torch.save(network.state_dict(), 'PATHS/network_mnist')
    print(""
          "#############################################################"
          "")

    ### FINE TUNING MNIST -> USPS ###
    print("----> Fine Tuning MNIST -> USPS")
    for lr in (0.01, 1e-3, 1e-4, 2e-4):
        print("-> lr = ", lr)
        network.load_state_dict(torch.load('PATHS/network_mnist'))
        opt1 = optim.Adam(network.parameters(), lr=lr)
        lwf.training(network, usps_loader, opt1, 10)
        print("\t\t Testing on MNIST: ", lwf.testing(network, mnist, mnist_loader))
        print("\t\t Testing on USPS: ", lwf.testing(network, usps, usps_loader))
        print("\t\t Testing on SVHN: ", lwf.testing(network, svhn, svhn_loader))
        print(" ")
    torch.save(network.state_dict(), 'PATHS/network_mnist+usps')
    print(""
          "#############################################################"
          "")

    ### FINE TUNING MNIST + USPS -> SVHN ###
    print("----> Fine Tuning MNIST + USPS -> SVHN")
    for lr in (0.01, 1e-3, 1e-4):
        print("-> lr = ", lr)
        network.load_state_dict(torch.load('PATHS/network_mnist+usps2'))
        opt11 = optim.Adam(network.parameters(), lr=lr)
        lwf.training(network, svhn_loader, opt11, 10)
        print("\t\t Testing on MNIST: ", lwf.testing(network, mnist, mnist_loader))
        print("\t\t Testing on USPS: ", lwf.testing(network, usps, usps_loader))
        print("\t\t Testing on SVHN: ", lwf.testing(network, svhn, svhn_loader))
        print(" ")
    print(""
          "#############################################################"
          "")

    ## go back to mnist trained to try other methods ##
    network.load_state_dict(torch.load('PATHS/network_mnist'))

    network2 = lwf.Network()
    ### LwF MNIST -> USPS ###
    print("----> Learning Without Forgetting MNIST -> USPS")
    for lr in (0.01, 1e-3, 1e-4, 2e-4):
        for lam in (0.0001, 0.001, 0.01, 10, 100, 1):
            print("-> lr = ", lr, " lam = ", lam)
            network2.load_state_dict(torch.load('PATHS/network_mnist'))
            lwf.LwF(network2, network, usps_loader, lr=lr, lam=lam, num_epochs=10)
            print("\t\t Testing on MNIST: ", lwf.testing(network2, mnist, mnist_loader))
            print("\t\t Testing on USPS: ", lwf.testing(network2, usps, usps_loader))
            print("\t\t Testing on SVHN: ", lwf.testing(network2, svhn, svhn_loader))
            print(" ")
    torch.save(network2.state_dict(), 'PATHS/network_mnist+usps_Lwf')  # salvo la versione con lr=2e-4 e lam=1
    print(""
          "#############################################################"
          "")

    network3 = lwf.Network()
    ### LwF MNIST + USPS -> SVHN ###
    print("----> Learning Without Forgetting MNIST + USPS -> SVHN")
    for lr in (0.01, 1e-3, 1e-4, 2e-4):
        for lam in (0.0001, 0.001, 0.01, 10, 1):
            print("-> lr = ", lr, " lam = ", lam)
            network3.load_state_dict(torch.load('PATHS/network_mnist+usps_Lwf2'))
            lwf.LwF(network3, network2, svhn_loader, lr=lr, lam=lam, num_epochs=10)
            print("\t\t Testing on MNIST: ", lwf.testing(network3, mnist, mnist_loader))
            print("\t\t Testing on USPS: ", lwf.testing(network3, usps, usps_loader))
            print("\t\t Testing on SVHN: ", lwf.testing(network3, svhn, svhn_loader))
            print(" ")
