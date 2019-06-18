import argparse
import time

import torch
from torch import optim as optim
from torch.nn import functional as F
from torchvision import datasets, transforms

from CNN import Net


#code to train the model for given epoch
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        #Zero the gradient buffers of all parameters
        optimizer.zero_grad()

        #feed data to model to get output
        output = model(data)

        # computing loss using loss function given below
        loss = F.nll_loss(output, target)

        # back propagate the loss to compute gradients
        loss.backward()

        #update weights using optimizer
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

#code to evaluate the model
def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # feed data to model to get output
            output = model(data)

            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    accuracy = 100. * correct / len(test_loader.dataset)

    return accuracy


def train_and_test(args, device, model, test_loader, train_loader):
    # ======for STEP 4 in PA2, use L2 regularization for only fully connected layers.======#
    #
    # ******HINT*******: use weight_decay parameter when defining optimizer. may use an if statement to check if args.weight_decay is not zero, then use per parameter decay.
    # ***************** See https://pytorch.org/docs/stable/optim.html for per parameter options
    # remove following two lines for NotImplementedError after implementation of all models
    # if len(list(model.parameters())) == 0:
    #     return NotImplementedError
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    for epoch in range(1, args.epochs + 1):
        # in your training loop:
        start_time = time.time()

        train(args, model, device, train_loader, optimizer, epoch)

        end_time = time.time()
        print('the training took: %d(s)' % (end_time - start_time))

        accuracy = test(args, model, device, test_loader)
    if (args.save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")

    return accuracy


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--mode', type=int, default=1, metavar='N',
                        help='mode to define which model to be used.')
    parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--hidden-size', type=int, default=100, metavar='N',
                        help='hidden layer size for network (default: 100)')
    parser.add_argument('--weight-decay', type=int, default=0, metavar='N',
                        help='Weight decay, used for L2 regularization (default: 0)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=60, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    #check if we can use GPU for training.
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device('cuda' if use_cuda else 'cpu')

    #may increase number of workers to speed up the dataloading.
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # ======================================================================
    #  STEP 0: Load data from the MNIST database.
    #  This loads our training and test data from the MNIST database files available in torchvision package.
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(), #scale pixel values between 0 and 1
                           transforms.Normalize((0.1307,), (0.3081,)) #normalize using mean and standard deviation
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
            transforms.ToTensor(), #scale pixel values between 0 and 1
            transforms.Normalize((0.1307,), (0.3081,)) #normalize using mean and standard deviation
        ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    # ======================================================================
    #  STEP 1: Train a baseline model.
    #  This trains a feed forward neural network with one hidden layer.
    #  Expected accuracy >= 97.80%
    if args.mode == 1:
        model = Net(1, args).to(device)

        accuracy = train_and_test(args, device, model, test_loader, train_loader)

        # Output accuracy.
        print(20 * '*' + 'model 1' + 20 * '*')
        print('accuracy is %f' % (accuracy))
        print()

    # ======================================================================
    #  STEP 2: Use two convolutional layers.
    #  Expected accuracy >= 99.06%

    if args.mode == 2:
        model = Net(2, args).to(device)

        accuracy = train_and_test(args, device, model, test_loader, train_loader)

        # Output accuracy.
        print(20 * '*' + 'model 2' + 20 * '*')
        print('accuracy is %f' % (accuracy))
        print()

    # ======================================================================
    #  STEP 3: Replace sigmoid activation with ReLU.
    #
    #  Expected accuracy>= 99.23%

    if args.mode == 3:
        args.learning_rate = 0.03
        model = Net(3, args).to(device)

        accuracy = train_and_test(args, device, model, test_loader, train_loader)

        # Output accuracy.
        print(20 * '*' + 'model 3' + 20 * '*')
        print('accuracy is %f' % (accuracy))
        print()

    # ======================================================================
    #  STEP 4: Add one more fully connected layer.
    #
    #  Expected accuracy>= 99.37%

    if args.mode == 4:
        args.learning_rate = 0.03
        args.weight_decay = 1e-5
        model = Net(4, args).to(device)

        accuracy = train_and_test(args, device, model, test_loader, train_loader)

        # Output accuracy.
        print(20 * '*' + 'model 4' + 20 * '*')
        print('accuracy is %f' % (accuracy))
        print()

    # ======================================================================
    #  STEP 5: Add dropout to reduce overfitting.
    #
    #  Expected accuracy: 99.40%

    if args.mode == 5:
        args.learning_rate = 0.03
        args.epochs = 40
        args.hiddenSize = 1000

        model = Net(5, args).to(device)

        accuracy = train_and_test(args, device, model, test_loader, train_loader)

        # Output accuracy.
        print(20 * '*' + 'model 5' + 20 * '*')
        print('accuracy is %f' % (accuracy))
        print()


if __name__ == '__main__':
    main()
