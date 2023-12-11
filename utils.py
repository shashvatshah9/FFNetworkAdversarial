from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch

def get_data_loaders(train_batch_size=50, test_batch_size=50):
    transform = Compose(
        [
            ToTensor(),
            Normalize((0.1307,), (0.3081,)),
            Lambda(lambda x: torch.flatten(x)),
        ]
    )

    train_loader = DataLoader(
        MNIST("./data/", train=True, download=True, transform=transform),
        batch_size=train_batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        MNIST("./data/", train=False, download=True, transform=transform),
        batch_size=test_batch_size,
        shuffle=False,
    )

    return train_loader, test_loader


def visualize_sample(data, name="", idx=0):
    reshaped = data[idx].cpu().reshape(28, 28)
    plt.figure(figsize=(4, 4))
    plt.title(name)
    plt.imshow(reshaped, cmap="gray")
    plt.show()


def plot_errors(training_errors, testing_errors, EPOCHS, file_name):
    plt.figure(figsize=(10, 6))  # Adjust the figsize to make the plot wider
    plt.plot(
        range(1, EPOCHS + 1), training_errors, label="Training Error"
    )  # Use training_errors instead of errors
    plt.plot(
        range(1, EPOCHS + 1), testing_errors, label="Testing Error"
    )  # Use testing_errors instead of errors
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.title("Error over Epochs")
    plt.legend()
    plt.xticks(range(1, EPOCHS + 1))  # Set x-axis tick labels to 1, 2, ...
    file_name = file_name + '.png'
    plt.savefig(file_name)  # Save the plot as a PNG file
    plt.show()


def plot_epsilon_accuracy_graph(accuracies, epsilons, file_name):
    plt.figure(figsize=(5,5))
    plt.plot(epsilons, accuracies, "*-")
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(0, .35, step=0.05))
    plt.title("Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    file_name = file_name + '.png'
    plt.savefig(file_name)
    plt.show()

def plot_attack_examples(examples, epsilons):
    cnt = 0
    plt.figure(figsize=(8,10))
    for i in range(len(epsilons)):
        for j in range(len(examples[i])):
            cnt += 1
            plt.subplot(len(epsilons),len(examples[0]),cnt)
            plt.xticks([], [])
            plt.yticks([], [])
            if j == 0:
                plt.ylabel(f"Eps: {epsilons[i]}", fontsize=14)
            orig,adv,ex = examples[i][j]
            plt.title(f"{orig} -> {adv}")
            plt.imshow(ex.reshape(28, 28), cmap="gray")
    plt.tight_layout()
    plt.savefig("attack_examples.png")
    plt.show()