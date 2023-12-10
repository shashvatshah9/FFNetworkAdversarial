import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import datasets, transforms
import numpy as np

from FFNetwork import FFNetwork
from FFEncoding import FFEncoding

overlay_y_on_x = FFEncoding.overlay

import time


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


def plot_errors(training_errors, testing_errors, EPOCHS):
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
    plt.savefig("error_plot_30_epochs.png")  # Save the plot as a PNG file
    plt.show()


def training_loop(model, iterator, device, encoding="overlay"):
    model.train()
    if batched_per_layer:
        model(iterator, device)
    else:
        model.to(device)
        for _, x_data in tqdm(enumerate(iterator)):
            model(x_data, device)


def test_loop(model, data_loader, device):
    model.eval()
    batch_error = 0
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        batch_error += calc_error(model, x, y, device)

    avg_error = batch_error / len(data_loader)
    print(f"error: {avg_error}")
    return avg_error


def eval_loop(model, input, device, batched_per_layer=False, encoding="overlay"):
    """
    eval_loop(
        model -> nn.Module model
        input -> tensor input for eval
        device -> torch.device
        bached_per_layer -> False by default, if true then load each layer sequentially on device and store the output
        encoding -> overlay by default
    )
    """

    if batched_per_layer == True:
        if encoding == "overlay":
            goodness_per_label = []
            for label in range(10):
                h = overlay_y_on_x(input, label)
                goodness = []
                for module in model.children():
                    module.to(device)
                    h = module(h)
                    goodness += [h.pow(2).mean(1)]
                    module.to("cpu")
                goodness_per_label += [sum(goodness).unsqueeze(1)]
            goodness_per_label = torch.cat(goodness_per_label, 1)
            return goodness_per_label.argmax(1)
    else:
        model.to(device)
        if encoding == "overlay":
            goodness_per_label = []
            for label in range(10):
                h = overlay_y_on_x(input, label)
                goodness = []
                for module in model.children():
                    h = module(h)
                    goodness += [h.pow(2).mean(1)]
                goodness_per_label += [sum(goodness).unsqueeze(1)]
            goodness_per_label = torch.cat(goodness_per_label, 1)
            return goodness_per_label.argmax(1)

def eval_loop_attack(model, input, device, batched_per_layer=False, encoding="overlay"):
    """
    eval_loop_attack(
        model -> nn.Module model
        input -> tensor input for eval
        device -> torch.device
        bached_per_layer -> False by default, if true then load each layer sequentially on device and store the output
        encoding -> overlay by default
    )
    """

    if batched_per_layer == True:
        if encoding == "overlay":
            goodness_per_label = []
            for label in range(10):
                h = overlay_y_on_x(input, label)
                goodness = []
                for module in model.children():
                    module.to(device)
                    h = module(h)
                    goodness += [h.pow(2).mean(1)]
                    module.to("cpu")
                goodness_per_label += [sum(goodness).unsqueeze(1)]
            goodness_per_label = torch.cat(goodness_per_label, 1)
            return goodness_per_label
    else:
        model.to(device)
        if encoding == "overlay":
            goodness_per_label = []
            for label in range(10):
                h = overlay_y_on_x(input, label)
                goodness = []
                for module in model.children():
                    h = module(h)
                    goodness += [h.pow(2).mean(1)]
                goodness_per_label += [sum(goodness).unsqueeze(1)]
            goodness_per_label = torch.cat(goodness_per_label, 1)
            return goodness_per_label

def calc_error(model, x, y, device) -> float:
    model.eval()
    return (
        1
        - eval_loop(model, x, device, batched_per_layer=batched_per_layer)
        .eq(y)
        .float()
        .mean()
        .item()
    )

def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon*sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def denorm(batch, mean=[0.1307], std=[0.3081]):
    """
    Convert a batch of tensors to their original scale.
    Args:
        batch (torch.Tensor): Batch of normalized tensors.
        mean (torch.Tensor or list): Mean used for normalization.
        std (torch.Tensor or list): Standard deviation used for normalization.

    Returns:
        torch.Tensor: batch of tensors without normalization applied to them.
    """

    if isinstance(mean, list):
        mean = torch.tensor(mean).to(device)
    if isinstance(std, list):
        std = torch.tensor(std).to(device)

    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)


def test_attack( model, device, test_loader, epsilon ):

    correct = 0
    correct_benign = 0
    adv_examples = []
    total = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        data.requires_grad = True

        output = eval_loop_attack(model, data, device, batched_per_layer=batched_per_layer)
        output = output.float()
        correct_benign += (output.argmax(1) == target).sum().item()
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)

        model.zero_grad()

        loss.backward()

        data_grad = data.grad.data

        data_denorm = denorm(data)

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data_denorm, epsilon, data_grad)

        # Reapply normalization
        perturbed_data_normalized = transforms.Normalize((0.1307,), (0.3081,))(perturbed_data)

        output = eval_loop_attack(model, perturbed_data_normalized.squeeze(0).squeeze(0), device, batched_per_layer=batched_per_layer)
        total += target.size(0)
        correct += (output.argmax(1) == target).sum().item()
        adv_ex = perturbed_data_normalized.squeeze(0).squeeze(0)[0,:].detach().cpu().numpy()
        if len(adv_examples) < 5:
            adv_examples.append( (target[0].item(), output.argmax(1)[0].item(), adv_ex) )
    
    final_acc_benign = correct_benign/total
    final_acc_adversarial = correct/total
    print(f"Epsilon: {epsilon}\tTest Accuracy Benign = {correct_benign} / {total} = {final_acc_benign}")
    print(f"Epsilon: {epsilon}\tTest Accuracy Adversarial = {correct} / {total} = {final_acc_adversarial}")
    return final_acc_adversarial, adv_examples

def plot_epsilon_accuracy_graph(accuracies, epsilons):
    plt.figure(figsize=(5,5))
    plt.plot(epsilons, accuracies, "*-")
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.xticks(np.arange(0, .35, step=0.05))
    plt.title("Accuracy vs Epsilon")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.savefig("Accuracy.png")
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

if __name__ == "__main__":
    # Define parameters
    EPOCHS = 10
    BATCH_SIZE = 5000
    TRAIN_BATCH_SIZE = BATCH_SIZE
    TEST_BATCH_SIZE = BATCH_SIZE
    batched_per_layer = False
    encoding = "overlay"
    torch.manual_seed(1234)

    # Build train and test loaders
    train_loader, test_loader = get_data_loaders(
        train_batch_size=TRAIN_BATCH_SIZE, test_batch_size=TEST_BATCH_SIZE
    )

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build network
    net = FFNetwork([784, 1000, 1000])

    training_errors = []
    testing_errors = []
    # Iterator in place of DataLoader
    data_iter = []

    # Encode true and false labels on images to create positive and negative data
    print("Encoding positive and negative data with correct and incorrect labels")
    for x, y in tqdm(train_loader):
        x_pos, x_neg = None, None
        if encoding == "overlay":
            x_pos = overlay_y_on_x(x, y)
            rand_mask = torch.randint(0, 9, y.size())
            y_rnd = (y + rand_mask + 1) % 10
            x_neg = overlay_y_on_x(x, y_rnd)

        data_iter.append((x_pos, x_neg))

    for epoch in range(EPOCHS):
        print(f"==== EPOCH: {epoch} ====")
        start = time.time()
        print("Training.....")
        training_loop(net, data_iter, device)
        print("eval train data")
        training_error = test_loop(net, train_loader, device)
        training_errors.append(training_error)
        print("eval test data")
        testing_error = test_loop(net, test_loader, device)
        testing_errors.append(testing_error)
        end = time.time()
        elapsed = end - start
        print(f"Completed epoch {epoch} in {elapsed} seconds")
    # Plot errors

    plot_errors(training_errors, testing_errors, EPOCHS)

    # FGSM attck
    accuracies = []
    examples = []
    epsilons = [0, .05, .1, .15, .2, .25, .3]

    for eps in epsilons:
        print("Epsilon: ", eps)
        acc, ex = test_attack(net, device, test_loader, eps)
        accuracies.append(acc)
        examples.append(ex)
    
    plot_epsilon_accuracy_graph(accuracies, epsilons)
    plot_attack_examples(examples, epsilons)
