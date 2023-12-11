import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from torchvision import transforms
import numpy as np
import copy
from utils import plot_errors, get_data_loaders, plot_attack_examples, plot_epsilon_accuracy_graph
from FFNetwork import FFNetwork
from FFEncoding import FFEncoding

overlay_y_on_x = FFEncoding.overlay

import time


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
            return goodness_per_label.argmax(1), goodness_per_label
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
            return goodness_per_label.argmax(1), goodness_per_label


def calc_error(model, x, y, device) -> float:
    model.eval()
    return (
        1
        - eval_loop(model, x, device, batched_per_layer=batched_per_layer)[0]
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

        _, output = eval_loop(model, data, device, batched_per_layer=batched_per_layer)
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

        _, output = eval_loop(model, perturbed_data_normalized.squeeze(0).squeeze(0), device, batched_per_layer=batched_per_layer)
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

def prepare_data(x, y):
    x_pos, x_neg = None, None
    if encoding == "overlay":
        x_pos = overlay_y_on_x(x, y)
        rand_mask = torch.randint(0, 9, y.size())
        y_rnd = (y + rand_mask + 1) % 10
        x_neg = overlay_y_on_x(x, y_rnd)
    return (x_pos, x_neg)

def test_attack_and_data_preparation(model, device, data_loader, epsilon):
    correct = 0
    correct_benign = 0
    total = 0
    data_iter = []
    for data, target in data_loader:
        model.eval()
        model_2 = copy.deepcopy(model)
        
        encoded_data = prepare_data(data, target)
        data_iter.append(encoded_data)

        data, target = data.to(device), target.to(device)
        
        data.requires_grad = True

        _, output = eval_loop(model_2, data, device, batched_per_layer=batched_per_layer)
        output = output.float()
        
        correct_benign += (output.argmax(1) == target).sum().item()
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)

        model_2.zero_grad()

        loss.backward()

        data_grad = data.grad.data

        data_denorm = denorm(data)

        # Call FGSM Attack
        perturbed_data = fgsm_attack(data_denorm, epsilon, data_grad)

        # Reapply normalization
        perturbed_data_normalized = transforms.Normalize((0.1307,), (0.3081,))(perturbed_data)
        encoded_data_perturbed = prepare_data(perturbed_data_normalized.squeeze(0).squeeze(0).cpu(), target.cpu())
        data_iter.append(encoded_data_perturbed)

        _, output = eval_loop_attack(model_2, perturbed_data_normalized.squeeze(0).squeeze(0), device, batched_per_layer=batched_per_layer)
        total += target.size(0)
        correct += (output.argmax(1) == target).sum().item()

    final_acc_benign = correct_benign/total
    print(f"Epsilon: {epsilon}\Train Accuracy = {correct_benign} / {total} = {final_acc_benign}")
    return final_acc_benign, data_iter


if __name__ == "__main__":
    # Define parameters
    EPOCHS = 5
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

    plot_errors(training_errors, testing_errors, EPOCHS, "error_plot_before_adversarial_training")

    # FGSM attck
    accuracies = []
    examples = []
    epsilons = [0, .05, .1, .15, .2, .25, .3]

    for eps in epsilons:
        print("Epsilon: ", eps)
        acc, ex = test_attack(net, device, test_loader, eps)
        accuracies.append(acc)
        examples.append(ex)
    
    plot_epsilon_accuracy_graph(accuracies, epsilons, "attack_epsilon_vs_accuracy_before_adversarial")
    plot_attack_examples(examples, epsilons)

    # Resetting model
    net = FFNetwork([784, 1000, 1000])
    batched_per_layer = False
    epsilon = 0.5
    EPOCHS = 15
    BATCH_SIZE = 5000
    TRAIN_BATCH_SIZE = BATCH_SIZE
    TEST_BATCH_SIZE = BATCH_SIZE
    batched_per_layer = False
    encoding = "overlay"
    torch.manual_seed(1234)

    training_errors = []
    testing_errors = []
    # Iterator in place of DataLoader
    data_iter = []

    # Build train and test loaders
    train_loader, test_loader = get_data_loaders(
        train_batch_size=TRAIN_BATCH_SIZE, test_batch_size=TEST_BATCH_SIZE
    )

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

    # Adversarial training

    epsilon = 0.3
    for epoch in range(EPOCHS):
        print(f"==== EPOCH: {epoch} ====")
        start = time.time()
        print("Training.....")
        training_loop(net, data_iter, device)
        print("eval train data")
        training_error = test_loop(net, train_loader, device)
        training_errors.append(training_error)
        training_error, data_iter = test_attack_and_data_preparation(net, device, train_loader, epsilon)
        # training_errors.append(training_error)
        print("eval test data")
        testing_error = test_loop(net, test_loader, device)
        testing_errors.append(testing_error)
        end = time.time()
        elapsed = end - start
        print(f"Completed epoch {epoch} in {elapsed} seconds")
        epsilon += 0.2
    
    plot_errors(training_errors, testing_errors, EPOCHS, "error_plot_after_adversarial_training.png")

    # FGSM attck
    accuracies = []
    examples = []
    epsilons = [0, .05, .1, .15, .2, .25, .3]

    for eps in epsilons:
        print("Epsilon: ", eps)
        acc, ex = test_attack(net, device, test_loader, eps)
        accuracies.append(acc)
        examples.append(ex)

    plot_epsilon_accuracy_graph(accuracies, epsilons, "attack_epsilon_vs_accuracy_after_adversarial")
    plot_attack_examples(examples, epsilons)
