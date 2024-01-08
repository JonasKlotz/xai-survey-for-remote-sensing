import torch.nn.functional as F
from torch import optim


def surrogate_training(
    blackbox,
    surrogate,
    train_loader,
    val_loader,
    test_loader,
    device,
    epochs=10,
    lr=0.001,
    loss_fn=F.nll_loss,
    optimizer_fn=optim.Adam,
):
    """
    Train a surrogate model on the blackbox
    :param loss_fn:  loss function
    :param optimizer_fn:  optimizer function
    :param blackbox:  blackbox model
    :param surrogate:  surrogate model
    :param train_loader:  training data
    :param val_loader:  validation data
    :param test_loader:  test data
    :param device:  device to train on
    :param epochs:  number of epochs to train for
    :param lr:  learning rate
    :return:
    """
    optimizer = optimizer_fn(surrogate.parameters(), lr=lr)
    blackbox.eval()

    for epoch in range(epochs):
        surrogate.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            blackbox_target = blackbox(data).argmax(dim=1, keepdim=True)
            optimizer.zero_grad()
            output = surrogate(data)
            loss = loss_fn(output, blackbox_target)
            loss.backward()
            optimizer.step()

        surrogate.eval()
        for batch_idx, (data, target) in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            output = surrogate(data)
            blackbox_target = blackbox(data).argmax(dim=1, keepdim=True)

            loss = loss_fn(output, blackbox_target)
            preds = output.argmax(dim=1, keepdim=True)
            correct = preds.eq(target.view_as(preds)).sum().item()
            print(
                f"Epoch: {epoch} Val set: Average loss: {loss.item():.4f}, Accuracy: {correct}/{len(val_loader.dataset)} ({100. * correct / len(val_loader.dataset):.0f}%)"
            )
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = surrogate(data)
            blackbox_target = blackbox(data).argmax(dim=1, keepdim=True)

            loss = loss_fn(output, blackbox_target)
            preds = output.argmax(dim=1, keepdim=True)
            correct = preds.eq(target.view_as(preds)).sum().item()
            print(
                f"Epoch: {epoch} Test set: Average loss: {loss.item():.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)"
            )
    return surrogate
