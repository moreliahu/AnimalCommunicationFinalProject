import numpy as np
import argparse
import torch
from torch import nn
from torch.optim import Adam
import matplotlib.pyplot as plt

from models import GAT
from utils import load_zebra_finch_data  # ✅ הוספנו את הפונקציה הזו

#################################
### TRAIN AND TEST FUNCTIONS  ###
#################################

def train_iter(epoch, model, optimizer, criterion, input, target, mask_train, mask_val, print_every=10):
    model.train()
    optimizer.zero_grad()

    output = model(*input)
    loss = criterion(output[mask_train], target[mask_train])

    loss.backward()
    optimizer.step()

    loss_train, acc_train = test(model, criterion, input, target, mask_train)
    loss_val, acc_val = test(model, criterion, input, target, mask_val)

    if epoch % print_every == 0:
        print(f'Epoch: {epoch:04d} loss_train: {loss_train:.4f} acc_train: {acc_train:.4f} loss_val: {loss_val:.4f} acc_val: {acc_val:.4f}')

    return loss_train, acc_train, loss_val, acc_val

def test(model, criterion, input, target, mask):
    model.eval()
    with torch.no_grad():
        output = model(*input)
        output, target = output[mask], target[mask]

        loss = criterion(output, target)
        acc = (output.argmax(dim=1) == target).float().mean()
    return loss.item(), acc.item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph Attention Network for Animal Calls')
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train (default: 300)')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate (default: 0.005)')
    parser.add_argument('--l2', type=float, default=5e-4, help='weight decay (default: 5e-4)')
    parser.add_argument('--dropout-p', type=float, default=0.6, help='dropout probability (default: 0.6)')
    parser.add_argument('--hidden-dim', type=int, default=64, help='dimension of hidden layer (default: 64)')
    parser.add_argument('--num-heads', type=int, default=8, help='number of attention heads (default: 8)')
    parser.add_argument('--concat-heads', action='store_true', default=False, help='concatenate attention heads')
    parser.add_argument('--val-every', type=int, default=20, help='print every n epochs')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disable CUDA')
    parser.add_argument('--seed', type=int, default=13, help='random seed')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    print(f'Using device: {device}')

    # === Load precomputed matrices === ✅ כאן משתמשים בפונקציה שלך
    features, labels, adj_mat = load_zebra_finch_data(device=device)

    num_nodes = features.shape[0]
    num_classes = len(np.unique(labels.cpu().numpy()))

    idx = torch.randperm(num_nodes).to(device)
    idx_test = idx[:int(0.2*num_nodes)]
    idx_val = idx[int(0.2*num_nodes):int(0.4*num_nodes)]
    idx_train = idx[int(0.4*num_nodes):]

    # === Create model ===
    gat_net = GAT(
        in_features=features.shape[1],
        n_hidden=args.hidden_dim,
        n_heads=args.num_heads,
        num_classes=num_classes,
        concat=args.concat_heads,
        dropout=args.dropout_p,
        leaky_relu_slope=0.2
    ).to(device)

    optimizer = Adam(gat_net.parameters(), lr=args.lr, weight_decay=args.l2)
    criterion = nn.NLLLoss()

    train_acc_list, val_acc_list, train_loss_list, val_loss_list = [], [], [], []

    for epoch in range(1, args.epochs + 1):
        loss_train, acc_train, loss_val, acc_val = train_iter(
            epoch, gat_net, optimizer, criterion, (features, adj_mat),
            labels, idx_train, idx_val, args.val_every
        )
        train_acc_list.append(acc_train)
        val_acc_list.append(acc_val)
        train_loss_list.append(loss_train)
        val_loss_list.append(loss_val)

    # === Final test ===
    loss_test, acc_test = test(gat_net, criterion, (features, adj_mat), labels, idx_test)
    print(f'Test results: loss {loss_test:.4f} accuracy {acc_test:.4f}')

    # === Save weights ===
    torch.save(gat_net.state_dict(), 'gat_trained_weights.pth')
    print("Model weights saved to 'gat_trained_weights.pth'")

    # === Plot accuracy ===
    plt.figure(figsize=(10,5))
    plt.plot(train_acc_list, label='Train Accuracy')
    plt.plot(val_acc_list, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy per Epoch')
    plt.show()

    # === Plot loss ===
    plt.figure(figsize=(10,5))
    plt.plot(train_loss_list, label='Train Loss')
    plt.plot(val_loss_list, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss per Epoch')
    plt.show()
