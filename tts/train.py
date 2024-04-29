import torch

from tts.dataset.loaders import loaders as ldrs


def train(model, loaders, optimizer, criterion, **kwargs):
    dl_train = loaders["train"]
    dl_test = loaders["test"]
    dl_val = loaders["val"]

    num_epochs = kwargs["num_epochs"]

    for epoch_no in range(num_epochs):
        epoch(model, dl_train, optimizer, criterion)

        with torch.no_grad():
            epoch(model, dl_val, optimizer, criterion, val=True)

    with torch.no_grad():
        epoch(model, dl_test, optimizer, criterion, val=True)

    
def epoch(model, dl, optimizer, criterion, val=False):
    total_loss = 0
    for batch in dl:
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.float)
                if not val: optimizer.zero_grad()
                output = model(inputs)
                loss = criterion(outputs, labels)
                if not val: 
                    loss.backward()
                    optmizer.step()
                
                total_loss += loss
    print(f"total loss {total_loss}")
    


def main():
    # TODO: instantiate model
    model = lambda x: x # for now simple identity
    optimizer = lambda x: x
    criterion = lambda x: x

    train(
        model, 
        ldrs(),
        optimizer,
        criterion,
        num_epochs=100
    )


if __name__ == '__main__':
    main()