import torch


def test(model, test_dataloader):
    model.eval()

    preds = []
    trues = []

    with torch.no_grad():
        for x_test, y_test in test_dataloader:
            outputs = model(x_test)

            preds.append(outputs.detach().cpu())
            trues.append(y_test.detach().cpu())

    preds = torch.cat(preds, dim=0)
    trues = torch.cat(trues, dim=0)

    return preds, trues