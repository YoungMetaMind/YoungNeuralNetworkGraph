def train_multilabel():
    model = YoungNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    for step in range(200):
        x = torch.randn(64, 2)
        y = torch.randint(0, 2, (64, 2)).float()  # target shape (N,2), float 0/1

        logits = model(x)                   # shape (N,2)
        loss = loss_fn(logits, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

    # inference: probs = torch.sigmoid(logits)
    return model
