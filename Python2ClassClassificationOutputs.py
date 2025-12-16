def train_softmax_2class():
    model = YoungNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for step in range(200):
        x = torch.randn(64, 2)
        y = torch.randint(0, 2, (64,))      # target shape (N,), int class {0,1}

        logits = model(x)                   # shape (N,2)
        loss = loss_fn(logits, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

    # inference: probs = torch.softmax(logits, dim=1)
    return model

