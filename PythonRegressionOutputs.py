def train_regression():
    model = YoungNet()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for step in range(200):
        x = torch.randn(64, 2)
        y = torch.randn(64, 2)              # target shape (N,2), float

        pred = model(x)                     # raw outputs
        loss = loss_fn(pred, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

    return model
