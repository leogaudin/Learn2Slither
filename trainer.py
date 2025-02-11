import torch
import torch.nn as nn
import torch.optim as optim
from settings import device

# torch.autograd.set_detect_anomaly(True)


def getBack(var_grad_fn):
    print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], 'variable')
                print('n[0]:', n[0])
                print('Tensor with grad found:', tensor)
                print(' - gradient:', tensor.grad)
                print()
            except AttributeError:
                getBack(n[0])


class Trainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float32).to(device)
        action = torch.tensor(action, dtype=torch.float32).to(device)
        reward = torch.tensor(reward, dtype=torch.float32).to(device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(device)

        if len(next_state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            next_state = torch.unsqueeze(next_state, 0)
            done = (done,)

        prediction = self.model(state)
        target = prediction.clone()

        for i in range(len(done)):
            Q_new = reward[i]
            if not done[i]:
                Q_new = reward[i] + (self.gamma *
                                     torch.max(self.model(next_state[i])))

            target[i][torch.argmax(action[i]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, prediction)
        loss.backward()
        print('Loss:', loss)
        # getBack(loss.grad_fn)
        self.optimizer.step()
