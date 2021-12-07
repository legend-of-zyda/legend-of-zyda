import json
from pathlib import Path

import numpy as np
import torch
from lux.constants import Constants
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

DIRECTIONS = Constants.DIRECTIONS
game_state = None

replays_dir = 'autoencoder_replays/'


##################################################
## From the imitation learning notebook
def to_label(action):
    strs = action.split(' ')
    unit_id = strs[1]
    if strs[0] == 'm':
        label = {'c': None, 'n': 0, 's': 1, 'w': 2, 'e': 3}[strs[2]]
    elif strs[0] == 'bcity':
        label = 4
    else:
        label = None
    return unit_id, label


def depleted_resources(obs):
    for u in obs['updates']:
        if u.split(' ')[0] == 'r':
            return False
    return True


def parse_obs(episodes):
    team_name = "Tigga"
    obses = {}
    samples = []

    for filepath in episodes:
        with open(filepath) as f:
            json_load = json.load(f)

        ep_id = json_load['info']['EpisodeId']
        index = np.argmax([r or 0 for r in json_load['rewards']])
        if json_load['info']['TeamNames'][index] != team_name:
            continue

        for i in range(len(json_load['steps']) - 1):
            if json_load['steps'][i][index]['status'] == 'ACTIVE':
                actions = json_load['steps'][i + 1][index]['action']
                obs = json_load['steps'][i][0]['observation']

                if depleted_resources(obs):
                    break

                obs['player'] = index
                obs = dict([
                    (k, v)
                    for k, v in obs.items()
                    if k in ['step', 'updates', 'player', 'width', 'height']
                ])
                obs_id = f'{ep_id}_{i}'
                obses[obs_id] = obs

                for action in actions:
                    unit_id, label = to_label(action)
                    if label is not None:
                        samples.append((obs_id, unit_id, label))

    return obses, samples


# Input for Autocoder (same as input for NN in imitation notebook)
def make_input(obs, unit_id):
    width, height = obs['width'], obs['height']
    x_shift = (32 - width) // 2
    y_shift = (32 - height) // 2
    cities = {}

    b = np.zeros((20, 32, 32), dtype=np.float32)

    for update in obs['updates']:
        strs = update.split(' ')
        input_identifier = strs[0]

        if input_identifier == 'u':
            x = int(strs[4]) + x_shift
            y = int(strs[5]) + y_shift
            wood = int(strs[7])
            coal = int(strs[8])
            uranium = int(strs[9])
            if unit_id == strs[3]:
                # Position and Cargo
                b[:2, x, y] = (1, (wood + coal + uranium) / 100)
            else:
                # Units
                team = int(strs[2])
                cooldown = float(strs[6])
                idx = 2 + (team - obs['player']) % 2 * 3
                b[idx:idx + 3, x,
                  y] = (1, cooldown / 6, (wood + coal + uranium) / 100)
        elif input_identifier == 'ct':
            # CityTiles
            team = int(strs[1])
            city_id = strs[2]
            x = int(strs[3]) + x_shift
            y = int(strs[4]) + y_shift
            idx = 8 + (team - obs['player']) % 2 * 2
            b[idx:idx + 2, x, y] = (1, cities[city_id])
        elif input_identifier == 'r':
            # Resources
            r_type = strs[1]
            x = int(strs[2]) + x_shift
            y = int(strs[3]) + y_shift
            amt = int(float(strs[4]))
            b[{'wood': 12, 'coal': 13, 'uranium': 14}[r_type], x, y] = amt / 800
        elif input_identifier == 'rp':
            # Research Points
            team = int(strs[1])
            rp = int(strs[2])
            b[15 + (team - obs['player']) % 2, :] = min(rp, 200) / 200
        elif input_identifier == 'c':
            # Cities
            city_id = strs[2]
            fuel = float(strs[3])
            lightupkeep = float(strs[4])
            cities[city_id] = min(fuel / lightupkeep, 10) / 10

    # Day/Night Cycle
    b[17, :] = obs['step'] % 40 / 40
    # Turns
    b[18, :] = obs['step'] / 360
    # Map Size
    b[19, x_shift:32 - x_shift, y_shift:32 - y_shift] = 1

    return b


class LuxDataset(Dataset):

    def __init__(self, obses, samples):
        self.obses = obses
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obs_id, unit_id, action = self.samples[idx]
        obs = self.obses[obs_id]
        state = make_input(obs, unit_id)

        return state, action


#####################################################################
## Autoencoder based off imitation learning note & online AE examples
class LuxAE(nn.Module):

    def __init__(self):
        super(LuxAE, self).__init__()

        self.encoder = torch.nn.Sequential(torch.nn.Flatten(),
                                           torch.nn.Linear(32, 5, bias=False),
                                           torch.nn.ReLU(True))
        self.decoder = torch.nn.Sequential(torch.nn.Linear(5, 32, bias=False),
                                           torch.nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def state_autoencoder():
    episodes = [
        path for path in Path(replays_dir).glob('*.json')
        if 'output' not in path.name
    ]
    obses, samples = parse_obs(episodes)

    labels = [sample[-1] for sample in samples]
    # actions = ['north', 'south', 'west', 'east', 'bcity']
    # for value, count in zip(*np.unique(labels, return_counts=True)):
    #     print(f'{actions[value]:^5}: {count:>3}')

    train, val = train_test_split(samples,
                                  test_size=0.1,
                                  random_state=42,
                                  stratify=labels)
    batch_size = 64
    train_loader = DataLoader(LuxDataset(obses, train),
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=2)
    val_loader = DataLoader(LuxDataset(obses, val),
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=2)
    dataloaders_dict = {"train": train_loader, "val": val_loader}

    # Model Initialization
    model = LuxAE()

    # Validation using MSE Loss function
    loss_function = torch.nn.MSELoss()

    # Using an Adam Optimizer with lr = 0.1
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-1, weight_decay=1e-8)
    epochs = []
    losses = []

    num_epochs = 15
    for epoch in range(num_epochs):
        for phase in ['train']:
            dataloader = dataloaders_dict[phase]

            epoch_losses = []
            for item in dataloader:
                states = item[0].float()
                actions = item[1].long()

                flat = states.reshape(-1, states.shape[-1])  # (40960, 32)
                output = model(flat)
                loss = loss_function(output, flat)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss)

                print('Epoch {}/{} - Loss: {:.4f}'.format(
                    epoch + 1, num_epochs, loss))
            epochs.append(epoch + 1)
            losses.append(sum(epoch_losses) / len(epoch_losses))

    # # Plot training results
    # plt.plot(epochs, losses)
    # plt.xlabel("Epochs")
    # plt.ylabel("Losses")
    # plt.show()

    # Single out encoder module from trained autoencoder
    encoder = None
    for name, module in model.named_modules():
        if name == 'encoder':
            encoder = module

    torch.save(encoder, "encoder.h5")

    # # Do anything else
    # for phase in ['val']:
    #     dataloader = dataloaders_dict[phase]

    #     for item in dataloader:
    #         states = item[0].float()
    #         actions = item[1].long()

    #         flat = states.reshape(-1, states.shape[-1]) # (40960, 32)
    #         output = encoder(flat)
    #         print(output)
    #         break


if __name__ == "__main__":
    state_autoencoder()
