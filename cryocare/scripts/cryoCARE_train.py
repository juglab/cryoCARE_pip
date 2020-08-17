import argparse
import json
import numpy as np
from cryocare.internals.CryoCARE import CryoCARE
from csbdeep.models import Config
import pickle
from os.path import join

def main():
    parser = argparse.ArgumentParser(description='Load training config.')
    parser.add_argument('--conf')

    args = parser.parse_args()
    with open(args.conf, 'r') as f:
        config = json.load(f)

    train_data = np.load(config['train_data'])
    X = train_data['X']
    Y = train_data['Y']
    X_val = train_data['X_val']
    Y_val = train_data['Y_val']

    net_conf = Config(
        axes='ZYX',
        train_loss='mse',
        train_epochs=config['epochs'],
        train_steps_per_epoch=min(int(X.shape[0]/config['batch_size']), 2),
        train_batch_size=config['batch_size'],
        unet_kern_size=config['unet_kern_size'],
        unet_n_depth=config['unet_n_depth'],
        unet_n_first=config['unet_n_first'],
        train_tensorboard=False,
        train_learning_rate=config['learning_rate']
    )

    model = CryoCARE(net_conf, config['model_name'], basedir=config['path'])

    history = model.train(X, Y, (X_val, Y_val))

    with open(join(config['path'], config['model_name'], 'history.dat'), 'wb+') as f:
        pickle.dump(history, f)

if __name__ == "__main__":
    main()