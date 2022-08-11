#! python
import argparse
import json
from cryocare.internals.CryoCARE import CryoCARE
from csbdeep.models import Config
import pickle
from os.path import join
from cryocare.internals.CryoCAREDataModule import CryoCARE_DataModule

def main():
    parser = argparse.ArgumentParser(description='Load training config.')
    parser.add_argument('--conf')

    args = parser.parse_args()
    with open(args.conf, 'r') as f:
        config = json.load(f)

    dm = CryoCARE_DataModule()
    dm.load(config['train_data'])

    net_conf = Config(
        axes='ZYXC',
        train_loss='mse',
        train_epochs=config['epochs'],
        train_steps_per_epoch=config['steps_per_epoch'],
        train_batch_size=config['batch_size'],
        unet_kern_size=config['unet_kern_size'],
        unet_n_depth=config['unet_n_depth'],
        unet_n_first=config['unet_n_first'],
        train_tensorboard=False,
        train_learning_rate=config['learning_rate']
    )

    model = CryoCARE(net_conf, config['model_name'], basedir=config['path'])

    history = model.train(dm.get_train_dataset(), dm.get_val_dataset())
    mean, std = dm.train_dataset.mean, dm.train_dataset.std

    with open(join(config['path'], config['model_name'], 'history.dat'), 'wb+') as f:
        pickle.dump(history.history, f)


    # Write norm to disk
    norm = {
        "mean": float(mean),
        "std": float(std)
    }
    with open(join(config['path'], config['model_name'], 'norm.json'), 'w') as fp:
        json.dump(norm, fp)

    import tarfile
    import os
    with tarfile.open(join(config['path'], f"{config['model_name']}.tar.gz"), "w:gz") as tar:
        tar.add(join(config['path'], config['model_name']), arcname=os.path.basename(join(config['path'], config['model_name'])))



if __name__ == "__main__":
    main()