import json
from os.path import join
from argparse import ArgumentParser

if __name__ == '__main__':
    configs_path = r'/home/xxxxxx/configs'
    filename = '{}.json'.format('config')
    configs = {
        'batch_size': 64,
        'lr': 1e-4,
        'data_dir': '/home/xxxxxx',
        'num_epochs': 500,
        'save_interval': 10,
        'notif_interval': 10,
        "fine_tune": True,
    }
    with open(join(configs_path, filename), 'w') as f:
        json.dump(configs, f)

    with open('config.json', 'r') as f:
        dic = json.load(f)
    parser = ArgumentParser()
    for k in dic:
        parser.add_argument('--{}'.format(k), type=type(dic[k]), default=dic[k])
    args = parser.parse_args()
    print(args)
