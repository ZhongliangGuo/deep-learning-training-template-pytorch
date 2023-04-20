import json
from os.path import join
from argparse import ArgumentParser

if __name__ == '__main__':
    configs_path = r'/home/zg34/Desktop/SignatureVerification/configs'
    filename = '{}.json'.format('config')
    configs = {
        'batch_size': 64,
        'lr': 1e-4,
        'data_dir': '/home/zg34/datasets/signature/CEDAR_224',
        'num_epochs': 500,
        'save_interval': 10,
        'notif_interval': 10,
        'pretrained_model_path': '/home/zg34/Desktop/SignatureVerification/nets/swin_trans/swin_v2_b-781e5279.pth',
        "fine_tune": True,
    }
    # with open(join(configs_path, filename), 'w') as f:
    #     json.dump(configs, f)

    with open('config.json', 'r') as f:
        dic = json.load(f)
    parser = ArgumentParser()
    for k in dic:
        parser.add_argument('--{}'.format(k), type=type(dic[k]), default=dic[k])
    args = parser.parse_args()
    print(args)
