import importlib
import hashlib
import os
import requests
from tqdm import tqdm

URL_MAP = {
    'vggishish_lpaps': 'https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/specvqgan_public/vggishish16.pt',
    'vggishish_mean_std_melspec_10s_22050hz': 'https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/specvqgan_public/train_means_stds_melspec_10s_22050hz.txt',
    'melception': 'https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/specvqgan_public/melception-21-05-10T09-28-40.pt',
}

CKPT_MAP = {
    'vggishish_lpaps': 'vggishish16.pt',
    'vggishish_mean_std_melspec_10s_22050hz': 'train_means_stds_melspec_10s_22050hz.txt',
    'melception': 'melception-21-05-10T09-28-40.pt',
}

MD5_MAP = {
    'vggishish_lpaps': '197040c524a07ccacf7715d7080a80bd',
    'vggishish_mean_std_melspec_10s_22050hz': 'f449c6fd0e248936c16f6d22492bb625',
    'melception': 'a71a41041e945b457c7d3d814bbcf72d',
}


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit('.', 1)
    print(module, cls)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not 'target' in config:
        raise KeyError('Expected key `target` to instantiate.')
    return get_obj_from_str(config['target'])(**config.get('params', dict()))


def download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)


def md5_hash(path):
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()


def get_ckpt_path(name, root, check=False):
    assert name in URL_MAP
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path) or (check and not md5_hash(path) == MD5_MAP[name]):
        print("Downloading {} model from {} to {}".format(name, URL_MAP[name], path))
        download(URL_MAP[name], path)
        md5 = md5_hash(path)
        assert md5 == MD5_MAP[name], md5
    return path
