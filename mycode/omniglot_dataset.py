import torch
import os
import errno
import shutil

class OmniglotDataset(torch.utils.data.Dataset):
    vinyals_baseurl = 'https://raw.githubusercontent.com/jakesnell/prototypical-networks/master/data/omniglot/splits/vinyals/'
    vinyals_split_sizes = {
        'test': vinyals_baseurl + 'test.txt',
        'train': vinyals_baseurl+'train.txt',
        'trainval':vinyals_baseurl+'trainval.txt',
        'val': vinyals_baseurl+'val.txt',
    }

    urls = [
        'https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip',
        'https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip'
    ]
    splits_folder = os.path.join('splits', 'vinyals')
    raw_folder = 'raw'
    processed_folder = 'data'

    def __init__(self, mode='train', root='..'+os.sep+'dataset', transform=None, target_transform=None, download=True):
        super(OmniglotDataset, self).__init__()
        self.root=root
        self.transform=transform
        self.target_transform=target_transform

        if download:
            self.download()
    
    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder))

    def download(self):
        from six.moves import urllib
        import zipfile

        if self._check_exists():
            return

        # Create directories if not exists
        try:
            os.makedirs(os.path.join(self.root, self.splits_folder))
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root,self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        # Download split file from https://github.com/jakesnell/prototypical-networks/
        for k,url in self.vinyals_split_sizes.items():
            print('== Downloading '+url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition(os.sep)[-1]
            file_path = os.path.join(self.root, self.splits_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())

        # Download omniglot dataset zip files from https://github.com/brendenlake/omniglot and unzip them
        for url in self.urls:
            print('== Downloading '+url)
            data=urllib.request.urlopen(url)
            filename = url.rpartition(os.sep)[-1]
            file_path=os.path.join(self.root,self.raw_folder,filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            
            orig_root=os.path.join(self.root,self.raw_folder)
            print("== Unzip from " +file_path + " to " + orig_root)
            zip_ref = zipfile.ZipFile(file_path, 'r')
            zip_ref.extractall(orig_root)
            zip_ref.close()

        # Organize files
        file_processed = os.path.join(self.root, self.processed_folder)
        for p in ['images_background', 'images_evaluation']:
            for f in os.listdir(os.path.join(orig_root, p)):
                shutil.move(os.path.join(orig_root,p,f),file_processed)
            os.rmdir(os.path.join(orig_root, p))
        print("Donwload finished.")
