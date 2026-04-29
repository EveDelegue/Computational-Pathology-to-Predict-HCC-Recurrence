import os
import yaml

class TestPaths:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)


    def test_exists(self):
        assert os.path.exists(self.config['paths']['pth_to_tumor_ckpts'])
        assert os.path.exists(self.config['paths']['pth_to_nuc_ckpts'])
        assert os.path.exists(self.config['paths']['pth_to_inflams_ckpts'])
    
    def test_same_content(self):
        list_nuc = os.listdir(self.config['paths']['pth_to_nuc_ckpts']) 
        assert len(list_nuc)>0
        list_tumor = os.listdir(self.config['paths']['pth_to_tumor_ckpts'])
        assert len(list_tumor)>0
        list_p_tumor = [os.path.basename(name).split('_')[0]+'_'+os.path.basename(name).split('_')[1] for name in list_tumor]
        list_p_nuc = [os.path.basename(name).split('_')[0]+'_'+os.path.basename(name).split('_')[1] for name in list_nuc]
        assert list_p_nuc == list_p_tumor

