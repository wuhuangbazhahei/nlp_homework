from pathlib import Path
from torch.utils.data import Dataset
from simplenmt.helpers import read_list_from_file

class TranslationDataset(Dataset):
    def __init__(
        self,
        path,
        src_lang,
        trg_lang=None,
        split="train",
        has_trg=True,
        sequence_encoder=None
    ):
        self.path = path
        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.has_trg = has_trg
        self.split = split
    
        self.sequence_encoder = {self.src_lang: None, self.trg_lang: None} if sequence_encoder is None else sequence_encoder
        self.data = self.load_data(self.path)

    def load_data(self, path):
    
        path = Path(path)
        src_file = path.with_suffix(f"{path.suffix}.{self.src_lang}")
        assert src_file.is_file(), f"{src_file} not found. Abort."
        
        src_list = read_list_from_file(src_file)# 返回一个list
        data = {self.src_lang: src_list}

        if self.has_trg:
            trg_file = path.with_suffix(f"{path.suffix}.{self.trg_lang}")
            assert trg_file.is_file(), f"{trg_file} not found. Abort."

            trg_list = read_list_from_file(trg_file)
            data[self.trg_lang] = trg_list
            assert len(src_list) == len(trg_list)
        return data

    def get_item(self, index, lang):
        line = self.data[lang][index]
        item = line.split(' ')
        return item
    
    def __getitem__(self, index):
        src, trg = None, None
        src = self.get_item(index=index, lang=self.src_lang)
        if self.has_trg:
            trg = self.get_item(index=index, lang=self.trg_lang)
        
        return src, trg
    
    def get_list(self, lang):
        """获取所需语言列表
        """
        item_list = []
        for idx in range(self.__len__()):
            item = self.data[lang][idx]
            item_list.append(item)
        return item_list
    
    def __len__(self):
        return len(self.data[self.src_lang])
    
    def __repr__(self):
        return (f"{self.__class__.__name__}(split={self.split}, len={self.__len__()}, "
                f"src_lang={self.src_lang}, trg_lang={self.trg_lang}, "
                f"has_trg={self.has_trg})")