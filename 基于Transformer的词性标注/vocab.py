from collections import defaultdict


class Vocab:
    def __init__(self):
        self.idx_to_token = list() # 根据索引值获取相应的标记
        
        self.token_to_idx = dict()   # 使用字典实现标记到索引值的映射
        
    
    
    
    
    
    def build(self,text,min_freq=1,reserved_tokens=None):
        '''
        
        Builds the vocabulary from the given text.
        
        
         text: 包含若干句子，每个句子由若干标记构成
         
        '''
        
        # 创建一个默认值为0的字典token_freqs，使用defaultdict来实现。
        # 当访问字典中不存在的键时，defaultdict会自动创建该键并将其值设为默认值0
        token_freqs = defaultdict(int)
        
        for sentence in text:
            for token in sentence:
                token_freqs[token]+=1
                
        # unique_tokens中预留了未登录词标记<unk>，以及其他用户自定义的预留标记reserved_tokens
        # unique_tokens的作用是：处理后的唯一元素集合（例如，从一系列数据中提取的不重复的token）

        
        # Ensure reserved_tokens is a list
        if reserved_tokens is None:
            reserved_tokens = []
            
            
         # Add reserved tokens to the vocab
        for token in reserved_tokens:
            self.add_token(token)
    

        
        for token,freq in token_freqs.items():
            # 频率大于等于1， 并且不在vocab中，那就加进去
            if freq>=min_freq and token not in self.token_to_idx:
                self.add_token(token)
                
                
                
    def add_token(self,token):
        '''
        
        Adds a token to the vocabulary.
        
        
        '''
        
        if token in self.token_to_idx:
            return
        
        index = len(self.idx_to_token)
        self.idx_to_token.append(token)
        self.token_to_idx[token] = index
        
        
    
    def __len__(self):
        # 返回词表的大小【有多少个户不相同的标记】
        return len(self.idx_to_token)
    
    
    
    
    def __getitem__(self,token):
        '''
            查找输出的token对应的索引值
            不存在，则返回《unk》的索引值 0
        '''
        return self.token_to_idx.get(token,self.unk)
    
    
    
    
    def convert_tokens_to_ids(self, tokens):
        return [self.token_to_idx[token] for token in tokens] 
    
    
    
    def convert_ids_to_tokens(self,indices):
        return [self.idx_to_token[index] for index in indices]
    
    
    
    
    
