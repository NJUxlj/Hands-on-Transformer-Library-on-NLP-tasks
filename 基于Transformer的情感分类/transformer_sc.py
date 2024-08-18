from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import Dataset


from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence


import vocab
from vocab import Vocab

class Transformer(nn.Module):
    def __init__(self,vocab_size,embedding_dim,hidden_dim, num_class, 
                 dim_feedforward=512, num_heads=2, num_layers=2, dropout=0.1,
                 max_len=128, activation:str="relu"):
        
        '''
            max_len：最大序列长度
            dropout:          丢弃概率
        '''
                     
        super(Transformer, self).__init__()
        self.embedding_dim = embedding_dim
        
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

        # 位置编码层
        self.position_embedding = PositionalEncoding(d_model = embedding_dim, dropout=dropout, max_len=max_len)
        
        
        # Encoder层
        encoder_layer = nn.TransformerEncoderLayer(embedding_dim, num_heads, dim_feedforward, dropout, activation)
        
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 输出层
        self.output = nn.Linear(embedding_dim, num_class)
        
       

    
    
    def forward(self, inputs, lengths):
        
        inputs = torch.transpose(inputs, 0, 1) # # seq_len x batch x vector_dim
        
        # transformer要求batch_size在第二维，第一维是序列长度
        
        hidden_states:torch.Tensor=self.embeddings(inputs)  # seq_len x batch x embedding_dim
        
        print("hidden_states.shape = \n",hidden_states.shape)
    
        
        hidden_states = self.position_embedding(hidden_states) # seq_len x batch x embedding_dim
        
        print("经过位置编码后，hidden_states.shape = ", hidden_states.shape)
        
        # 根据批次中的每个序列长度生成mask矩阵
        # # 根据序列长度创建掩码，以忽略填充的token
        attention_mask = length_to_mask(lengths) == False  # batch x seq_len (max_len)
        
        
        # 传递到Transformer编码器，使用掩码防止对填充token的关注
        '''
            src_key_padding_mask参数被设置为上一步生成的attention_mask，
            这意味着在进行注意力计算时，所有标记为True的位置将被忽略，
            从而避免了对填充token的不必要计算。
        '''
        # 经过TransformerEncoder层后，
        # hidden_states的形状仍然是 (seq_len, batch, embedding_dim)。
        # 这是因为Transformer编码器层的设计是为了保持序列长度不变，
        # 只是对序列中的每个元素进行了变换，以捕捉更复杂的特征表示。
        hidden_states = self.transformer(hidden_states, src_key_padding_mask=attention_mask) # (seq_len, batch, embedding_dim)

        print("经过transformer层后， hidden_states.shape = ",hidden_states.shape)
        # 取该批次每个样本中的第一个token的输出结果作为分类层的输入 
        hidden_states = hidden_states[0,:,:] # (batch, embedding_dim)
        
        
        output = self.output(hidden_states) # batch x num_class
        
        # softmax
        log_probs = F.log_softmax(output,dim=1) # batch x num_class
        
        
        print("after softmax = \n", log_probs)
        print("after softmax's shape = \n", log_probs.shape)
        
        
        return log_probs


import math
class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, dropout=0.1, max_len = 128):
        '''
            d_model: 词向量的维度
            max_len: 序列长度
        '''
        super(PositionalEncoding,self).__init__()
        
        # 存储位置编码
        pe = torch.zeros(max_len, d_model)
        
        # 存储每个token的位置（i）
        position = torch.arange(0,max_len, dtype=torch.float).unsqueeze(1) # 转换为形状为(max_len, 1)的张量
        
        # pos/10000^(2i/d_model)
        # 它是一个形状为(d_model//2)的张量，包含了位置编码的系数。
        div_term = torch.exp(torch.arange(0, d_model, 2).float()*(-math.log(10000.0)/d_model))
        
        # 对偶数位置编码
        pe[:, 0::2] = torch.sin(position * div_term)
        # 对奇数位置编码
        pe[:,1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0,1) # 形状为(max_len, 1, d_model) = (seq_len, 1, embedding_dim)
        
        self.register_buffer('pe', pe) # 不对位置编码层求梯度
        
        
        
    def forward(self, x):
        x # seq_len, embedding_dim, 单个句子
        x = x + self.pe[:x.size(0),:]
        
        return x
        
        
        
        
        
        
        
def length_to_mask(lengths):
    """
        将序列的长度转换成 Mask 矩阵
        >>> lengths = torch.tensor([3, 5, 4])
        >>> length_to_mask(lengths)
        
        >>> tensor([[ True,  True,  True, False, False],
                    [ True,  True,  True,  True,  True],
                    [ True,  True,  True,  True, False]])
                    
        :param lengths: [batch,]
        :return: batch * max_len
    """
    
    
    max_len = torch.max(lengths) # 获取所有序列中最长序列的长度
    
    # 把向量扩成矩阵
    matrix = torch.arange(max_len).expand(lengths.shape[0], max_len) # [batch, max_len]
    
    # 把lengths多加一维，这样可以利用广播机制
    mask = matrix < lengths.unsqueeze(1) # [batch, max_len]
    return mask

def load_sentence_polarity():
    import nltk
    nltk.download('sentence_polarity')
    from nltk.corpus import sentence_polarity
    
    vocab:Vocab= Vocab()
    vocab.build(text = sentence_polarity.sents())
    
    
    print("vocab length = ",len(vocab))

    train_data = [(vocab.convert_tokens_to_ids(sentence), 0) 
                    for sentence in sentence_polarity.sents(categories='pos')[:4000]]\
    + [(vocab.convert_tokens_to_ids(sentence), 1) 
       for sentence in sentence_polarity.sents(categories='neg')[:4000]]


    
    test_data = [(vocab.convert_tokens_to_ids(sentence),0) for sentence in sentence_polarity.sents(categories='pos')[4000:]]\
    +[(vocab.convert_tokens_to_ids(sentence),1) for sentence in sentence_polarity.sents(categories='neg')[4000:]]


    
    return train_data, test_data, vocab


class BowDataset(Dataset):
    '''
    Bow: 词袋
    '''
    def __init__(self,data):
        # data为原始的数据,如使用load_sentence_polarity函数获得的训练数据
        #和测试数据
        self.data = data
        
    
    
    def __len__(self):
        return len(self.data)


    
    def __getitem__(self,i):
        return self.data[i]


from torch.utils.data import DataLoader
# data_loader = DataLoader(
#     dataset=,
#     batch_size=4,
#     shuffle=,
#     collate_fn=collate_fn
# )


from torch.nn.utils.rnn import pad_sequence
def collate_fn(examples):   
    
    
    lengths = torch.tensor([len(ex[0]) for ex in examples])
    
    inputs = [torch.tensor(ex[0]) for ex in examples]
    

    
    labels = torch.tensor([ex[1] for ex in examples], dtype=torch.long)
    
    
    # 使每个句子的长度都一样
    inputs = pad_sequence(inputs, batch_first=True)
    
    
    return inputs, labels, lengths




def main():
    from tqdm.auto import tqdm
    
    # 超参数设置
    embedding_dim = 128
    hidden_dim = 256
    num_class = 2
    batch_size = 32
    num_epoch = 5

    
    
    

    
    
    # 加载数据
    train_data, test_data, vocab = load_sentence_polarity()
    
    train_dataset = BowDataset(train_data) # 包装一下
    
    test_dataset = BowDataset(test_data)    
    
    # 打乱， 分批， 分列返回
    train_data_loader = DataLoader(dataset=train_data,batch_size=batch_size,
                                   shuffle=True,collate_fn=collate_fn)
    
    
    
    test_data_loader = DataLoader(dataset=test_data,batch_size=batch_size,
                                  shuffle=True,collate_fn=collate_fn)

    
    
    
    
    
    
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Transformer(len(vocab),embedding_dim,hidden_dim=hidden_dim,num_class=num_class)
    model.to(device)
    
      
    
    # 训练过程
    
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    
    model.train()
    
    for epoch in range(num_epoch):
        total_loss =0
        
        for batch in tqdm(train_data_loader,desc=f"Training Epoch： {epoch}"):
            '''
                假设 batch 是一个包含三个元素的元组或列表，那么列表推导式会生成一个同样有三个元素的新列表。

                当这个列表被解包时，inputs 会接收列表中的第一个元素，labels 接收第二个元素，
                而 offsets 接收第三个元素。这是因为 Python 允许你将一个包含多个元素的容器
                （如列表或元组）直接分配给多个变量，只要两边的数量匹配即可。
            '''
            inputs, labels, lengths = batch
            
            optimizer.zero_grad()
            log_probs = model.forward(inputs,lengths)
            
            loss = nn.NLLLoss()(log_probs,labels)
            
            loss.backward()
            
            optimizer.step()
            
            total_loss+=loss.item()
        
        print(f'Loss:{total_loss:.2f}')
            
            
    
    # 测试
    acc = 0
    for batch in tqdm(test_data_loader,desc= f"Testing"):
        inputs,labels,offsets = [x.to(device) for x in batch]
        with torch.no_grad():
            log_pred = model(inputs,offsets)
            output = torch.argmax(log_pred,dim=1)
            acc+=(output == labels).sum().item()
            
    print(f"Acc: {acc/len(test_dataset)}")
    
    

if __name__ == '__main__':
    main()
