
# import dependancies mainly pytorch
import torch.nn as nn 
import torch

# creating the positional encoding class
# the posisional encoding is mainly used to  help our transformer recognise the position of each embadding in the sequence
# we can do that by adding a fixed tensor to each position in that sequence
# the tensor that we are adding is mainly sin value to the odd numbers of the embadding of the token  and cos  to the even numbers of the embadding of the token 
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x
    
# Now we create the main class the whisper model

class whisper(nn.Module):
    
    
    def __init__(self,d_model,vocab_size, num_layers=4 , num_heads=8, dropout=0.1):
        super(whisper, self).__init__()
        self.positionalencoding = PositionalEncoding(d_model)
        # we create the encoder of the transformer
        self.encoderlayer= nn.TransformerEncoderLayer(d_model, nhead=num_heads,dim_feedforward=d_model*4,dropout=0.15 ,batch_first=True)
        # we create the encoderlayer of the transformer which  in our case is 8 layers
        self.transformer_encoder= nn.TransformerEncoder(self.encoderlayer, num_layers=num_layers)
        # we create the decoder of the transformer
        self.decoderlayer=nn.TransformerDecoderLayer(d_model=d_model,nhead=num_heads,batch_first=True)
        # we create the decoderlayer of the transformer which  in our case is 8 layers
        self.transformer_decoder= nn.TransformerDecoder(self.decoderlayer,num_layers=num_layers)
        # create ragular neural neutwork 
        self.Linear_encoder=nn.Linear(736,d_model)
        # create two 1D convoletonal neural network 
        self.conv1=nn.Conv1d(128, 70, kernel_size=3, padding=1,stride=3)
        self.conv2=nn.Conv1d(70, 50, kernel_size=3, padding=1,stride=3)
        # create a GELU acrivation function 
        self.GELU=nn.GELU()
        self.Linear= nn.Linear(d_model,vocab_size)
        # create a embadding table 
        self.embadding=nn.Embedding(vocab_size,d_model)
        # create padding mask function that will help us to  mask the padding tokens
    def create_padding_mask(self,seq):
            padding_mask = (seq != 0).float()
            return padding_mask
        # create attention mask function that will help mask out the tokekns
    def generate_attention_mask(self,sequence_length):
        attention_mask = torch.triu(torch.ones((sequence_length, sequence_length)), diagonal=1)
        attention_mask *= float('-inf')
        attention_mask[attention_mask != attention_mask] = 0
        return attention_mask


    # the forward function represente the whisper  model at the training time 
    def forward(self,src,tgt):
        tgt_=tgt
        # we will create the whisper model according to the paper whisper
        # Apply a convelotional neural network and GELU activation function to the audio data
        # then apply linear layer and add the positional_encoding
        src=self.conv1(src)
        src=self.GELU(src)
        src=self.conv2(src)
        src=self.GELU(src)
        src=self.Linear_encoder(src)
        src=self.positionalencoding(src)
        # add embadding to the text data and positional encoding
        tgt= self.embadding(tgt)
        tgt=self.positionalencoding(tgt) 
        # apply the audio to the encoder part of the whisper
        out_encoder=self.transformer_encoder(src)
        # apply the result of the encoder and the embbaded test to the decoder
        out_decoder=self.transformer_decoder(tgt,out_encoder,tgt_mask=self.generate_attention_mask(tgt.shape[1]),tgt_key_padding_mask=self.create_padding_mask(tgt_))
        # apply linear layer
        return self.Linear(out_decoder)
    # the inference function represente whisper  model at  the inference time 
    def inference(self,src):
        # initiatise tansor with  the value of 1 wich represent the start token <sos>
        tgt=torch.tensor([[1]])
        while True:
            # apply the src audio and tgt to the wisper model 
            logits = self(src,tgt)
            # take the last tensor the represent a probably destribution of the vocab
            logits=logits[:,-1,:]
            # apply the argmax the take the token with the heighst probablity according to the model 
            next_tgt= torch.argmax(logits,dim=-1)
            # add that token to the list tgt
            next_tgts=next_tgt.reshape(1,1)
            print(next_tgt)
            tgt=torch.cat((tgt,next_tgts),dim=1)
            # vreak the loop if we reatch the end of santance token in my case 2
            if next_tgt.item()==2:
                break
                    # the ljvocab is object  of the torchtext.vocab
        # you need to custimise this object to your dataset 
        # this object will help you tokenize your dataset
        print("predicted  ",ljvocab.lookup_tokens(tgt.reshape(-1).tolist()))
        # return list of the transcribed audio 
        return tgt



# creating objact of the model
# the ljvocab is object  of the torchtext.vocab
model=LJtrainerTransformer(d_model=256,vocab_size=ljvocab.__len__(),num_layers=2)
# this line of code will help us calculate the number of parametars of the model 
sum(p.numel() for p in model.parameters())      