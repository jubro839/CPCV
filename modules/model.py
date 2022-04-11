import torch 
from torch import nn 


class LSTM_Model(nn.Module):
    def __init__(self, params):
        '''
        params (dict): model parameters
        '''
        super(LSTM_Model, self).__init__()
        self.device = params["device"]
        self.num_layers = params["num_layers"]
        
        self.lstm_in_dim = params["num_assets"]
        self.lstm_out_dim = params["num_assets"] 
        
        self.fc_in_dim = params["past_seq"]
        self.fc_out_dim = params["future_seq"]

## LSTM 정의 
        self.lstm = nn.LSTM(self.lstm_in_dim, self.lstm_out_dim, num_layers = self.num_layers, batch_first = True)
        self.fc = nn.Conv2d(self.fc_in_dim, self.fc_out_dim, kernel_size= (1,1))        
        
    def forward(self, x):
        '''
        x: [batch size, past sequence]
        '''
        batch_size = x.size(0)
## 초기 hidden state, cell state 정의해주기 : 0으로 준다
        hidden_init = torch.zeros(self.num_layers, batch_size, self.lstm_out_dim).to(self.device)
        cell_init = torch.zeros(self.num_layers, batch_size, self.lstm_out_dim).to(self.device)
        
        output, hidden = self.lstm(x, (hidden_init, cell_init))
        prediction = output.unsqueeze(-1)
        
####### 예측길이 바꿔주고 싶을 때 실행        
        prediction = self.fc(prediction)
        prediction = prediction.squeeze()
        
        return prediction         
        