from os import device_encoding
import torch
from torch import nn
import torch.nn.utils.rnn as rnn
import numpy as np
import copy
import torch.nn.functional as F

def getMatr4TempEnc(embDim,mode):
    matr=np.zeros((int(embDim/2),embDim))
    col=mode
    for i in range(int(embDim/2)):
        matr[i][col]=1
        col+=2
    return torch.from_numpy(matr).float()

def getMatr4SpaEnc(embDim):
    return [[[1,1,0,0]*int(embDim/4),[0,0,1,1]*int(embDim/4)]]

def getAttMask_MultiTrajs(validLen,Len,maskPer):
    attMaskMatr,maskedIndexArr=[],[]
    for vl in validLen:
        maskCount=torch.ceil(maskPer * vl).int()
        maskedIndex = torch.randperm(vl)[:maskCount]
        temMatr=np.array([False]*vl+(Len-vl)*[True])
        temMatr[maskedIndex]=True
        attMaskMatr.append(torch.zeros(Len, Len).masked_fill(torch.tensor(temMatr), torch.tensor(float('-inf'))).tolist())
        maskedIndexArr.append(maskedIndex)
    return attMaskMatr,maskedIndexArr

def getAttMask_MaskNextLoc(validLen,Len):
    mask_matr=[]
    for i in range(validLen):
        mask_matr.append((i+1)*[False]+(Len-i-1)*[True])
    mask_matr=torch.tensor(mask_matr)
    mask_matr=torch.cat([mask_matr,mask_matr[-1:].repeat(Len-validLen,1)],0)
    mask_matr=torch.zeros(Len, Len).masked_fill(mask_matr, torch.tensor(float('-inf')))
    return mask_matr

def getAttMask(validLen,Len,maskPer):
    maskCount=torch.ceil(maskPer * validLen).int()
    maskedIndex = torch.randperm(validLen)[:maskCount]
    temMatr=np.array([False]*validLen+(Len-validLen)*[True])
    temMatr[maskedIndex]=True
    return torch.zeros(Len, Len).masked_fill(torch.tensor(temMatr), torch.tensor(float('-inf'))),maskedIndex

def getWeightOfModel(model,exceptLayer):
    weights=[]
    for key in model.state_dict():
        if key not in exceptLayer:
            weights.append(model.state_dict()[key])
    return torch.cat([x.flatten() for x in weights])

def FedAvgForEncoder(usr_model_weights,avg_weight):
    weight_avg = copy.deepcopy(usr_model_weights[0])
    for k in weight_avg.keys():
        if ('encoder' in k) and ('Vis' not in k) :
            weight_avg[k]=weight_avg[k]*avg_weight[0]
            for i in range(1, len(usr_model_weights)):
                weight_avg[k] += usr_model_weights[i][k]*avg_weight[i]
    return weight_avg

def getWeightOfModel(model,exceptLayer):
    weights=[]
    for key in model:
        if key not in exceptLayer:
            weights.append(model[key])
    return torch.cat([x.flatten() for x in weights])

def getGumbelSoftmax(pro,temper,scalar):
    return nn.functional.gumbel_softmax(nn.functional.softmax(pro,1)*scalar,temper,False)

def getInitValForGS(batch_size,num,mode):
    if mode=='Random':
        return torch.tensor(np.random.random((batch_size,num)))
    elif mode=='Avg':
        return torch.tensor(20*np.ones((batch_size,num)))

def updateWeights(weights,modelWeights,exceptLayer):
    weight_res = copy.deepcopy(modelWeights[0])
    for k in weight_res.keys():
        if k not in exceptLayer:
            weight_res[k]=weight_res[k]*weights[0]
            for i in range(1, len(modelWeights)):  
                weight_res[k] += modelWeights[i][k]*weights[i]
    return weight_res

def PerFedUpdateWeight(outputOfPerFedModel,modelStaLis,exceptLayer):
    maxpos=outputOfPerFedModel.max(1).indices
    numOfCliOnSer=int(maxpos.max())+1
    CliOnSerDic={}
    for ind in range(len(maxpos)):
        if int(maxpos[ind]) in CliOnSerDic.keys():
            CliOnSerDic[int(maxpos[ind])].append(ind)
        else:
            CliOnSerDic[int(maxpos[ind])]=[ind]
    weightsOfCliOnSer=[]
    for i in range(numOfCliOnSer):
        weights=outputOfPerFedModel[:,i]/sum(outputOfPerFedModel[:,i])
        weightsOfCliOnSer.append(updateWeights(weights,modelStaLis,exceptLayer))
    return weightsOfCliOnSer,CliOnSerDic

def re_order(old_dic,new_dic):
    final_dic={}
    new_dic_key=list(new_dic.keys())
    old_dic_last_keys=[]
    for key in old_dic:
        aim_k=-1
        cur_len=0
        for k in new_dic_key:
            if len(set(old_dic[key])&set(new_dic[k]))>cur_len:
                aim_k=k
                cur_len=len(set(old_dic[key])&set(new_dic[k]))
        if aim_k!=-1:
            final_dic[key]=new_dic[aim_k]
            new_dic_key.remove(aim_k)
        else:
            old_dic_last_keys.append(key)
    if len(new_dic_key)!=0:
        if len(old_dic_last_keys)!=0:
            for ind in range(len(old_dic_last_keys)):
                final_dic[old_dic_last_keys[ind]]=new_dic[new_dic_key[ind]]
        else:
            for key in new_dic_key:
                final_dic[len(final_dic)]=new_dic[key]
    return final_dic

def PerFedUpdateWeightGroup(outputOfPerFedModel,modelStaLis,exceptLayer,LastCliOnSerDic={}):
    maxpos=outputOfPerFedModel.max(1).indices
    numOfCliOnSer=outputOfPerFedModel.shape[1]
    CliOnSerDic={}
    for ind in range(len(maxpos)):
        if int(maxpos[ind]) in CliOnSerDic.keys():
            CliOnSerDic[int(maxpos[ind])].append(ind)
        else:
            CliOnSerDic[int(maxpos[ind])]=[ind]
    for ind in range(numOfCliOnSer):
        if ind not in CliOnSerDic:
            CliOnSerDic[ind]=[]
    CliOnSerDic=dict([(k,CliOnSerDic[k]) for k in sorted(CliOnSerDic.keys())])
    weightsOfCliOnSer={}
    for i in range(numOfCliOnSer):
        if i in CliOnSerDic.keys():
            if len(CliOnSerDic[i])!=0:
                weights=outputOfPerFedModel[CliOnSerDic[i],i]/sum(outputOfPerFedModel[CliOnSerDic[i],i])
                tem_modelStaLis=[]
                for ind in CliOnSerDic[i]:
                    tem_modelStaLis.append(modelStaLis[ind])
                weightsOfCliOnSer[i]=updateWeights(weights,tem_modelStaLis,exceptLayer)
    return weightsOfCliOnSer,CliOnSerDic

class TemporalEncodingLayer(nn.Module):
    def __init__(self,embDim):
        super().__init__()
        self.embDim=embDim
        self.omega_t=nn.Parameter((torch.from_numpy(np.ones(int(embDim/2))*0.4)).float(), requires_grad=True)
        self.matr4Cos=getMatr4TempEnc(embDim,0)
        self.matr4Sin=getMatr4TempEnc(embDim,1)
    
    def forward(self,x,device):
        timestamp=x
        bias=torch.tensor([0,1]*int(self.embDim/2)).float()
        if type(device)!=str:
            self.matr4Cos=self.matr4Cos.to(device)
            self.matr4Sin=self.matr4Sin.to(device)
            bias=bias.to(device)
        time_encode=torch.cos(timestamp.unsqueeze(-1)*torch.mm(self.omega_t.reshape(1,-1),self.matr4Cos).reshape(1,1,-1)).float()+torch.sin(timestamp.unsqueeze(-1)*torch.mm(self.omega_t.reshape(1,-1),self.matr4Sin).reshape(1,1,-1)).float()
        time_encode=time_encode-bias
        if type(device)!=str:
            self.matr4Cos=self.matr4Cos.cpu()
            self.matr4Sin=self.matr4Sin.cpu()
            bias=bias.cpu()
        return time_encode

class SpatialEncodingLayer(nn.Module):
    def __init__(self,embDim):
        super().__init__()
        self.embDim=embDim
        self.omega_s=nn.Parameter((torch.from_numpy(np.ones(int(embDim/2))*0.4)).float(), requires_grad=True)
        self.matr4Cos=getMatr4TempEnc(embDim,0)
        self.matr4Sin=getMatr4TempEnc(embDim,1)
        self.expMatr=getMatr4SpaEnc(embDim)
    
    def forward(self,x,device):
        expMatrix=torch.from_numpy(np.repeat(self.expMatr,x.shape[0],axis=0)).float()
        bias=torch.tensor([0,1]*int(self.embDim/2)).float()
        if type(device)!=str:
            expMatrix=expMatrix.to(device)
            self.matr4Cos=self.matr4Cos.to(device)
            bias=bias.to(device)
            self.matr4Sin=self.matr4Sin.to(device)
        spatialInf=torch.matmul(x.float(),expMatrix)
        spatial_encode=torch.cos(spatialInf*torch.mm(self.omega_s.reshape(1,-1),self.matr4Cos).reshape(1,1,-1)).float()+torch.sin(spatialInf*torch.mm(self.omega_s.reshape(1,-1),self.matr4Sin).reshape(1,1,-1)).float()
        spatial_encode=spatial_encode-bias
        if type(device)!=str:
            expMatrix=expMatrix.cpu()
            self.matr4Cos=self.matr4Cos.cpu()
            bias=bias.cpu()
            self.matr4Sin=self.matr4Sin.cpu()
        return spatial_encode

class FeatureFusionAttentionLayer(nn.Module):
    def __init__(self,NumOfFeature,LocWeight):
        super().__init__()
        self.NumOfFeature=NumOfFeature
        self.w=nn.Parameter(torch.from_numpy(np.array([1.0,LocWeight,1.0,1.0,1.0])).float(), requires_grad=True)
        self.softmax=nn.Softmax(dim=0)
    
    def forward(self,x,device):
        if type(device)!=str:
            x=x.to(device)
        weights=self.softmax(self.w)
        res=weights[0]*x[0]
        for i in range(1,self.NumOfFeature):
            res+=weights[i]*x[i]
        return res

class EachLocationFeatureFusionAttentionLayer(nn.Module):
    def __init__(self,NumOfFeature,Len,LocWeight):
        super().__init__()
        self.NumOfFeature=NumOfFeature
        self.Len=Len
        self.w=nn.Parameter(torch.from_numpy((np.array([LocWeight,1.0]))).unsqueeze(1).repeat(1,Len).float(), requires_grad=True)
        self.softmax=nn.Softmax(dim=0)
    
    def forward(self,x,device):
        if type(device)!=str:
            x=x.to(device)
        weights=self.softmax(self.w).unsqueeze(2).unsqueeze(1).repeat(1,x.shape[1],1,1)
        res=torch.sum(weights*x,dim=0)
        return res

class Attention(nn.Module):

    def __init__(self, input_Dim):
        super(Attention, self).__init__()
        self.mask = None
        self.output_Dim = 10
        self.default_input_dim = input_Dim

        self.attentionLayer = nn.Sequential(
            nn.Linear(self.default_input_dim, self.output_Dim),
            nn.ReLU(),
            nn.Linear(self.output_Dim, 1))

    def set_mask(self, mask):
        self.mask = mask

    def forward(self, input):
        batch_size = input.size(0)
        hidden_size = input.size(2)
        input_size = input.size(1)

        attn = self.attentionLayer(input)

        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))

        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
        output = torch.bmm(attn, input)
        return torch.squeeze(output, 1), torch.squeeze(attn, 1)

class Encoder(nn.Module):
    def __init__(self,emb,headNum,hiddenSize,Len):
        super().__init__()
        self.locEmbDim=emb['locEmbDim']
        self.usrEmbDim=emb['usrEmbDim']
        self.VisFreEmbDim=emb['VisFreEmbDim']
        self.VisDurEmbDim=emb['VisDurEmbDim']
        self.locOHDim=emb['locOHDim']
        self.usrOHDim=emb['usrOHDim']
        self.TemporalEmbDim=emb['TemporalEmbDim']
        self.SpatialEmbDim=emb['SpatialEmbDim']
        self.locationSize=emb['locationSize']
        self.TemporalEncodingLayer=nn.Embedding(169, self.TemporalEmbDim)
        self.SpatialEmbeddingLayer=nn.Embedding(301, self.SpatialEmbDim)
        self.LocEmbeddingLayer=nn.Embedding(self.locationSize, self.locEmbDim)
        self.FeFusAttention=Attention(hiddenSize)
        self.getAttMask=getAttMask
        self.getAttMask_MaskNextLoc=getAttMask_MaskNextLoc
        self.Len=Len
        self.headNum=headNum
        
    def forward(self,x,usrOneHot,locOneHot,validLen,maskPer,device,mode):
        usrOneHot=usrOneHot.argmax(-1)
        locOneHot=locOneHot.squeeze(-1)
        locEmb=self.LocEmbeddingLayer(locOneHot)
        timEmb=self.TemporalEncodingLayer(x[:,:,4].long())
        for bat_ind in range(locEmb.shape[0]):
            cat_res=torch.cat([locEmb[bat_ind].unsqueeze(1),timEmb[bat_ind].unsqueeze(1)],axis=1)
            res,w=self.FeFusAttention(cat_res.to(torch.float32))
            res=res.unsqueeze(0)
            if bat_ind==0:
                PerLocRep=res
            else:
                PerLocRep=torch.cat([PerLocRep,res],axis=0)
        return PerLocRep

class LocationSemanticModel(nn.Module):
    def __init__(self,emb,headNum,hiddenSize,Len,SpaTemSca):
        super().__init__()
        self.locationSize=emb['locationSize']
        self.encoder=Encoder(emb,headNum,hiddenSize,Len)
        self.rnn=nn.GRU(input_size=hiddenSize, hidden_size=hiddenSize,batch_first=True,num_layers=1)
        self.FC=nn.Linear(hiddenSize,self.locationSize)
        self.pack_sequence=rnn.pack_sequence
        self.pad_packed_sequence=rnn.pad_packed_sequence
        self.pad_sequence=rnn.pad_sequence
        self.scalar=SpaTemSca[0]
        self.spatial_scalar=SpaTemSca[1]
    
    def forward(self,data,mode,device):
        if mode=='WholeModelTrain':
            zero=torch.tensor(0.0).to(device)
            x,usrOneHot,locOneHot,valLen=data['x'].to(device),data['usrOneHot'].to(device),data['locOneHot'].to(device),data['valLen'].to(device)
            spa_dis_mat_e=data['spa_dis_mat_e']
            RepreOfTraj_all=self.encoder(x,usrOneHot,locOneHot,valLen,zero,device,'Trjs')
            pred_pro=torch.tensor([])
            scalar=self.scalar
            spatial_scalar=self.spatial_scalar
            time=x[:,:,2].long()
            spa=x[:,:,0:2]
            loc_id=data['locOneHot']
            Input4LSTM,MinusTime,last_time,MinusSpa,last_spa=[],[],[],[],[]
            last_pos_ids=[]
            for valTrjs in data['ValidTrjs']:
                ValidTrjsRepOfEnc,validTime,validSpa=[],[],[]
                ind=0
                for trj in valTrjs:
                    ind+=1
                    trjIndex=trj[0][0]-data['StartTj']
                    startPoint=trj[0][1]
                    endPoint=trj[-1][1]+1
                    if len(ValidTrjsRepOfEnc)==0:
                        ValidTrjsRepOfEnc=RepreOfTraj_all[trjIndex][startPoint:endPoint]
                        validTime=time[trjIndex][startPoint:endPoint]
                        validSpa=spa[trjIndex][startPoint:endPoint]
                    else:
                        ValidTrjsRepOfEnc=torch.cat([ValidTrjsRepOfEnc,RepreOfTraj_all[trjIndex][startPoint:endPoint]])
                        validTime=torch.cat([validTime,time[trjIndex][startPoint:endPoint]])
                        validSpa=torch.cat([validSpa,spa[trjIndex][startPoint:endPoint]])
                    if ind==len(valTrjs):
                        last_pos_ids.append(int(loc_id[trjIndex][endPoint-1][0]))
                Input4LSTM.append(ValidTrjsRepOfEnc)
                MinusTime.append(validTime)
                MinusSpa.append(validSpa)
                if len(last_time)==0:
                    last_time=validTime[-1:]
                else:
                    last_time=torch.cat([last_time,validTime[-1:]])
                if len(last_spa)==0:
                    last_spa=validSpa[-1:]
                else:
                    last_spa=torch.cat([last_spa,validSpa[-1:]])
            pad_res=self.pad_sequence(MinusTime,batch_first=True)
            last_time=last_time.reshape(-1,1).repeat(1,pad_res.shape[1])
            minus_res=((last_time-pad_res)/86400)
            keep=(minus_res!=(last_time)).long()
            minus_res=minus_res*scalar
            wgh=torch.exp(-torch.abs(torch.log(keep*(np.e)))*minus_res)
            pad_res_spa=self.pad_sequence(MinusSpa,batch_first=True)
            last_spa=last_spa.reshape(-1,1,2).repeat(1,pad_res_spa.shape[1],1)
            minus_res_spa=torch.norm(last_spa-pad_res_spa,dim=-1)
            keep_spa=(minus_res_spa!=(torch.norm(last_spa,dim=-1))).long()
            minus_res_spa=minus_res_spa*spatial_scalar
            wgh_spa=torch.exp(-torch.abs(torch.log(keep_spa*(np.e)))*minus_res_spa)
            spa_tem_wgh=wgh*wgh_spa
            spa_tem_wgh_norm=spa_tem_wgh/spa_tem_wgh.sum(axis=1).reshape(-1,1)
            pack_data= self.pack_sequence(Input4LSTM,enforce_sorted=False)
            output,h_s=self.rnn(pack_data)
            hidden_state,trj_seq_len=self.pad_packed_sequence(output,batch_first=True)
            sum_state=(spa_tem_wgh_norm.unsqueeze(-1)*hidden_state).sum(axis=1)
            FC_out=self.FC(sum_state).squeeze(1)
            bias=spa_dis_mat_e[last_pos_ids]
            bias=bias.to(device)
            FC_out_bias=FC_out+bias
            return FC_out_bias

class PersonalFederatedModel(nn.Module):
    def __init__(self,batchSize,numOfCliOnSer,temper,initalMode,exceptLayer):
        super().__init__()
        self.numOfCliOnSer=numOfCliOnSer
        self.probability=nn.Parameter(getInitValForGS(batchSize,numOfCliOnSer,initalMode),requires_grad=True)
        self.GumbelSoftmax=getGumbelSoftmax
        self.temper=temper
        self.PerFedUpdateWeight=PerFedUpdateWeightGroup
        self.exceptLayer=exceptLayer
    
    def forward(self,stateLis,LastCliOnSerDic,scalar):
        raw_pro=self.probability
        result_gs=self.GumbelSoftmax(raw_pro,self.temper,scalar)
        CliOnSerDic,chosenCliOnSerDic=self.PerFedUpdateWeight(result_gs,stateLis,self.exceptLayer,LastCliOnSerDic)
        return CliOnSerDic,chosenCliOnSerDic

class LSTMSingle(nn.Module):
    def __init__(self, location_size, embedding_dim, hidden_size, num_layers):
        super(LSTMSingle, self).__init__()
        self.location_size = location_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_dim=embedding_dim
        
        self.LocEmbeddingLayer=nn.Embedding(self.location_size, self.hidden_size)
        self.lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size,batch_first=True,num_layers=self.num_layers)
        self.fc = nn.Linear(hidden_size, location_size)
        self.pack_sequence=rnn.pack_sequence
        self.softmax=nn.Softmax(dim=1)
        
    def forward(self, x, ValidTrjs, StartTj, device):
        x=x.squeeze(-1).to(device)
        locEmb=self.LocEmbeddingLayer(x)
        Input4LSTM=[]
        for valTrjs in ValidTrjs:
            ValidTrjsRepOfEnc=[]
            for trj in valTrjs:
                trjIndex=trj[0][0]-StartTj
                startPoint=trj[0][1]
                endPoint=trj[-1][1]+1
                if len(ValidTrjsRepOfEnc)==0:
                    ValidTrjsRepOfEnc=locEmb[trjIndex][startPoint:endPoint]
                else:
                    ValidTrjsRepOfEnc=torch.cat([ValidTrjsRepOfEnc,locEmb[trjIndex][startPoint:endPoint]])
            if type(device)!=str:
                ValidTrjsRepOfEnc=ValidTrjsRepOfEnc.to(device)
            Input4LSTM.append(ValidTrjsRepOfEnc)
        locEmb=0
        pack_data= self.pack_sequence(Input4LSTM,enforce_sorted=False)
        _,(h_s,_)= self.lstm(pack_data)
        fc_output = self.fc(h_s).squeeze(0)
        return fc_output