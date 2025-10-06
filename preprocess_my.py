import argparse
import torch
import torch.nn as nn
from layers.mlp import MLP
from models.Preprocess import Model_my
import numpy as np
from data_provider.data_loader import Dataset_ETT_hour, Dataset_Custom, Dataset_ETT_minute, Dataset_M4, Dataset_TSF
from torch.utils.data import DataLoader
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class Model_input(torch.nn.Module):
    def __init__(self, mlp_hidden_layers=0, use_multi_gpu=False,
                 local_rank=0, mlp_activation='relu',
                 mlp_hidden_dim=256, dropout=0.1, token_len=96, hidden_dim_of_llama=4096):
        super(Model_input, self).__init__()
        self.token_len = token_len
        if mlp_hidden_layers == 0:
            if not use_multi_gpu or (use_multi_gpu and local_rank == 0):
                print("use linear as tokenizer and detokenizer")
            self.encoder = nn.Linear(token_len, hidden_dim_of_llama)
            # self.decoder = nn.Linear(self.hidden_dim_of_llama, self.token_len)
        else:
            if not use_multi_gpu or (use_multi_gpu and local_rank == 0):
                print("use mlp as tokenizer and detokenizer")
            self.encoder = MLP(token_len, hidden_dim_of_llama,
                               mlp_hidden_dim, mlp_hidden_layers,
                               dropout, mlp_activation)
            # self.decoder = MLP(self.hidden_dim_of_llama, self.token_len,
            #                    mlp_hidden_dim, mlp_hidden_layers,
            #                    dropout, mlp_activation)

    def forward(self, x_enc):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        bs, _, n_vars = x_enc.shape
        # x_enc: [bs x nvars x seq_len]
        x_enc = x_enc.permute(0, 2, 1)
        # x_enc: [bs * nvars x seq_len]
        x_enc = x_enc.reshape(x_enc.shape[0] * x_enc.shape[1], -1)
        # fold_out: [bs * n_vars x token_num x token_len]
        fold_out = x_enc.unfold(dimension=-1, size=self.token_len, step=self.token_len)
        token_num = fold_out.shape[1]
        # times_embeds: [bs * n_vars x token_num x hidden_dim_of_llama]
        times_embeds = self.encoder(fold_out)
        return times_embeds


def main(root_path="/media/zhangjianqi/D/python_code/AutoTimes-v2/dataset/ETTh1/",
         dataset="ETTh1", save_path="None", data_path=None):
    # input model
    input_model = Model_input()
    # 给 key 加前缀
    state_dict = torch.load(root_path + "fc1.pth")
    new_state_dict = {f"encoder.{k}": v for k, v in state_dict.items()}
    # 加载到模型
    input_model.load_state_dict(new_state_dict)
    input_model = input_model.cuda()
    input_model.eval()

    # LLM model
    parser = argparse.ArgumentParser(description='AutoTimes Preprocess')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--llm_ckp_dir', type=str, default='./llama', help='llm checkpoints dir')
    parser.add_argument('--dataset', type=str, default=dataset,
                        help='dataset to preprocess')
    parser.add_argument('--seq_len', type=int, default=672, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=576, help='label length')
    parser.add_argument('--token_len', type=int, default=96, help='token length')
    parser.add_argument('--save_path', type=str, default=save_path)
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='')
    args = parser.parse_args()
    print(args.dataset)

    LLM_model = Model_my(args)
    LLM_model = LLM_model.cuda()
    LLM_model.eval()
    if args.dataset == "ETTh1":
        data_set = Dataset_ETT_hour(
            root_path,
            size=[672, 576, 96], data_path="ETTh1.csv")
    elif args.dataset == "ETTh2":
        data_set = Dataset_ETT_hour(
            root_path,
            size=[672, 576, 96], data_path="ETTh2.csv")
    elif args.dataset == "ETTm1":
        data_set = Dataset_ETT_minute(
            root_path,
            size=[672, 576, 96], data_path="ETTm1.csv")
    elif args.dataset == "ETTm2":
        data_set = Dataset_ETT_minute(
            root_path,
            size=[672, 576, 96], data_path="ETTm2.csv")
    elif args.dataset == "weather":
        data_set = Dataset_Custom(
            root_path,
            size=[672, 576, 96], data_path="weather.csv")
    elif args.dataset == "electricity":
        data_set = Dataset_Custom(
            root_path,
            size=[672, 576, 96], data_path="electricity.csv")
    elif args.dataset == "traffic":
        data_set = Dataset_Custom(
            root_path,
            size=[672, 576, 96], data_path="traffic.csv")
    elif args.dataset == "M4":
        data_set = Dataset_M4(
            root_path=root_path,
            data_path=data_path,
            flag="train",
            size=[args.seq_len, args.label_len, args.token_len],
            seasonal_patterns=args.seasonal_patterns,
        )
    else:
        data_set = Dataset_TSF(
            root_path=root_path,
            data_path=data_path,
            flag="train",
            size=[args.seq_len, args.label_len, args.token_len],
            seasonal_patterns=args.seasonal_patterns,
        )
    data_loader = DataLoader(
        data_set,
        batch_size=64,
        shuffle=False,
        num_workers=10,
        drop_last=False)
    output_list = []
    output_hidden_states = True
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
        # if i > 30000:
        #     break
        print(i, "/", data_loader.__len__())
        batch_x = batch_x.float().cuda()
        batch_y = batch_y.float().cuda()
        batch_x_emb = input_model(batch_x)
        batch_y_emb = input_model(batch_y)
        batch_x_emb = batch_x_emb.to(torch.float16)
        batch_y_emb = batch_y_emb.to(torch.float16)
        output = LLM_model(batch_x_emb, batch_y_emb, output_hidden_states)
        if not output_hidden_states:
            output_list.append(output.detach().cpu())
        else:
            # print("")
            output = [item.mean(0, keepdim=True).detach().cpu() for item in output]
            output_list.extend(output)
    result = torch.cat(output_list, dim=0)
    print(result.shape)
    # torch.save(result, save_dir_path + f'/{args.dataset}.pt')
    result_npy = result.reshape(33, -1, result.shape[-1])
    result_npy = result_npy.to(torch.float32).mean(1).permute(1, 0).to(torch.float16)
    result_npy = result_npy.detach().cpu().numpy()
    np.save(root_path+'task_v_ICL.npy', result_npy)


if __name__ == "__main__":
    main(root_path="XXX",
         dataset="ETTh2")
