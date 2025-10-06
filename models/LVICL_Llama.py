import torch
import torch.nn as nn
from transformers import LlamaForCausalLM
from layers.mlp import MLP


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.token_len = configs.token_len
        if configs.use_multi_gpu:
            self.device = f"cuda:{configs.local_rank}"
        else:
            self.device = f"cuda:{configs.gpu}"
        print(self.device)
        
        self.llama = LlamaForCausalLM.from_pretrained(
            configs.llm_ckp_dir,
            device_map=self.device,
            torch_dtype=torch.float16 if configs.use_amp else torch.float32,
            # torch_dtype=torch.float32,
        )
        self.hidden_dim_of_llama = 4096
        self.mix = configs.mix_embeds
        if self.mix:
            self.add_scale = nn.Parameter(torch.ones([]))
            temp = torch.ones([33]).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            self.add_scale_33 = nn.Parameter(temp)
            temp1 = torch.ones([33]).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            self.add_scale_33_m = nn.Parameter(temp1)
            self.add_scale_33_m = None
        
        for name, param in self.llama.named_parameters():
            # print(name)
            # if "input_layernorm" not in name:
            param.requires_grad = False

        if configs.mlp_hidden_layers == 0:
            if not configs.use_multi_gpu or (configs.use_multi_gpu and configs.local_rank == 0):
                print("use linear as tokenizer and detokenizer")
            self.encoder = nn.Linear(self.token_len, self.hidden_dim_of_llama)
            self.decoder = nn.Linear(self.hidden_dim_of_llama, self.token_len)
        else:
            if not configs.use_multi_gpu or (configs.use_multi_gpu and configs.local_rank == 0):
                print("use mlp as tokenizer and detokenizer")
            self.encoder = MLP(self.token_len, self.hidden_dim_of_llama, 
                            configs.mlp_hidden_dim, configs.mlp_hidden_layers, 
                            configs.dropout, configs.mlp_activation)
            self.decoder = MLP(self.hidden_dim_of_llama, self.token_len,
                            configs.mlp_hidden_dim, configs.mlp_hidden_layers,
                            configs.dropout, configs.mlp_activation)
        self.inter_f = nn.Linear(4096, 4096)
        # self.inter_f.weight.data.zero_()
        # self.inter_f.bias.data.zero_()
        # self.inter_f = MLP(4096, 4096, 8192, activation='relu')
    
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, output_hidden_states=False):
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

        # x_mark_enc = x_mark_enc / x_mark_enc.norm(dim=2, keepdim=True)
        # times_embeds = times_embeds / times_embeds.norm(dim=2, keepdim=True)
        # if self.mix:
        #     x_mark_enc_1 = x_mark_enc[:, :, :, -1].clone()
        #     times_embeds = times_embeds / times_embeds.norm(dim=2, keepdim=True)
        #     x_mark_enc_1 = x_mark_enc_1 / x_mark_enc_1.norm(dim=2, keepdim=True)
        #     times_embeds = times_embeds + self.add_scale * x_mark_enc_1
        #     # times_embeds = times_embeds / times_embeds.norm(dim=2, keepdim=True)
        #     # x_mark_enc = x_mark_enc / x_mark_enc.norm(dim=2, keepdim=True)
        #     # times_embeds = times_embeds + self.add_scale * x_mark_enc
        # x_mark_enc = None
        # outputs: [bs * n_vars x token_num x hidden_dim_of_llama]

        x_mark_enc = x_mark_enc.permute(0, 1, 3, 2)
        x_mark_enc = self.inter_f(x_mark_enc).permute(0, 1, 3, 2)
        # # # x_mark_enc = x_mark_enc[:, :, :, -1:]
        # # x_mark_enc = x_mark_enc.repeat([1, 1, 1, 33])
        # # x_mark_enc = x_mark_enc * self.add_scale_33

        # x_mark_enc = x_mark_enc
        # x_mark_enc = x_mark_enc.repeat(times_embeds.shape[0], 1, 1, 1)
        # x_mark_enc = torch.cat([times_embeds.unsqueeze(-1).repeat(1, 1, 1, 33), x_mark_enc], dim=1)
        # x_mark_enc = x_mark_enc.permute(0, 2, 3, 1)
        # x_mark_enc = self.inter_f(x_mark_enc)
        # x_mark_enc = x_mark_enc.permute(0, 3, 1, 2)

        # times_embeds = times_embeds.unsqueeze(-1)
        # times_embeds = torch.cat([times_embeds, x_mark_enc], dim=-1)
        # outputs = self.llama.model(
        #     inputs_embeds=times_embeds, inputs_mark_enc=x_mark_enc)[0]
        outputs_init = self.llama.model(
            inputs_embeds=times_embeds, inputs_mark_enc=x_mark_enc,
            add_scale_33_m=self.add_scale_33_m, output_hidden_states=output_hidden_states)
        if output_hidden_states:
            outputs = outputs_init[0]
            hidden_states = outputs_init['hidden_states']
            hidden_states_avg = [item[:, -1].mean(0) for item in hidden_states]
        else:
            outputs = outputs_init[0]
        # dec_out: [bs * n_vars x token_num x token_len]
        dec_out = self.decoder(outputs)
        dec_out = dec_out.reshape(bs, n_vars, -1)
        # dec_out: [bs x token_num * token_len x n_vars]
        dec_out = dec_out.permute(0, 2, 1)
        
        dec_out = dec_out * \
            (stdev[:, 0, :].unsqueeze(1).repeat(1, token_num * self.token_len, 1))
        dec_out = dec_out + \
            (means[:, 0, :].unsqueeze(1).repeat(1, token_num * self.token_len, 1))
        if output_hidden_states:
            return dec_out, hidden_states_avg
        else:
            return dec_out
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, output_hidden_states=False):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, output_hidden_states)
