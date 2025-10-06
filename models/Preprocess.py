import torch
import torch.nn as nn
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
)


class Model_my(nn.Module):
    def __init__(self, configs):
        super(Model_my, self).__init__()
        self.device = configs.gpu
        print(self.device)

        self.llama = LlamaForCausalLM.from_pretrained(
            configs.llm_ckp_dir,
            device_map=self.device,
            torch_dtype=torch.float16,
        )
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(configs.llm_ckp_dir)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.vocab_size = self.llama_tokenizer.vocab_size
        self.hidden_dim_of_llama = 4096

        for name, param in self.llama.named_parameters():
            param.requires_grad = False

    def tokenizer(self, x):
        output = self.llama_tokenizer(x, return_tensors="pt")['input_ids'].to(self.device)
        result = self.llama.get_input_embeddings()(output)
        return result

    def forecast(self, x_enc, y_enc, output_hidden_states=False):
        # x_mark_enc: [bs x T x hidden_dim_of_llama]
        x_mark_enc1 = self.tokenizer("The input series is ")
        x_mark_enc2 = self.tokenizer(", and the predicted series is ")
        x_mark_enc1 = x_mark_enc1.repeat([x_enc.shape[0], 1, 1])
        x_mark_enc2 = x_mark_enc2.repeat([x_enc.shape[0], 1, 1])
        x_mark_enc = torch.cat([x_mark_enc1, x_enc, x_mark_enc2, y_enc], dim=1)
        if not output_hidden_states:
            text_outputs = self.llama.model(inputs_embeds=x_mark_enc)[0]
            text_outputs = text_outputs[:, -1, :]
        else:
            hidden_states = self.llama.model(inputs_embeds=x_mark_enc, output_hidden_states=True)['hidden_states']
            text_outputs = [item[:, -1, :] for item in hidden_states]
        return text_outputs
        # token_list = [self.tokenizer(x_mark_enc[i]) for i in range(len(x_mark_enc))]
        # min_len = 10000000
        # for i in range(len(token_list)):
        #     if token_list[i].shape[1] < min_len:
        #         min_len = token_list[i].shape[1]
        # token_list = [item[:, :min_len] for item in token_list]
        # x_mark_enc = torch.cat(token_list, 0)
        # if not output_hidden_states:
        #     text_outputs = self.llama.model(inputs_embeds=x_mark_enc)[0]
        #     text_outputs = text_outputs[:, -1, :]
        # else:
        #     hidden_states = self.llama.model(inputs_embeds=x_mark_enc, output_hidden_states=True)['hidden_states']
        #     text_outputs = [item[:, -1, :] for item in hidden_states]
        # return text_outputs

    def forward(self, x_enc, y_enc, output_hidden_states=False):
        return self.forecast(x_enc, y_enc, output_hidden_states)


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.device = configs.gpu
        print(self.device)
        
        self.llama = LlamaForCausalLM.from_pretrained(
            configs.llm_ckp_dir,
            device_map=self.device,
            torch_dtype=torch.float16,
        )
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(configs.llm_ckp_dir)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.vocab_size = self.llama_tokenizer.vocab_size
        self.hidden_dim_of_llama = 4096
        
        for name, param in self.llama.named_parameters():
            param.requires_grad = False

    def tokenizer(self, x):
        output = self.llama_tokenizer(x, return_tensors="pt")['input_ids'].to(self.device)
        result = self.llama.get_input_embeddings()(output)
        return result   
    
    def forecast(self, x_mark_enc, output_hidden_states=False):
        # x_mark_enc: [bs x T x hidden_dim_of_llama]
        token_list = [self.tokenizer(x_mark_enc[i]) for i in range(len(x_mark_enc))]
        min_len = 10000000
        for i in range(len(token_list)):
            if token_list[i].shape[1] < min_len:
                min_len = token_list[i].shape[1]
        token_list = [item[:, :min_len] for item in token_list]
        x_mark_enc = torch.cat(token_list, 0)
        if not output_hidden_states:
            text_outputs = self.llama.model(inputs_embeds=x_mark_enc)[0]
            text_outputs = text_outputs[:, -1, :]
        else:
            hidden_states = self.llama.model(inputs_embeds=x_mark_enc, output_hidden_states=True)['hidden_states']
            text_outputs = [item[:, -1, :] for item in hidden_states]
        return text_outputs
    
    def forward(self, x_mark_enc, output_hidden_states=False):
        return self.forecast(x_mark_enc, output_hidden_states)