from models import LVICL_Llama, LVICL_Gpt2, LVICL_Opt_1b


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'LVICL_Llama': LVICL_Llama,
            'LVICL_Gpt2': LVICL_Gpt2,
            'LVICL_Opt1b': LVICL_Opt_1b
        }
        self.model = self._build_model()

    def _build_model(self):
        raise NotImplementedError

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
