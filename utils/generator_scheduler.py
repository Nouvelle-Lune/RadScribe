# decode_scheduler.py


class DynamicDecodeHyperparamScheduler:
    def __init__(self, param_funcs):
        """
        param_funcs: dict[str, Union[Callable[[int], Any], Any]]
        """
        self.param_funcs = param_funcs

    def get_params(self, epoch: int) -> dict:
        params = {}
        for name, func_or_value in self.param_funcs.items():
            if callable(func_or_value):
                params[name] = func_or_value(epoch)
            else:
                params[name] = func_or_value
        return params


param_funcs_epoch = {
    "max_new_tokens": 128,
    "min_new_tokens": 64,
    "num_beams": lambda e: int(2 if e < 2 else (2 + (e - 2) / 2 * 3 if e < 4 else 5)),
    "do_sample": lambda e: False if e < 4 else True,
    "top_p": lambda e: 0.8 + min(e, 6) / 6 * 0.15,
    "temperature": lambda e: 1.2 - min(e, 6) / 6 * 0.4,
    "no_repeat_ngram_size": lambda e: int(2 if e < 4 else 3),
    "repetition_penalty": lambda e: 1.3 - min(e, 6) / 6 * 0.3,
}


scheduler = DynamicDecodeHyperparamScheduler(param_funcs_epoch)
