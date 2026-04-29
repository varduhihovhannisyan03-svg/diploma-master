import torch


class StepLR:
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 decay_epochs: float,
                 decay_rate: float = 1.,
                 warmup_epochs=0,
                 warmup_lr_init=0
                 ) -> None:
        super().__init__()

        self.optimizer = optimizer

        for idx, param_group in enumerate(self.optimizer.param_groups):
            if "lr" not in param_group:
                raise KeyError(f"`lr` missing from param_groups[{idx}]")
            param_group.setdefault("initial_lr", param_group["lr"])

        self.base_values = [param_group["initial_lr"] for param_group in self.optimizer.param_groups]
        self.update_groups(self.base_values)

        self.decay_epochs = decay_epochs
        self.decay_rate = decay_rate
        self.warmup_epochs = warmup_epochs
        self.warmup_lr_init = warmup_lr_init

        self.warmup_steps = [(v - warmup_lr_init) / self.warmup_epochs for v in self.base_values]
        self.update_groups(self.warmup_lr_init)

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict) -> None:
        self.__dict__.update(state_dict)

    def step(self, epoch: int) -> None:
        if epoch < self.warmup_epochs:
            values = [self.warmup_lr_init + epoch * s for s in self.warmup_steps]
        else:
            values = [v * (self.decay_rate ** (epoch // self.decay_epochs)) for v in self.base_values]

        if values is not None:
            self.update_groups(values)

    def update_groups(self, values):
        if not isinstance(values, (list, tuple)):
            values = [values] * len(self.optimizer.param_groups)
        for param_group, value in zip(self.optimizer.param_groups, values):
            if 'lr_scale' in param_group:
                param_group["lr"] = value * param_group['lr_scale']
            else:
                param_group["lr"] = value
