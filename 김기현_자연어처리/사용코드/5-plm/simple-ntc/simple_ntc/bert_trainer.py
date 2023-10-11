import torch
import torch.nn.utils as torch_utils

from ignite.engine import Events

from simple_ntc.utils import get_grad_norm, get_parameter_norm

VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_BATCH_WISE = 2

from simple_ntc.trainer import Trainer, MyEngine


class EngineForBert(MyEngine):

    def __init__(self, func, model, crit, optimizer, scheduler, config):
        self.scheduler = scheduler

        super().__init__(func, model, crit, optimizer, config)

    @staticmethod
    def train(engine, mini_batch):  # process function (feed forward, loss, backpropagation, gradient descent)
        # You have to reset the gradients of all model parameters
        # before to take another step in gradient descent.
        engine.model.train() # Because we assign model as class variable, we can easily access to it.
        engine.optimizer.zero_grad() # optimizer initialization

        x, y = mini_batch['input_ids'], mini_batch['labels']  # text와 label을 받아옴
        x, y = x.to(engine.device), y.to(engine.device)  # 램 메모리에 있던 것을 GPU로 옮김
        mask = mini_batch['attention_mask']  # mask도 받아옴
        mask = mask.to(engine.device)  # mask도 GPU로 옮김

        x = x[:, :engine.config.max_length]

        # Take feed-forward
        y_hat = engine.model(x, attention_mask=mask).logits

        loss = engine.crit(y_hat, y)  # cross entropy loss 구함
        loss.backward()  # backpropagation

        # Calculate accuracy only if 'y' is LongTensor,
        # which means that 'y' is one-hot representation.
        if isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor):
            accuracy = (torch.argmax(y_hat, dim=-1) == y).sum() / float(y.size(0))
        else:
            accuracy = 0

        p_norm = float(get_parameter_norm(engine.model.parameters()))  # parameter의 l2 norm
        g_norm = float(get_grad_norm(engine.model.parameters()))  # gradient의 l2 norm

        # Take a step of gradient descent.
        engine.optimizer.step()
        engine.scheduler.step()

        return {
            'loss': float(loss),
            'accuracy': float(accuracy),
            '|param|': p_norm,
            '|g_param|': g_norm,
        }

    @staticmethod
    def validate(engine, mini_batch):
        engine.model.eval()

        with torch.no_grad():  # gradient 계산할 필요가 없으므로 빠르고 메모리 절약하도록 torch.no_grad
            x, y = mini_batch['input_ids'], mini_batch['labels']
            x, y = x.to(engine.device), y.to(engine.device)
            mask = mini_batch['attention_mask']
            mask = mask.to(engine.device)

            x = x[:, :engine.config.max_length]

            # Take feed-forward
            y_hat = engine.model(x, attention_mask=mask).logits

            loss = engine.crit(y_hat, y)

            if isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor):
                accuracy = (torch.argmax(y_hat, dim=-1) == y).sum() / float(y.size(0))
            else:
                accuracy = 0

        return {
            'loss': float(loss),
            'accuracy': float(accuracy),
        }


class BertTrainer(Trainer):

    def __init__(self, config):
        self.config = config

    def train(
        self,
        model, crit, optimizer, scheduler,
        train_loader, valid_loader,
    ):
        train_engine = EngineForBert(
            EngineForBert.train,  # train 함수를 매 iteration 마다 불러줄 예정 (train process function 등록)
            model, crit, optimizer, scheduler, self.config
        )
        validation_engine = EngineForBert(
            EngineForBert.validate,  # (val process function 등록) 
            model, crit, optimizer, scheduler, self.config
        )

        EngineForBert.attach(
            train_engine,
            validation_engine,
            verbose=self.config.verbose
        )

        def run_validation(engine, validation_engine, valid_loader):  # train epoch이 끝날때마다 실행해야 함
            validation_engine.run(valid_loader, max_epochs=1)

        train_engine.add_event_handler(  # train Engine에 등록
            Events.EPOCH_COMPLETED, # event (epoch이 끝나면)
            run_validation, # function  (run_validation 함수를 실행)
            validation_engine, valid_loader, # arguments
        )
        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED, # event
            EngineForBert.check_best, # function
        )

        train_engine.run(
            train_loader,
            max_epochs=self.config.n_epochs,
        )

        model.load_state_dict(validation_engine.best_model)  # best model을 불러옴

        return model
