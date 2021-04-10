class NetParams:
    def __init__(self,
                 model,
                 criterion,
                 optimizer,
                 n_epochs,
                 max_no_improve_epochs,
                 train_on_gpu,
                 train_loader,
                 test_loader,
                 validation_loader,
                 batch_size):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.max_no_improve_epochs = max_no_improve_epochs
        self.train_on_gpu = train_on_gpu
        self.train_loader = train_loader = train_loader
        self.test_loader = test_loader = test_loader
        self.validation_loader = validation_loader = validation_loader
        self.batch_size = batch_size = batch_size
