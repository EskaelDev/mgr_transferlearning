class NetParams:
    def __init__(self,
                 batch_size,
                 confusion_loader,
                 criterion,
                 max_no_improve_epochs,
                 model,
                 n_epochs,
                 optimizer,
                 test_loader,
                 train_loader,
                 train_on_gpu,
                 validation_loader,
                 ):
        self.batch_size = batch_size = batch_size
        self.confusion_loader = confusion_loader
        self.criterion = criterion
        self.max_no_improve_epochs = max_no_improve_epochs
        self.model = model
        self.n_epochs = n_epochs
        self.optimizer = optimizer
        self.test_loader = test_loader
        self.train_loader = train_loader
        self.train_on_gpu = train_on_gpu
        self.validation_loader = validation_loader
