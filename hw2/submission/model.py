import torch
import torch.nn as nn
import pytorch_lightning as pl
#from sklearn.preprocessing import StandardScaler

class LitGenericClassifier(pl.LightningModule):
    """
    General purpose classification model in PyTorch Lightning.
    The 2 models for the 2 respective datasets are inherited from this class.

    The 2 inherited classes define the model along with the choice of the optimizer.
    Rest of the code which is responsible for setting up training is common to both.
    """
    def __init__(self, lr=0):
        super().__init__()
        self.lr = lr
        self.loss_func = nn.CrossEntropyLoss()
        self.model = nn.Sequential() # modify this in individual model class

    def forward(self, x):
        out = self.model(x)
        #out = nn.functional.log_softmax(x, dim=1)
        return out

    def training_step(self, batch, batch_idx=0):
        """
        Arguments
        =================
        `batch`: (x, y) a python tuple.
        `x` is a torch.Tensor of size (B, d) such that B = batch size and d = input feature dimensions.
        `y` is a torch.LongTensor of size (B,) and contains input labels.
          Additional processing of both `x` and `y` may be done by calling `self.transform_input(batch)`
        before proceeding with the call. It is your responsibility to implement this function in both
        models. If you are not preprocessing the data, either don't call it at all or add a dummy 
        function as 
        ```
        def transform_input(self, batch):
            return batch
        ```
        `batch_idx`: A batch ID within a dataloader. This is an optional parameter that PyTorch 
          Lightning will use for determining how much training data has been used within an epoch.
        In general, your operation should not use `batch_idx` at all. If you think you need absolutely
        need to use it, contact TAs first.
        
        Operation
        =================
        Compute the loss and accuracy for this batch and store them in `loss` and `acc` variables.

        Returns
        =================
        `loss`: A `torch.Tensor` with correct loss value and gradient. If you are using PyTorch 
        operations, the gradient should not be destroyed. If your model is not improving or if 
        the loss becomes NaN, check this loss computation very carefully and make sure it preserves
        gradient for the autograd engine.
          PyTorch Lightning will automatically take the `loss` and run `loss.backward()` to compute 
        gradient and update weights by calling `optim.step()`. You just need to return the `loss`
        appropriately. We log these values every step so that it is easier to compare various runs.
        """
        #batch = self.transform_input(batch)
        x, y = batch
        logits = self(x)
        #print(logits)
        #print(y)
        loss = self.loss_func(logits, y)
        #loss.requires_grad = True

        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == y)/y.size(0)
        self.log('train_loss', loss.item())
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx=0):
        """
        Arguments
        =================
        `batch`: (x, y) a python tuple.
        `x` is a torch.Tensor of size (B, d) such that B = batch size and d = input feature dimensions.
        `y` is a torch.LongTensor of size (B,) and contains input labels.
          Additional processing of both `x` and `y` may be done by calling `self.transform_input(batch)`
        before proceeding with the actual implementation.
        `batch_idx`: A batch ID within a dataloader. This is an optional parameter that PyTorch 
          Lightning will use for determining how much validation data has been used during evaluation.
        In general, your operation should not use `batch_idx` at all. If you think you need absolutely
        need to use it, contact TAs first.
        
        Operation
        =================
        Compute the loss and accuracy for this batch and store them in `loss` and `acc` variables.

        Returns
        =================
        `loss`: A `torch.Tensor` or a scalar with loss value. Gradient is not required here.
        `acc`: A `torch.Tensor` or a scalar with accuracy value between 0 to 1.
          These values will be useful for you to assess overfitting and help you determine which model
        to submit on the leaderboard and in the final submission.
        """
        #batch = self.transform_input(batch)
        x, y = batch
        logits = self(x)
        #print(logits)
        #print(y)
        loss = self.loss_func(logits, y)
        loss.requires_grad = True

        preds = torch.argmax(logits, dim=1)
        #acc = self.accuracy(preds, y)
        acc = torch.sum(preds == y)/y.size(0)
        self.log('valid_loss', loss)
        self.log('valid_acc', acc)
        return {
            'valid_loss': loss,
            'valid_acc': acc,
        }

    def test_step(self, batch):
        """
        Arguments
        =================
        `batch`: (x, y) a python tuple.
        `x` is a torch.Tensor of size (B, d) such that B = batch size and d = input feature dimensions.
        `y` is a torch.LongTensor of size (B,) and contains input labels.
          Additional processing of both `x` and `y` may be done by calling `self.transform_input(batch)`
        before proceeding with the actual implementation. 
        `batch_idx`: A batch ID within a dataloader. This is an optional parameter that PyTorch 
          Lightning will use for determining how much validation data has been used during evaluation.
        In general, your operation should not use `batch_idx` at all. If you think you need absolutely
        need to use it, contact TAs first.
        
        Operation
        =================
        Compute the loss and accuracy for this batch and store them in `loss` and `acc` variables.

        Returns
        =================
        `loss`: A `torch.Tensor` or a scalar with loss value. Gradient is not required here.
        `acc`: A `torch.Tensor` or a scalar with accuracy value between 0 to 1.
          This function is very similar to `validation_step` and will be used by the autograder while
        evaluating your model. You can simply copy over the code from `validation_step` into this if 
        you wish. Just ensure that this calculation is correct.
        """
        #batch = self.transform_input(batch)
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y)
        #loss.requires_grad = True

        preds = torch.argmax(logits, dim=1)
        #acc = self.accuracy(preds, y)
        acc = torch.sum(preds == y)/y.size(0)
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        return {
            'test_loss': loss,
            'test_acc': loss,
        }
    
    def predict(self, x):
        """
        Arguments
        =================
        `x`: `torch.Tensor` of size (B, d) such that B = batch size and d = input feature dimensions.
          You can optinally transform this appropriately using `self.transform_input(batch)` but you 
        may need to create fake labels so that the function call stays the same. Something like this
        could work: `self.transform_input((x, torch.zeros(x.size(0)).long()))`
        
        Operation
        =================
        Classify each instance of `x` into appropriate classes.

        Returns
        =================
        `y_pred`: `torch.LongTensor` of size (B,) such that `y_pred[i]` for 0 <= i < B is the label
        predicted by the classifier for `x[i]`
        """
        #x,y = self.transform_input((x, torch.zeros(x.size(0)).long()))
        logits = self(x)
        y_pred = torch.argmax(logits, dim=1)
        return y_pred

class LitSimpleClassifier(LitGenericClassifier):
    def __init__(self, lr=0):
        super().__init__(lr=lr)
        self.model = nn.Sequential(
            nn.Linear(2, 5), # d = 2
            nn.ReLU(),
            #nn.Linear(3, 5), # d = 2
            #nn.ReLU(),
            #nn.Linear(4, 5), # d = 2
            #nn.ReLU(),
            #..., # build your model here using `torch.nn.*` modules
            nn.Linear(5, 4) # num_classes = 4
        )

    def transform_input(self, batch):
        # hardcode your transform here
        #x, y = batch
        #x = StandardScaler.fit_transform(x)
        #batch = (x,y)
        return batch

    def configure_optimizers(self):
        # choose an optimizer from `torch.optim.*`
        # use `self.lr` to set the learning rate
        # other parameters (e.g. momentum) may be hardcoded here
        return torch.optim.SGD(self.model.parameters(), lr = self.lr, momentum=0.9)

class LitDigitsClassifier(LitGenericClassifier):
    def __init__(self, lr=0):
        super().__init__(lr=lr)
        self.model = nn.Sequential(
            nn.Linear(64, 256), # d = 64
            nn.ReLU(),
            #nn.Linear(64, 128), # d = 2
            #nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128), # d = 2
            nn.ReLU(),
            #nn.Linear(256, 512), # d = 2
            #nn.ReLU(),
            #nn.Linear(512, 128), # d = 2
            #nn.ReLU(),
            #nn.Linear(64, 32), # d = 2
            #nn.ReLU(),
            #..., # build your model here using `torch.nn.*` modules
            nn.Linear(128, 10), # num_classes = 10
            # nn.LogSoftmax()
        )
    
    def transform_input(self, batch):
        # hardcode your transform here
        #x, y = batch
        #x = self.conv1(x)
        #x = nn.functional.relu(x)
        #x = self.conv2(x)
        #x = nn.functional.relu(x)
        #x = self.pool1(x)
        #x = self.conv3(x)
        #x = nn.functional.relu(x)
        #x = self.conv4(x)
        #x = nn.functional.relu(x)
        #x = self.pool2(x)
        #print(x.data.view(batch_size, -1).size(1))
        #x = x.view(x.size(0, -1))
        #batch = (x,y)
        return batch

    def configure_optimizers(self):
        # choose an optimizer from `torch.optim.*`
        # use `self.lr` to set the learning rate
        # other parameters (e.g. momentum) may be hardcoded here
        return torch.optim.Adam(self.model.parameters(), lr = self.lr, weight_decay=0.001)
