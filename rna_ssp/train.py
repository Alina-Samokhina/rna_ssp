import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt

def train(model, opt,loss_func,  train_iter, val_iter, epochs = 100, scheduler = None,print_every = 2):
    tr_mean_losses = []
    val_mean_losses = []
    print_every = 1
    val_losses = []
    train_losses = []
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        running_corrects = 0
        model.train() 
        for batch in train_iter: 
            
            src = batch.src
            trg = batch.trg

            opt.zero_grad()  

            output = model(src)            
            output_dim = output.shape[-1]            
            output = output.reshape(-1, output_dim)
            trg = trg.view(-1)
            
            loss = loss_func(output, trg)
            loss.backward()                   
            opt.step()
            running_loss += loss.item()

        if scheduler:
            scheduler.step()
        
        epoch_loss = running_loss / len(train_iter)
        train_losses.append(epoch_loss)
        
        val_loss = 0.0
        model.eval()
        for batch in val_iter:
            
            src = batch.src
            trg = batch.trg
            
            output = model(src)            
            output_dim = output.shape[-1]            
            output = output.reshape(-1, output_dim)
            trg = trg.view(-1)
            
            loss = loss_func(output, trg)            
            val_loss += loss.item()  
            
        val_loss = val_loss/len(val_iter)
        val_losses.append(val_loss)
        
        if epoch % print_every == 0:
            clear_output(True)
            tr_mean_losses.append(np.array(train_losses[-print_every:]).mean())
            val_mean_losses.append(np.array(val_losses[-print_every:]).mean())
            plt.figure(figsize = (12, 8))
            print(f'Epoch: {epoch}, Training Loss: {epoch_loss:.3f}, Validation Loss: {val_loss:.3f}')
            plt.plot(np.arange(0, len(tr_mean_losses)), tr_mean_losses, label = 'train loss')
            plt.plot(np.arange(0, len(val_mean_losses)), val_mean_losses, label = 'val loss')
            plt.legend(loc = 'best')
            plt.grid(True)
            plt.show()

    return tr_mean_losses, val_mean_losses