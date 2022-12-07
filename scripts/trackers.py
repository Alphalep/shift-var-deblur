import matplotlib.pyplot as plt
import os
import scripts.coordConfig as cfg

losses_dir = cfg.losses_dir
if not os.path.exists(losses_dir):
    os.mkdir(losses_dir)

    
def count_parameters(model):
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_parameters/1e6 # in terms of millions

def plot_losses(running_train_loss, running_val_loss, train_epoch_loss, val_epoch_loss, epoch):
    fig = plt.figure(figsize=(16,16))
    fig.suptitle('loss trends', fontsize=20)
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)

    ax1.title.set_text('running val loss VS #batches')
    ax1.set_xlabel('#batches')
    ax1.set_ylabel('running val loss')
    ax1.plot(train_epoch_loss)
    
    ax2.title.set_text('epoch val loss VS #epochs')
    ax2.set_xlabel('#epochs')
    ax2.set_ylabel('epoch val loss')
    ax2.plot(val_epoch_loss)
 
    ax3.title.set_text('batch train loss VS #batches')
    ax3.set_xlabel('#batches')
    ax3.set_ylabel('batch train loss')
    ax3.plot(running_train_loss)

    ax4.title.set_text('batch disc loss VS #batches')
    ax4.set_xlabel('#batches')
    ax4.set_ylabel('batch disc loss')
    ax4.plot(running_val_loss)
    
    plt.savefig(os.path.join(losses_dir,'losses_{}.png'.format(str(epoch + 1).zfill(2))))
#Generate the forward blur