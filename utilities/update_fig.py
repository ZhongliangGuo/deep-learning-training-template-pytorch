from matplotlib.axes import Axes
from matplotlib.figure import Figure


def update_fig(fig: Figure, ax1: Axes, ax2: Axes, train_his, eval_his):
    ax1.clear()
    ax1.plot(train_his[0], color='blue', label='train')
    ax1.plot(eval_his[0], color='coral', label='eval')
    ax1.set_title('loss')
    ax1.set_xlabel('epoch')
    ax1.legend(loc="upper right")

    ax2.clear()
    ax2.plot(train_his[1], color='blue', label='train')
    ax2.plot(eval_his[1], color='coral', label='eval')
    ax2.set_title('accuracy')
    ax2.set_xlabel('epoch')
    ax2.legend(loc="upper right")

    fig.canvas.draw()
    fig.canvas.flush_events()
