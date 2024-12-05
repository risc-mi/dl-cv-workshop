import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import torch


def resize(img: Image.Image, s: int = 48) -> Image.Image:
    w, h = img.size
    if w < h:
        x = (h - w) // 2
        return img.resize((s, s), box=(0, x, w, x + w))
    elif w > h:
        x = (w - h) // 2
        return img.resize((s, s), box=(x, 0, x + h, h))
    elif w == s:
        return img
    else:
        return img.resize((s, s))


class Canvas:

    def __init__(self, fig, ax, model):
        self.fig = fig
        self.ax = ax
        self.model = model
        self.lines = []
        self.xs = []
        self.ys = []
        self.pressed = False
        self.cidpress = fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = fig.canvas.mpl_connect('motion_notify_event', self.on_move)
        self.cidaxleave = fig.canvas.mpl_connect('axes_leave_event', self.on_release)
        self.cidfigleave = fig.canvas.mpl_connect('figure_leave_event', self.on_release)

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        elif event.button == 3:
            self.ax.clear()
            self.ax.set_xlim(-1, 1)
            self.ax.set_ylim(-1, 1)
            self.lines = []
            self.xs = []
            self.ys = []
            self.on_release(event)
        else:
            self.pressed = True
            self.xs = [event.xdata]
            self.ys = [event.ydata]
            self.lines.append(self.ax.plot(self.xs, self.ys, c='k', lw=20)[0])
            self.fig.canvas.draw()

    def on_release(self, event):
        self.pressed = False
        x = self.get_image().reshape(1, 1, 28, 28)
        self.model.eval()
        with torch.inference_mode():
            y = torch.argmax(self.model(torch.tensor(x))[0]).cpu().numpy()
        self.ax.set_title('Predicted: ' + str(y))

    def on_move(self, event):
        if self.pressed and event.inaxes == self.ax:
            self.xs.append(event.xdata)
            self.ys.append(event.ydata)
            self.lines[-1].set_data(self.xs, self.ys)

    def disconnect(self):
        self.fig.canvas.mpl_disconnect(self.cidpress)
        self.fig.canvas.mpl_disconnect(self.cidrelease)
        self.fig.canvas.mpl_disconnect(self.cidmotion)
        self.fig.canvas.mpl_disconnect(self.cidaxleave)
        self.fig.canvas.mpl_disconnect(self.cidfigleave)
    
    def get_pillow(self) -> Image.Image:
        # don't use `fig.canvas.get_width_height()`
        size = (self.fig.get_size_inches() * self.fig.dpi).astype(np.int32)
        return Image.frombytes('RGBA', (size[0], size[1]), self.fig.canvas.buffer_rgba())

    def get_image(self) -> np.ndarray:
        img = self.get_pillow()
        w, h = img.size
        x = np.asarray(resize(img.crop((w // 23, h // 10, w - w // 25, h - h // 25)), s=28))[:, :, :3].mean(axis=2)
        return 1. - x.astype(np.float32) / 255.


def canvas(model):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_title('Predicted: ?')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.tight_layout()
    return Canvas(fig, ax, model)
