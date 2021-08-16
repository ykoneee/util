from __future__ import print_function

import matplotlib.pyplot as plt


class IndexTracker(object):
    def __init__(self, ax, X, index_axis):
        self.ax = ax
        self.index_axis = index_axis
        self.X = X
        l = list(X.shape)
        self.slices = l[index_axis]
        del l[index_axis]
        rows, cols = l
        self.ind = self.slices // 2

        self.im = ax.imshow(self.X.take(self.ind, axis=self.index_axis))
        self.update()

    def onscroll(self, event):
        if event.button == "up":
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X.take(self.ind, axis=self.index_axis))
        self.ax.set_ylabel("slice %s" % self.ind)
        self.im.axes.figure.canvas.draw()


def plot_3dslice(P):
    fig, ax = plt.subplots(1, 3)
    tracker0 = IndexTracker(ax[0], P, 0)
    tracker1 = IndexTracker(ax[1], P, 1)
    tracker2 = IndexTracker(ax[2], P, 2)

    fig.canvas.mpl_connect("scroll_event", tracker0.onscroll)
    fig.canvas.mpl_connect("scroll_event", tracker1.onscroll)
    fig.canvas.mpl_connect("scroll_event", tracker2.onscroll)

    plt.show()


# fig, ax = plt.subplots(1, 1)
#
# X = np.random.rand(20, 20, 40)
#
# tracker = IndexTracker(ax, X)
#
#
# fig.canvas.mpl_connect("scroll_event", tracker.onscroll)
# plt.show()
