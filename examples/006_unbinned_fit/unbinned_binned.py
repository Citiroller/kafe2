import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from kafe2 import UnbinnedContainer, UnbinnedFit, UnbinnedPlotContainer
from kafe2 import HistContainer, HistFit, HistPlotContainer, HistPlot
from kafe2.fit.util.function_library import normal_distribution_pdf


data = np.random.standard_normal(1000)

u_data = UnbinnedContainer(data[:100])
h_data = HistContainer(n_bins=100, bin_range=(-3, 3), fill_data=data)
u_fit = UnbinnedFit(u_data)
h_fit = HistFit(h_data)
h_fit.do_fit()
pc = UnbinnedPlotContainer(u_fit)
plot = HistPlot(h_fit)
plot.show_fit_info_box()
plot.plot()
plt.show()


'''
fig = p.figure
ims = []
for i in range(100, len(data)):
    f.data = data[:i]
    f.do_fit()
    print(f.parameter_values)
    p.plot()
    # p.show_fit_info_box()
    artists = list(p._artist_store[0].values())
    im = []
    # flatten the array
    for artist in artists:
        try:
            artist = artist[0]
        except TypeError:  # except indexing error for already flattened elements
            pass
        artist.set_animated(True)
        im.append(artist)
    ims.append(im)

anim = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=100)
plt.show()
'''
