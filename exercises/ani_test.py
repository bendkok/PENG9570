import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

x = np.arange(0, 2*np.pi, 0.1)
y = np.sin(x)

fig, axes = plt.subplots(nrows=6)

styles = ['r-', 'g--', 'yo', 'm^', 'k-.', 'ch']
def plot(ax, style):
    return ax.plot(x, y, style, animated=True, label="start")[0] 
lines = [plot(ax, style) for ax, style in zip(axes, styles)]
time_text = axes[2].text(1, 0.6,'',horizontalalignment='right',verticalalignment='top', transform=axes[0].transAxes) # fig.suptitle("t = {:d} of {:d}.".format(0, 200)) 

def animate(i):
    time_text.set_text("t = {:d} of {:d}".format(i, 200))
    for j, line in enumerate(lines, start=1):
        line.set_ydata(np.sin(j*x + i/10.0))
    return lines + [time_text,]

# We'd normally specify a reasonable "interval" here...
ani = animation.FuncAnimation(fig, animate, range(1, 200), 
                              interval=10, blit=True)
plt.show()


