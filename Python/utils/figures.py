def add_subplot_axes(ax,figure,rect):
    
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = figure.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)    
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    #subax = fig.add_axes([x,y,width,height],facecolor=facecolor)  # matplotlib 2.0+
    subax = figure.add_axes([x,y,width,height])
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.15
    y_labelsize *= rect[3]**0.15
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax

from matplotlib.colors import Normalize

class InvertedNormalize(Normalize):
    def __call__(self, *args, **kwargs):
        return 1 - super(InvertedNormalize, self).__call__(*args, **kwargs)
 

def set_plot_elements(axis, xlim =[None, None], ylim=[None, None], xlabel = None, ylabel = None, xscale = None, yscale = None, set_legend = True, font_size = 10):
    """
    Setting plot style
    ---------------------------
    axis:      (subplot/plot object)
        Axis object of plot to set legend for

    xlim:       (list/tuple)
        Limits for x-axis range
    ylim:       (list/tuple)
        Limits for y-axis range

    xlabel:     (r-string)
        X-axis label
    ylabel:     (r-string)
        Y-axis label
    xscale:     (string)
        scale type for x-axis
    yscale:     (string)
        scale type for y-axis
    
    set_legend: (boolean)
        Set legend to default options
    
    font_size:   (int)
        Fontsize of handles
    """
    if xlabel is not None: axis.set_xlabel(xlabel, rotation=0, fontsize=font_size+2, labelpad=font_size-8)
    if ylabel is not None: axis.set_ylabel(ylabel, rotation=90, fontsize=font_size+2)
    if xscale is not None: axis.set_xscale(xscale)
    if yscale is not None: axis.set_yscale(yscale)

    axis.set_axisbelow(True)
    axis.tick_params(axis='both', which='major', direction="in",length=6, labelsize=font_size, bottom=True, top=True, left=True, right=True)
    axis.tick_params(axis='both', which='minor', direction="in",length=3, labelsize=font_size, bottom=True, top=True, left=True, right=True)
    
    if set_legend:
        axis.legend(frameon=False
                , loc='best'
                , fontsize=font_size)
    x1, x2 = xlim
    y1, y2 = ylim
    if x1 != None and x2 != None:
        if x1 < x2: axis.set_xlim([x1,x2])
        else: axis.set_xlim([x2,x1])
    if y1 != None and y2 != None:
        if y1 < y2: axis.set_ylim([y1, y2])
        else: axis.set_ylim([y2, y1])

# axes[0,1].legend(frameon=0, fontsize=16, loc='upper left', handletextpad=0.25, handlelength = 1.25, bbox_to_anchor=(-0.02,1.03))
def set_legend(axis, fontsize = 16, loc = 'best', anchor = None):
    """
    Setting plot legend
    ---------------------------
    axis:      (subplot/plot object)
        Axis object of plot to set legend for

    fontsize:   (int)
        Fontsize of handles
    
    loc:        (string)
        Location of legend on the plot (upper, center, lower) x (left, center, right)
    """
    if anchor is None:
        if loc == 'upper left': anchor = (-0.02, 1.02)
        elif loc == 'upper center': anchor = (0.5, 1.02)
        elif loc == 'upper right': anchor = (1.02, 1.02)
        elif loc == 'center left': anchor = (-0.02, 0.5)
        elif loc == 'center center': anchor = (0.5, 0.5)
        elif loc == 'center right': anchor = (1.02, 0.5)
        elif loc == 'lower left': anchor = (-0.02, -0.02)
        elif loc == 'lower center': anchor = (0.5, -0.02)
        elif loc == 'lower right': anchor = (1.02, -0.02)
        else:
            axis.legend(frameon=0, fontsize=fontsize, loc=loc, handletextpad=0.25, handlelength = 1.25)
            return;
    axis.legend(frameon=0, fontsize=fontsize, loc=loc, handletextpad=0.25, handlelength = 1.25, bbox_to_anchor=anchor)