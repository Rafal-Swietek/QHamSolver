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



def set_plot_elements(axis, xlim =[None, None], ylim=[None, None], xlabel = None, ylabel = None, xscale = None, yscale = None, set_legend = True, font_size = 10):

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