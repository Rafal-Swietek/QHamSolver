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
def set_legend(axis, fontsize = 16, loc = 'best', anchor = None, frameon = False, **kwargs):
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
            axis.legend(frameon=frameon, fontsize=fontsize, loc=loc, handletextpad=0.25, handlelength = 1.25, **kwargs)
            return;
    return axis.legend(frameon=frameon, fontsize=fontsize, loc=loc, handletextpad=0.25, handlelength = 1.5, bbox_to_anchor=anchor, **kwargs)


from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib import text as mtext
import numpy as np
import math

class CurvedText(mtext.Text):
    """
    A text object that follows an arbitrary curve.
    """
    def __init__(self, x, y, text, axes, **kwargs):
        super(CurvedText, self).__init__(x[0],y[0],' ', **kwargs)

        axes.add_artist(self)

        ##saving the curve:
        self.__x = x
        self.__y = y
        self.__zorder = self.get_zorder()

        ##creating the text objects
        self.__Characters = []
        for c in text:
            if c == ' ':
                ##make this an invisible 'a':
                t = mtext.Text(0,0,'a')
                t.set_alpha(0.0)
            else:
                t = mtext.Text(0,0,c, **kwargs)

            #resetting unnecessary arguments
            t.set_ha('center')
            t.set_rotation(0)
            t.set_zorder(self.__zorder +1)

            self.__Characters.append((c,t))
            axes.add_artist(t)


    ##overloading some member functions, to assure correct functionality
    ##on update
    def set_zorder(self, zorder):
        super(CurvedText, self).set_zorder(zorder)
        self.__zorder = self.get_zorder()
        for c,t in self.__Characters:
            t.set_zorder(self.__zorder+1)

    def draw(self, renderer, *args, **kwargs):
        """
        Overload of the Text.draw() function. Do not do
        do any drawing, but update the positions and rotation
        angles of self.__Characters.
        """
        self.update_positions(renderer)

    def update_positions(self,renderer):
        """
        Update positions and rotations of the individual text elements.
        """

        #preparations

        ##determining the aspect ratio:
        ##from https://stackoverflow.com/a/42014041/2454357

        ##data limits
        xlim = self.axes.get_xlim()
        ylim = self.axes.get_ylim()
        ## Axis size on figure
        figW, figH = self.axes.get_figure().get_size_inches()
        ## Ratio of display units
        _, _, w, h = self.axes.get_position().bounds
        ##final aspect ratio
        aspect = ((figW * w)/(figH * h))*(ylim[1]-ylim[0])/(xlim[1]-xlim[0])

        #points of the curve in figure coordinates:
        x_fig,y_fig = (
            np.array(l) for l in zip(*self.axes.transData.transform([
            (i,j) for i,j in zip(self.__x,self.__y)
            ]))
        )

        #point distances in figure coordinates
        x_fig_dist = (x_fig[1:]-x_fig[:-1])
        y_fig_dist = (y_fig[1:]-y_fig[:-1])
        r_fig_dist = np.sqrt(x_fig_dist**2+y_fig_dist**2)

        #arc length in figure coordinates
        l_fig = np.insert(np.cumsum(r_fig_dist),0,0)

        #angles in figure coordinates
        rads = np.arctan2((y_fig[1:] - y_fig[:-1]),(x_fig[1:] - x_fig[:-1]))
        degs = np.rad2deg(rads)


        rel_pos = 10
        for c,t in self.__Characters:
            #finding the width of c:
            t.set_rotation(0)
            t.set_va('center')
            bbox1  = t.get_window_extent(renderer=renderer)
            w = bbox1.width
            h = bbox1.height

            #ignore all letters that don't fit:
            if rel_pos+w/2 > l_fig[-1]:
                t.set_alpha(0.0)
                rel_pos += w
                continue

            elif c != ' ':
                t.set_alpha(1.0)

            #finding the two data points between which the horizontal
            #center point of the character will be situated
            #left and right indices:
            il = np.where(rel_pos+w/2 >= l_fig)[0][-1]
            ir = np.where(rel_pos+w/2 <= l_fig)[0][0]

            #if we exactly hit a data point:
            if ir == il:
                ir += 1

            #how much of the letter width was needed to find il:
            used = l_fig[il]-rel_pos
            rel_pos = l_fig[il]

            #relative distance between il and ir where the center
            #of the character will be
            fraction = (w/2-used)/r_fig_dist[il]

            ##setting the character position in data coordinates:
            ##interpolate between the two points:
            x = self.__x[il]+fraction*(self.__x[ir]-self.__x[il])
            y = self.__y[il]+fraction*(self.__y[ir]-self.__y[il])

            #getting the offset when setting correct vertical alignment
            #in data coordinates
            t.set_va(self.get_va())
            bbox2  = t.get_window_extent(renderer=renderer)

            bbox1d = self.axes.transData.inverted().transform(bbox1)
            bbox2d = self.axes.transData.inverted().transform(bbox2)
            dr = np.array(bbox2d[0]-bbox1d[0])

            #the rotation/stretch matrix
            rad = rads[il]
            rot_mat = np.array([
                [math.cos(rad), math.sin(rad)*aspect],
                [-math.sin(rad)/aspect, math.cos(rad)]
            ])

            ##computing the offset vector of the rotated character
            drp = np.dot(dr,rot_mat)

            #setting final position and rotation:
            t.set_position(np.array([x,y])+drp)
            t.set_rotation(degs[il])

            t.set_va('center')
            t.set_ha('center')

            #updating rel_pos to right edge of character
            rel_pos += w-used
