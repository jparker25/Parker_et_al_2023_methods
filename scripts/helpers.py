import os

def makeNice(axes):
    if type(axes) == list:
        for axe in axes:
            for i in ['left','right','top','bottom']:
                if i != 'left' and i != 'bottom':
                    axe.spines[i].set_visible(False)
                    axe.tick_params('both', width=0,labelsize=8)
                else:
                    axe.spines[i].set_linewidth(3)
                    axe.tick_params('both', width=0,labelsize=8)
    else:
        for i in ['left','right','top','bottom']:
                if i != 'left' and i != 'bottom':
                    axes.spines[i].set_visible(False)
                    axes.tick_params('both', width=0,labelsize=8)
                else:
                    axes.spines[i].set_linewidth(3)
                    axes.tick_params('both', width=0,labelsize=8)

def run_cmd(str):
    print(str)
    os.system(str)
        