import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase
import os

xsize = 31
ysize = 101
epsilon = 1E-5

plt.style.use('classic')
tick_font_size = 28
label_font_size = 28
plt.rc('axes',facecolor='white')
plt.rc('savefig',facecolor='white')
plt.rc('text', usetex = True)
plt.rc('font', family='serif',weight='normal')
plt.rc('hatch', color='b')
fig, ax = plt.subplots(
    1, 3, sharey=True, 
    figsize=(18, 7),
    gridspec_kw={'wspace': 0},  # No horizontal space between plots
    facecolor="none"
)
for i in range(3):
    ax[i].tick_params(axis='both', which='major', labelsize=tick_font_size)
    ax[i].set_xscale("log")
    ax[i].set_yscale("log")
ax[0].set_ylabel(r"$\Delta m_{41}^2$ $[\mathrm{eV}^2]$", fontsize = 28)


ax[0].set_xlim((10**(-3),0.25))
ax[0].set_ylim((1e1,1e5))
ax[1].set_xlim((10**(-3),0.25))
ax[1].set_ylim((1e1,1e5))
ax[2].set_xlim((10**(-2.5),0.5))
ax[2].set_ylim((1e1,1e5))

#ax[0].axvline(x=0.0339998863636364, color='gray', linestyle='dotted',linewidth = 2,zorder = 1)
#ax[1].axvline(x=0.0348484890909091, color='gray', linestyle='dotted',linewidth = 2,zorder = 2)

#ax[2].axvline(x=0.0661778755361725, color='gray', linestyle='dotted',linewidth = 2)
ax[0].set_xlabel(r"$\left|U_{e 4}\right|^2$", fontsize = 28)
ax[1].set_xlabel(r"$\left|U_{\mu 4}\right|^2$", fontsize = 28)
ax[2].set_xlabel(r"$\left|U_{\tau 4}\right|^2$", fontsize = 28)




# =============================================================================
# class three_line_handler(HandlerBase):
#     def create_artists(self, legend, orig_handle, x0, y0, width, height, fontsize, trans):
#         l1 = plt.Line2D([x0+width/6,y0+width*5/6], [0.5*height,0.5*height], linestyle=orig_handle[0], color=orig_handle[1])
#         l2 = plt.Line2D([x0+width/6,y0+width*5/6], [0.25*height,0.25*height], linestyle=orig_handle[0], color=orig_handle[2])
#         l3 = plt.Line2D([x0+width/6,y0+width*5/6], [0.*height,0.*height], linestyle=orig_handle[0], color=orig_handle[3])
#         return [l1, l2, l3]
# ax[1].legend([((0, (5, 5)),(0, 0, 1),(0.6, 0, 0.6),(1, 0, 0)), ("-",(0, 0, 1),(0.6, 0, 0.6),(1, 0, 0))], [r'$\sigma_\mathrm{norm} = 10\%$', r'$\sigma_\mathrm{norm} = 20\%$'], 
#                 handler_map={tuple: three_line_handler()},loc='upper left', frameon=False, fontsize = 20)
# 
# class three_line_handler(HandlerBase):
#     def create_artists(self, legend, orig_handle, x0, y0, width, height, fontsize, trans):
#         l1 = plt.Line2D([x0+width/6,y0+width*5/6], [0.67*height,0.67*height], linestyle=orig_handle[1], color=orig_handle[0])
#         l2 = plt.Line2D([x0+width/6,y0+width*5/6], [0.33*height,0.33*height], linestyle=orig_handle[2], color=orig_handle[0])
#         return [l1, l2]
# ax[2].legend([((0, 0, 1), (0, (5, 5)), "-"),((0.6, 0, 0.6),(0, (5, 5)), "-"), ((1, 0, 0), (0, (5, 5)), "-")], [r'$R_\mathrm{L/S} = 0\%$',r'$R_\mathrm{L/S} = 10\%$', r'$R_\mathrm{L/S} =100\%$'], 
#                 handler_map={tuple: three_line_handler()},loc='upper left', frameon=False, fontsize = 20)
# =============================================================================



# Define a unified custom handler that works for both types
class CombinedLegendHandler(HandlerBase):
    def create_artists(self, legend, orig_handle, x0, y0, width, height, fontsize, trans):
        if len(orig_handle) == 4:
            # For handles with 4 elements: (linestyle, color1, color2, color3)
            ls = orig_handle[0]
            c1, c2, c3 = orig_handle[1], orig_handle[2], orig_handle[3]
            l1 = plt.Line2D([x0+width/6, x0+width*5/6],
                            [0.5*height, 0.5*height],
                            linestyle=ls, color=c1, linewidth = 2.5)
            l2 = plt.Line2D([x0+width/6, x0+width*5/6],
                            [0.25*height, 0.25*height],
                            linestyle=ls, color=c2, linewidth = 2.5)
            l3 = plt.Line2D([x0+width/6, x0+width*5/6],
                            [0*height, 0*height],
                            linestyle=ls, color=c3, linewidth = 2.5)
            return [l1, l2, l3]
        elif len(orig_handle) == 3:
            # For handles with 3 elements: (color, linestyle1, linestyle2)
            c = orig_handle[0]
            ls1, ls2 = orig_handle[1], orig_handle[2]
            l1 = plt.Line2D([x0+width/6, x0+width*5/6],
                            [0.67*height, 0.67*height],
                            linestyle=ls1, color=c, linewidth = 2.5)
            l2 = plt.Line2D([x0+width/6, x0+width*5/6],
                            [0.33*height, 0.33*height],
                            linestyle=ls2, color=c, linewidth = 2.5)
            return [l1, l2]

# Manually combine your legend handles and labels from both axes:
combined_handles = [
    ((0, (5, 5)), (0, 0, 1), (0.6, 0, 0.6), (1, 0, 0)),   # sigma_norm = 10%
    ("-", (0, 0, 1), (0.6, 0, 0.6), (1, 0, 0)),             # sigma_norm = 20%
    ((0, 0, 1), (0, (5, 5)), "-"),                          # R_L/S = 0%
    ((0.6, 0, 0.6), (0, (5, 5)), "-"),                      # R_L/S = 10%
    ((1, 0, 0), (0, (5, 5)), "-")                      # R_L/S = 100%
]
combined_labels = [
    r'$\sigma_\mathrm{norm} = 10\%$',
    r'$\sigma_\mathrm{norm} = 20\%$',
    r'$R_\mathrm{F/N} = 0\%$',
    r'$R_\mathrm{F/N} = 10\%$',
    r'$R_\mathrm{F/N} = 100\%$',
    ""
]

# Remove or comment out the individual legends on ax[1] and ax[2] in your plotting code.
# Then, after your plotting is complete, add the figure-level legend:
fig.legend(
    combined_handles,
    combined_labels,
    handler_map={tuple: CombinedLegendHandler()},
    loc='upper center',
    bbox_to_anchor=(0.52, 1.01),
    ncol=5,  # Set this to 5 for a single horizontal row of items
    frameon=False,
    fontsize=28,
    columnspacing=0.5,    # <-- reduce horizontal gap between entries
    handletextpad=0.4,    # <-- reduce space between marker and text
    handlelength=1.5,     # <-- shorten the little line/marker length
    labelspacing=0.2
)

flv = ["e","mu","tau"]

for ax_i in range(3):
    if ax_i < 2:
        x0 = torch.logspace(-3, np.log10(0.25), 31)
    else:
        x0 = torch.logspace(-2.5, np.log10(0.5), 31)
        
    y0 = torch.linspace(10**(1/4),1e5**(1/4),101)**4
    X0,Y0 = torch.meshgrid(x0, y0)
    
    if ax_i == 0:
        ax[0].contourf(X0,Y0,X0, levels=[0.0339998863636364,10000], colors = (np.array([0.9,0.9,0.9]),np.array([0.9,0.9,0.9])),zorder = 0)#, linestyles = "dotted",linewidths = 2)
        ax[0].contour(X0,Y0,X0, levels=[0.0339998863636364,10000], colors = (np.array([0.9,0.9,0.9])/1.1,np.array([0.9,0.9,0.9])/1.1),linewidths=2,zorder=1)#, linestyles = "dotted",linewidths = 2)
    if ax_i == 2:
        ax[2].contourf(X0,Y0,X0, levels=[0.0661778755361725,10000], colors = (np.array([0.9,0.9,0.9]),np.array([0.9,0.9,0.9])),zorder = 0)#, linestyles = "dotted",linewidths = 2)
        ax[2].contour(X0,Y0,X0, levels=[0.0661778755361725,10000], colors = (np.array([0.9,0.9,0.9])/1.1,np.array([0.9,0.9,0.9])/1.1),linewidths=2,zorder=1)#, linestyles = "dotted",linewidths = 2)
    
    constraints_name = os.listdir("../"+flv[ax_i]+"/data/constraints/same_param_space")
    constarints_paths = [os.path.join("../"+flv[ax_i]+"/data/constraints/same_param_space", item) for item in os.listdir("../"+flv[ax_i]+"/data/constraints/same_param_space")]
    chisq_3nu = torch.load("../"+flv[ax_i]+"/data/best_fit/chisq.zip", weights_only = False)
    colors = np.array([0.3, 1, 0.3])

    for i, path in enumerate(constarints_paths):
        constraints_data = np.loadtxt(path+"/data.csv", delimiter=",")
        ax[ax_i].fill_betweenx(constraints_data[:,1], 1, constraints_data[:,0], color = colors ,zorder = 1)
        ax[ax_i].plot(constraints_data[:,0], constraints_data[:,1], color = colors/1.1, linewidth = 2,zorder = 2)
    data = []
    index_draw = []
    for i in range(2):
        for j in range(3):
            try:
                
                chisq_nume = torch.load("../"+flv[ax_i]+"/data/MC_analyze/chisq_nume_"+str(i)+str(j)+".zip", weights_only = False)
                chisq_deno = torch.load("../"+flv[ax_i]+"/data/MC_analyze/chisq_deno_"+str(i)+str(j)+".zip", weights_only = False).min(1)[0]
                coord = torch.load("../"+flv[ax_i]+"/data/MC/MC_coord_"+str(i)+str(j)+".zip", weights_only = False).state_dict()["tensor"].to(dtype=torch.long)

                try:
                    chisq_nume1 = torch.load("../"+flv[ax_i]+"/data/MC_analyze/chisq_nume2_"+str(i)+str(j)+".zip", weights_only = False)
                    chisq_deno1 = torch.load("../"+flv[ax_i]+"/data/MC_analyze/chisq_deno2_"+str(i)+str(j)+".zip", weights_only = False).min(1)[0]
                    coord1 = torch.load("../"+flv[ax_i]+"/data/MC/MC2_coord_"+str(i)+str(j)+".zip", weights_only = False).state_dict()["tensor"].to(dtype=torch.long)
                    chisq_nume = torch.concat((chisq_nume, chisq_nume1))
                    chisq_deno = torch.concat((chisq_deno, chisq_deno1))
                    coord = torch.concat((coord, coord1))
                    
                except:
                    0
                    

                chisq_3nu_coord = chisq_3nu[i,j][coord[:, 0], coord[:, 1]]
                data_size = 1024

                extreme = torch.arctanh((2 - epsilon) * (torch.gt(chisq_3nu_coord.unsqueeze(1).expand((coord.size(0),data_size)),chisq_nume-chisq_deno).to(torch.int).sum(1)/data_size  - 0.5))

                if ax_i < 2:
                    X = (coord[:, 0]/30 * (3+np.log10(0.25)) - 3)
                else:
                    X = (coord[:, 0]/30 * (2.5+np.log10(0.5)) - 2.5)
#                Y = 10**(coord[:, 1]/100 * 4 + 1)
                Y = (coord[:, 1]/100*(1e5**(1/4)-1e1**(1/4))+1e1**(1/4))
                #if i == 1 and j == 2:
                #    ax[ax_i].scatter(10**X, Y**(4))
                contours = plt.tricontour((X), (Y), extreme, levels=[-1000,np.arctanh((2 - epsilon) * (0.9 - 0.5))])
                log_contour_paths = []
                for segs_in_level in contours.allsegs:
                    for seg in segs_in_level:
                        # seg is an Nx2 array of vertices
                        x_line = seg[:, 0]
                        y_line = seg[:, 1]
        
                        # Apply your transformation, for example a log transform:
                        x_log = 10**x_line
                        y_log = (y_line)**4

                        log_contour_paths.append(np.column_stack([x_log, y_log]))
                data.append(log_contour_paths)
                index_draw.append([i,j])
            except:
                0
            

            

    #for  i in range(32):
    #    CS = ax.tricontour(coordinates[0], coordinates[1], number_of_not_accpet_SM[i], levels=(5000,9000,9900), colors = [(1, 0, 0)])

    k = 0
    for i in range(2):
        for j in range(3):
            linestyles = []
            colors = []
            if i == 0:
                linestyles = [(0, (5, 5))]
            if i == 1:
                linestyles = '-'
            if j == 0:
                colors = [(0, 0, 1)]
            if j == 1:
                colors = [(0.6, 0, 0.6)]
            if j == 2:
                colors = [(1, 0, 0)]
            if [i,j] in index_draw:
                for line in data[k]:
                    ax[ax_i].plot(line[:,0], line[:,1], color = colors[0], linestyle=linestyles[0], linewidth = 2.5)
                k+=1
            #ax[ax_i].contour(X0, Y0, torch.sqrt(chisq_3nu[i,j]), levels=[-1000,np.sqrt(4.61)], colors = "grey", linestyles=linestyles)
            #ax[ax_i].contour(X0, Y0, torch.sqrt(chisq_3nu[i,j]), levels=[-1000,np.sqrt(18.549)], colors = "grey", linestyles=linestyles)





    
    
# =============================================================================
#     ax.set_title(r"$R_\mathrm{s/b}="+str(int(1/background_factor))+"$",fontsize = 24, pad=12)
#     ax.text(1.46E-1, 2.8E1, r'$R_\mathrm{F/N}=100\%$',fontsize=18,ha='center', rotation = -8, color = (1,0,0))
#     ax.text(1.5E-1, 4.6E1, r'$R_\mathrm{F/N}=10\%$',fontsize=18,ha='center', rotation = -11, color = (0.6,0,0.6))
#     ax.text(1.75E-1, 1.7E2, r'$\mathrm{NSND\ only}$',fontsize=18,ha='center', rotation = -9, color = (0,0,1))
#     ax.text(0.18/1.1, 3E4, r'$\mathrm{SK}$',fontsize=18,ha='center', rotation = 90, color = "grey")
#     ax.text(0.15/1.1, 2E4, r'$\mathrm{IceCube}$',fontsize=18,ha='center', rotation = 90)
# =============================================================================
fig.tight_layout(rect=[0, 0, 1, 0.95])

plt.show()


fig.savefig('FC_combined.pdf',transparent=True)

