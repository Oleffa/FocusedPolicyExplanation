import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.transforms import Bbox
from matplotlib.patches import Rectangle
import os

class Heatmap_2D:

    def __init__(self, plot_storage_path):
        self.plot_storage_path = plot_storage_path
        if not os.path.isdir(self.plot_storage_path):
            print("path ({}) does not exist, creating new folder".format(self.plot_storage_path))
            os.makedirs(self.plot_storage_path) 

    def prepare_data(self, x_res, y_res,d1,d2, explanation_generator, actions, entropy_threshold, threshold):
        # Prepare data
        eg = explanation_generator
        dims = eg.dimensions
        data = []
        p1_stab = np.zeros((y_res, x_res))
        p1_describ = np.zeros((y_res, x_res))
        p1_rel = np.zeros((y_res, x_res))
        p1_consist = np.zeros((y_res, x_res))
        p2_stab = np.zeros((y_res, x_res))
        p2_describ = np.zeros((y_res, x_res))
        p2_rel = np.zeros((y_res, x_res))
        p2_consist = np.zeros((y_res, x_res))
        for y in range(0,y_res):
            for x in range(0,x_res):
                step_size_x = dims[d1].max_range/x_res
                step_size_y = dims[d2].max_range/y_res
                # Generate a state based on the step size
                state = [round(step_size_x/2+step_size_x*x,2), round(dims[d2].max_range - step_size_y/2 - step_size_y*y,2)]
                # We have to set the unused dimension of the state as well otherwise we cant use the 3d plotting function
                for t in range(0, len(dims)-len(state)):
                    state.append(0)
                stability, describability, relevance,relevance_threshold, consistency = eg.generate_explanation(state, actions,entropy_threshold)
                p1_stab[y][x] = stability[d1]
                p1_describ[y][x] = describability[d1]
                p1_rel[y][x] = relevance[d1]
                p1_consist[y][x] = consistency[d1]
                p2_stab[y][x] = stability[d2]
                p2_describ[y][x] = describability[d2]
                p2_rel[y][x] = relevance[d2]
                p2_consist[y][x] = consistency[d2]
                print("{}/{}".format(y*y_res+x+1, y_res*x_res))
        # Prepare custom measure plots
        p1_combined = self.mask(p1_stab, p1_describ, p1_consist, self.mask_relevance(p1_rel, relevance_threshold), threshold)
        p2_combined = self.mask(p2_stab, p2_describ, p2_consist, self.mask_relevance(p2_rel, relevance_threshold), threshold)
        
        combined_stab = p1_stab * p2_stab
        combined_describ = p1_describ * p2_describ
        combined_rel = self.dim_plot(p1_rel, p2_rel, relevance_threshold) # plots for when which dimension is used to explain
        combined_consist = p1_consist * p2_consist
        combined = self.dim_plot(p1_combined, p2_combined, np.prod(threshold)) #np.zeros((y_res, x_res))
        #combined = np.zeros((y_res, x_res))
        
        data=[p1_stab, p1_describ, p1_rel, p1_consist, p1_combined, p2_stab, p2_describ, p2_rel, p2_consist, p2_combined, combined_stab, combined_describ, combined_rel, combined_consist, combined]
        return eg, data

    def plot_single(self, x_res, y_res,d1,d2, explanation_generator, policy, actions, entropy_threshold, threshold, interpolation=False, show_all_concepts=False, show_all_policies=False, savefig=False):
        eg, data = self.prepare_data(x_res, y_res,d1,d2, explanation_generator, actions, entropy_threshold, threshold)
        tags=["StabilityD{}".format(d1), "DescribabilityD{}".format(d1), "RelevanceD{}".format(d1), "ConsistencyD{}".format(d1), "Masked_CombinedD{}".format(d1), "StabilityD{}".format(d2), "DescribabilityD{}".format(d2), "RelevanceD{}".format(d2), "ConsistencyD{}".format(d2), "Masked_Combined D{}".format(d2), "StabilityCombined", "DescribabilityCombined" , "DimensionToExplainRelevance", "ConsistencyCombined", "DimensionsToExplain"]
        self.plot_data(tags, data, x_res, y_res, d1, d2, policy, eg, 0, savefig, False, True)
        self.plot_data(tags, data, x_res, y_res, d1, d2, policy, eg, 1, savefig, True, False)
        self.plot_data(tags, data, x_res, y_res, d1, d2, policy, eg, 2, savefig, False, True)
        self.plot_data(tags, data, x_res, y_res, d1, d2, policy, eg, 3, savefig, True, True)
        self.plot_data(tags, data, x_res, y_res, d1, d2, policy, eg, 4, savefig, True, True)
        self.plot_data(tags, data, x_res, y_res, d1, d2, policy, eg, 5, savefig, False, True)
        self.plot_data(tags, data, x_res, y_res, d1, d2, policy, eg, 6, savefig, True, False)
        self.plot_data(tags, data, x_res, y_res, d1, d2, policy, eg, 7, savefig, False, True)
        self.plot_data(tags, data, x_res, y_res, d1, d2, policy, eg, 8, savefig, True, True)
        self.plot_data(tags, data, x_res, y_res, d1, d2, policy, eg, 9, savefig, True, True)
        self.plot_dims_used(tags, data, x_res, y_res, d1, d2, policy, eg, 14, savefig)
    def plot_dims_used(self, tags, data, x_res, y_res, d1, d2, policy, eg, data_id, savefig):
        # Plot relevances
        cmap = 'YlGnBu'
        a = plt.imshow(data[data_id], vmin=0, vmax=3,cmap=cmap, extent=(0,x_res, 0, y_res))
        # Add colorbar for the relevance and masked plots
        labels = ['None','D0','D1','Both']
        values = np.unique(data[data_id].ravel())
        colors = [a.cmap(a.norm(value)) for value in values]
        patches = [ mpatches.Patch(color=colors[t], label="{}".format(labels[t]) ) for t in range(len(values)) ]
        # put those patched as legend-handles into the legend
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.1)
        self.heatmap_concepts_single(plt, eg, d1, d2, x_res, y_res, data_id)
        self.heatmap_policy_single(plt, x_res, y_res, policy)
        self.rebrand_axes_single(plt, x_res, y_res)
        #plt.title("Relevance")
        plt.xlabel("D0")
        plt.ylabel("D1")
        plt.tight_layout()
        # Save the figures as individual plots
        if savefig:
            #import matplotlib2tikz
            #matplotlib2tikz.save("plots/{}.tikz".format(tags[data_id]))
            plt.savefig(self.plot_storage_path+'{}.pdf'.format(tags[data_id]))
            print('saving figure in: {}/{}.pdf'.format(self.plot_storage_path, tags[data_id]))
        plt.show()

    def plot_data(self,tags, data, x_res, y_res, d1, d2, policy, eg, data_id, savefig, c, p):
        # Plot relevances
        cmap = 'viridis'
        a = plt.imshow(data[data_id], vmin=0, vmax=1,cmap=cmap, extent=(0,x_res, 0, y_res))
        # Add colorbar for the relevance and masked plots
        plt.colorbar(a)
        # put those patched as legend-handles into the legend
        if p:
            self.heatmap_policy_single(plt, x_res, y_res, policy)
        if c:
            self.heatmap_concepts_single(plt,eg,d1,d2, x_res, y_res,data_id)
        self.rebrand_axes_single(plt, x_res, y_res)
        #plt.title(tags[data_id])
        plt.xlabel("D0")
        plt.ylabel("D1")
        plt.tight_layout()
        # Save the figures as individual plots
        if savefig:
            #import matplotlib2tikz
            #matplotlib2tikz.save(self.plot_storage_path+"/{}.tikz".format(tags[data_id]))
            plt.savefig(self.plot_storage_path+'/{}.pdf'.format(tags[data_id]))
        plt.show()

    def plot(self, x_res, y_res,d1,d2, explanation_generator, policy, actions, entropy_threshold, threshold, interpolation=False, show_all_concepts=False, show_all_policies=False, savefig=False):
        eg, data = self.prepare_data(x_res, y_res,d1,d2, explanation_generator, actions, entropy_threshold, threshold)
        tags=["Stability D{}".format(d1), "Describability D{}".format(d1), "Relevance D{}".format(d1), "Consistency D{}".format(d1), "Masked + Combined D{}".format(d1), "Stability D{}".format(d2), "Describability D{}".format(d2), "Relevance D{}".format(d2), "Consistency D{}".format(d2), "Masked + Combined D{}".format(d2), "Stability Combined", "Describability Combined" , "Dimension used to explain based on relevance", "Consistency Combined", "Dimensions used to explain"]
        f, axes = plt.subplots(3,5, figsize=(33,14))
        for i in range(0, np.shape(axes)[0]):
            # Iterate through all measures to plot
            for j in range(0, np.shape(axes)[1]):
                ax_1 = axes[i,j]
                if (i == 2 and j== 2) or (i == 2 and j == 4):
                    # Plot relevances
                    #x_1.plot(np.arange(len(combined_rel[0])).tolist(), combined_rel[0], c='r')
                    #x_1.plot(np.arange(len(combined_rel[1])).tolist(), combined_rel[1], c='b')
                    # Plot in which place which dimensions are used
                    cmap = 'inferno'
                    a = ax_1.imshow(data[j+i*np.shape(axes)[1]], vmin=0, vmax=3,cmap=cmap, extent=(0,x_res, 0, y_res))
                    # Add colorbar for the relevance and masked plots
                    #plt.colorbar(a, ax=ax_1, ticks=[-1,0,1,2])
                    labels = ['None','D0','D1','Both']
                    values = np.unique(data[j+i*np.shape(axes)[1]].ravel())
                    colors = [a.cmap(a.norm(value)) for value in values]
                    patches = [ mpatches.Patch(color=colors[t], label="{}".format(labels[t]) ) for t in range(len(values)) ]
                    # put those patched as legend-handles into the legend
                    ax_1.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
                    #ax_1.legend(loc='best')
                    self.heatmap_concepts(ax_1, eg, d1, d2, x_res, y_res, i)
                    self.heatmap_policy(ax_1, x_res, y_res, policy)
                else:
                    if j is not 2:
                        vmin = 0
                        vmax = 1
                        cmap = 'viridis'
                    else:
                        vmin = 0
                        vmax = 1
                        cmap = 'viridis'
                    if interpolation:
                            a = ax_1.imshow(data[j+i*np.shape(axes)[1]], vmin=vmin, vmax=vmax,cmap=cmap, interplation='bilinear', extent=(0,x_res, 0,y_res))
                    else:
                        a = ax_1.imshow(data[j+i*np.shape(axes)[1]], vmin=vmin, vmax=vmax,cmap=cmap, extent=(0,x_res, 0,y_res))
                    # Dont draw concepts for stability and relevance (they dont rely on concepts)
                    if (j == 1 or j == 3 or j == 4 or j == 5) and show_all_concepts == False:
                        self.heatmap_concepts(ax_1, eg, d1, d2, x_res, y_res, i)
                    elif show_all_concepts == True:
                        self.heatmap_concepts(ax_1, eg, d1, d2, x_res, y_res, i)
                    # Dont draw policy for 
                    if not (j == 1) and show_all_policies == False:
                        self.heatmap_policy(ax_1, x_res, y_res, policy)
                    elif show_all_policies == True:
                        self.heatmap_policy(ax_1, x_res, y_res, policy)
                    cbar = plt.colorbar(a, ax=ax_1)
                self.rebrand_axes(f,ax_1, x_res, y_res)
                #ax_1.set_title(tags[j+i*np.shape(axes)[1]])
                ax_1.set_xlabel("D0")
                if j == 0:
                    h = ax_1.set_ylabel("D1")
                    h.set_rotation(0)
                # Save the figures as individual plots
                if savefig:
                    #extent = ax_1.get_window_extent().transformed(f.dpi_scale_trans.inverted())
                    extent = self.full_extent(ax_1).transformed(f.dpi_scale_trans.inverted())
                    #print(extent)
                    f.savefig(self.plot_storage_path+'/{}'.format(tags[j+i*np.shape(axes)[1]]), bbox_inches=extent)
        plt.show()
    def rebrand_axes(self,figure, ax, x_res, y_res):
        figure.canvas.draw()
        x_labels = [item.get_text() for item in ax.get_xticklabels()]
        y_labels = [item.get_text() for item in ax.get_yticklabels()]
        x_ticks = np.ndarray.tolist(np.linspace(0,1,len(x_labels)))
        y_ticks = np.ndarray.tolist(np.linspace(0,1,len(y_labels)))
        for i in range(1,len(x_labels)):
            x_labels[i] = abs(round(x_ticks[i],2))
        #x_labels[len(x_labels)-1] = 1
        for i in range(1,len(y_labels)):
            y_labels[i] = round(y_ticks[i],2)
        ax.set_yticklabels(y_labels)
        ax.set_xticklabels(x_labels)
    def heatmap_policy(self, ax, x_res, y_res, policy):
        x = np.linspace(0, x_res, 101)
        y = []
        for i in x:
            #if policy.getPolicyY(i/x_res)*y_res <= y_res:
            y.append(policy.getPolicyY(i/x_res)*y_res)
            #else:
            #    y.append(y_res)
        pol = ax.plot(x,y,label="Policy", linewidth=4, c='red')
        ax.set_ylim([0, y_res])

    def heatmap_concepts(self,ax, eg, d1, d2,x_res,y_res, i):
        dim1 = eg.dimensions[d1]
        dim2 = eg.dimensions[d2]
        # vertical concepts
        if i == 0:
            for k in range(0, len(dim1.thresholds)):
                ax.plot((dim1.thresholds[k]*x_res,dim1.thresholds[k]*x_res),(0,y_res-0.17*y_res), color='b', linewidth=4)
        # horizontal concepts
        elif i == 1:
            x = np.linspace(0,x_res, 101)
            for k in range(0, len(dim2.thresholds)):
                y = np.full((np.shape(x)), dim2.thresholds[k])*y_res
                ax.plot(x,y,'b',linewidth=4)
        else:
            for k in range(0, len(dim1.thresholds)):
                ax.plot((dim1.thresholds[k]*x_res,dim1.thresholds[k]*x_res),(0,y_res), color='b', linewidth=4)
            x = np.linspace(0,x_res, 101)
            for k in range(0, len(dim2.thresholds)):
                y = np.full((np.shape(x)), dim2.thresholds[k])*y_res
                ax.plot(x,y,'b',linewidth=4)
    def rebrand_axes_single(self, plt, x_res, y_res):
        ax = plt.gca()
        x_labels = [item.get_text() for item in ax.get_xticklabels()]
        y_labels = [item.get_text() for item in ax.get_yticklabels()]
        x_ticks = np.ndarray.tolist(np.linspace(0,1,len(x_labels)))
        y_ticks = np.ndarray.tolist(np.linspace(0,1,len(y_labels)))
        for i in range(1,len(x_labels)):
            x_labels[i] = abs(round(x_ticks[i],2))
        #x_labels[int(len(x_labels)/2)-1] = 1
        #x_labels[0] = 0
        for i in range(1,len(y_labels)):
            y_labels[i] = round(y_ticks[i],2)
        print(x_labels)
        ax.set_yticklabels(y_labels)
        ax.set_xticklabels(x_labels)
    def heatmap_policy_single(self, plt, x_res, y_res, policy):
        x = np.linspace(0, x_res, 101)
        y = []
        for i in x:
            #if policy.getPolicyY(i/x_res)*y_res <= y_res:
            y.append(policy.getPolicyY(i/x_res)*y_res)
            #else:
            #    y.append(y_res)
        pol = plt.plot(x,y,label="Policy", linewidth=4, c='red')
        plt.ylim([0, y_res])

    def heatmap_concepts_single(self,plt, eg, d1, d2,x_res,y_res, i):
        dim1 = eg.dimensions[d1]
        dim2 = eg.dimensions[d2]
        # vertical concepts
        if i < 5:
            for k in range(0, len(dim1.thresholds)):
                plt.plot((dim1.thresholds[k]*x_res,dim1.thresholds[k]*x_res),(0,y_res), color='b', linewidth=4)
        # horizontal concepts
        elif i > 5 and i < 14:
            x = np.linspace(0,x_res, 101)
            for k in range(0, len(dim2.thresholds)):
                y = np.full((np.shape(x)), dim2.thresholds[k])*y_res
                plt.plot(x,y,'b',linewidth=4)
        else:
            for k in range(0, len(dim1.thresholds)):
                plt.plot((dim1.thresholds[k]*x_res,dim1.thresholds[k]*x_res),(0,y_res), color='b', linewidth=4)
            x = np.linspace(0,x_res, 101)
            for k in range(0, len(dim2.thresholds)):
                y = np.full((np.shape(x)), dim2.thresholds[k])*y_res
                plt.plot(x,y,'b',linewidth=4)

    def combine_rel(self, rel1, rel2):
        """
        This functions squeezes the 
        """
        flip = False
        if flip:
            x = np.flip(np.average(rel1, axis=1))
            y = np.flip(np.average(rel2, axis=0))
        else:
            x = np.average(rel1, axis=1)
            y = np.average(rel2, axis=0)
        return [x,y]

    def dim_plot(self, in1, in2, threshold):
        """
        This function plots for each pixel in the plot which dimension is used to explain there.
        Works for both relevance and combined measures.
        """
        out = np.zeros((np.shape(in1)))
        for y in range(0, np.shape(in1)[0]):
            for x in range(0, np.shape(in1)[1]):
                if in1[y][x] > threshold:
                    out[y][x] = 1
                if in2[y][x] > threshold and not in1[y][x] > threshold:
                    out[y][x] = 2
                if in2[y][x] > threshold and in1[y][x] > threshold:
                    out[y][x] = 3            
        return out

    def mask_relevance(self,in_rel, relevance_threshold):
        out = np.zeros(np.shape(in_rel))
        out[in_rel > relevance_threshold] = 1
        return out

    def mask(self, stab, desc, cons, rel, threshold):
        out = np.zeros(np.shape(stab))
        for y in range(0, np.shape(stab)[0]):
            for x in range(0, np.shape(stab)[1]):
                if stab[y][x] <= threshold[0] or desc[y][x] <= threshold[1] or cons[y][x] <= threshold[2]:
                    out[y][x] = 0
                else:
                    out[y][x] = stab[y][x] * desc[y][x] * cons[y][x] * rel[y][x]
        return out
        
