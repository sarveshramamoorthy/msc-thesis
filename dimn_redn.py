import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from matplotlib import colors as mcolors
from matplotlib.patches import Patch

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

df = pd.read_csv('data/datasets/single_object/train.csv')

df['adj'] = df['pos'].apply(lambda x: x.split(' ')[0])
df['noun'] = df['pos'].apply(lambda x: x.split(' ')[1])

label_mapping = {}
for ind, val in enumerate(df['adj'].unique().tolist()):
    label_mapping[val] = ind
print(label_mapping)

df['adj_label'] = df['adj'].apply(lambda x: label_mapping[x])

# label_mapping = {}
# for ind, val in enumerate(df['noun'].unique().tolist()):
#     label_mapping[val] = ind
# print(label_mapping)

# df['noun_label'] = df['noun'].apply(lambda x: label_mapping[x])

for model_name in ['clip', 'blip', 'xvlm']:
    emb = np.load('{0}_single_object_train_embeddings.npy'.format(model_name))

    def dim_red_scatter(x, colors, label_mapping, model_name):
        color_names = list(label_mapping.keys())
        palette = np.array([mcolors.to_rgba(mcolors.CSS4_COLORS[color]) for color in color_names])

        
        f = plt.figure(figsize=(8, 8))
        ax = plt.subplot(aspect='equal')
        sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(int)])
        plt.xlim(-25, 25)
        plt.ylim(-25, 25)
        ax.axis('off')
        ax.axis('tight')

        legend_handles = [Patch(color=palette[i], label=color) for i, color in enumerate(color_names)]
        legend = ax.legend(handles=legend_handles, loc='upper right', title='Adjectives')

        plt.savefig('{0}_scatter_plot_adj.jpg'.format(model_name), format='jpg', dpi=300)
        return f, ax, sc, legend

    # def dim_red_scatter(x, colors, label_mapping):
    #     num_classes = len(np.unique(colors))
    #     palette = np.array(sns.color_palette("hls", num_classes))

    #     f = plt.figure(figsize=(8, 8))
    #     ax = plt.subplot(aspect='equal')
    #     sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=palette[colors.astype(int)])
    #     plt.xlim(-25, 25)
    #     plt.ylim(-25, 25)
    #     ax.axis('off')
    #     ax.axis('tight')

    #     legend_handles = [Patch(color=palette[i], label=label) for label, i in label_mapping.items()]
    #     legend = ax.legend(handles=legend_handles, loc='upper right', title='Nouns', bbox_to_anchor=(1,1))


    #     plt.savefig('{0}_scatter_plot_noun.jpg'.format(model_name), format='jpg', dpi=300, bbox_inches='tight')
    #     return f, ax, sc, legend


    pca = PCA(n_components=50)

    pca_res = pca.fit_transform(emb)

    tsne_with_pca = TSNE(random_state=1729)

    tsne_with_pca_res = tsne_with_pca.fit_transform(pca_res)

    dim_red_scatter(tsne_with_pca_res,df['adj_label'], label_mapping. model_name)
    # dim_red_scatter(tsne_with_pca_res,df['noun_label'], label_mapping)