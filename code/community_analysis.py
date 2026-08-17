from time import perf_counter
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import numpy as np
from scipy import sparse
from slim4_util.visualization import (
    polygon_shp_to_patch, polygon_shp_centroids
)
import igraph as ig
from typing import List, Tuple
import leidenalg as la
from leidenalg.VertexPartition import ModularityVertexPartition
import networkx as nx
import matplotlib.font_manager as fm
import geopandas as gpd

ft_country = fm.FontProperties(fname="/export/homes/dhanuise/GBR/Fonts/metropolis-bold.otf")
ft_country.set_size(15)
ft_country2 = fm.FontProperties(fname="/export/homes/dhanuise/GBR/Fonts/metropolis-bold.otf")
ft_country2.set_size(13)

# %%
# Parameters
species = 'Group1'
wind = 'BARRA' #'ACCESS' #'ERA5' #'BARRA'
basedir = "/export/homes/dhanuise/GBR/"
land_shp = "/export/homes/dhanuise/GBR/shp/Land_no_PNG/Land_no_PNG.shp"
#reef_shp = "/export/homes/dhanuise/GBR/shp/seagrass_GBR/Seagrass_patches_GBR.shp"
cm_file_dir = "/export/cephfs/tmp/dhanuise/Group1/{year}/Group1_{season}_{year}/cm_seagrass_GBR.npz"
#seeding_file = "/export/work/dhanuise/lpt/2009/Group1_09-12_2009/n_released.npy"
significance_threshold = 0.0 
min_width = 0.3 #0.01 
max_width = 0.7
width = 0.5
figname = f"/export/homes/dhanuise/GBR/Chap4_lpt_wind/impact_connectivity/figs/communities/{species}_{wind}_community_2011.png"

years = [2011]
seasons = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

cm_total = None
n_seeded_total = None

reef_shp = "path_to/shp/Group1_fragments.shp"
spp_shp = "path_to/shp/Halophila_ovalis.shp"

def compute_cm_total(species, wind, years, seasons):
    cm_total = None
    n_seeded_total = None
    
    for year in years:
        for season in seasons:

            if wind == 'ACCESS':
                cm_file = f"/path_to/connectivity_matrices/ACCESS/cm_ACCESS_{season}-{year}.npz"
                seeding_file = f"/path_to/connectivity_matrices/ACCESS/n_released_ACCESS_{season}-{year}.npy"
    
            elif wind == 'BARRA':
                cm_file = f"/path_to/connectivity_matrices/BARRA/cm_BARRA_{season}-{year}.npz"
                seeding_file = f"/path_to/connectivity_matrices/BARRA/n_released_BARRA_{season}-{year}.npy"

            elif wind == 'ERA5':
                cm_file = f"/path_to/connectivity_matrices/ERA5/cm_ERA5_{season}-{year}.npz"
                seeding_file = f"/path_to/connectivity_matrices/ERA5/n_released_ERA5_{season}-{year}.npy"
    
            try:
                cm = sparse.load_npz(cm_file)
                n_seeded = np.load(seeding_file)

                if cm_total is None:
                    cm_total = cm.copy()
                    n_seeded_total = n_seeded.copy()
                else:
                    cm_total += cm
                    n_seeded_total += n_seeded
    
            except FileNotFoundError:
                pass

    return cm_total, n_seeded_total


# Clustering functions
def compute_leiden_communities(
    cm: sparse.csr_matrix, min_size: int = 2
) -> Tuple[ ModularityVertexPartition, List[List[int]] ]:
    """
    Compute strongly connected components of the connectivity matrix

    Parameters:
        cm -- Absolute connectivity matrix (sparse matrix)
        min_size -- minimum size of communities (default: 2)

    Returns:
        partition -- Leiden partition object
        comms -- list of communities (lists of reef indices)
    """
    g = ig.Graph.Weighted_Adjacency(cm)
    tic = perf_counter()
    partition = la.find_partition(g, la.ModularityVertexPartition)#, weights='weights')
    pids, n_node = np.unique(partition.membership, return_counts=True)
    comms = [ np.where(partition.membership == pid)[0] for pid in pids if n_node[pid] >= min_size ]
    toc = perf_counter()
    print(f'Found {len(comms)} communities in {toc-tic:.2f} seconds')
    return partition, comms

def compute_scc(
    cm: sparse.csr_matrix, ns: np.ndarray, threshold: float = 0.0, min_size: int = 2
) -> List[List[int]] :
    """
    Compute strongly connected components of the connectivity matrix

    Parameters:
        cm -- Absolute connectivity matrix (sparse matrix)
        ns -- number of particles seeded by reef
        threshold -- significance threshold on the normalized connectivity matrix edges (default: 0.0)
        min_size -- minimum size of the strongly connected components (default: 2)

    Returns:
        sccs -- list of strongly connected components (lists of reef indices)
    """
    row, col = cm.nonzero()
    normalized = cm.data / ns[row]
    sel = normalized > threshold
    new_cm = sparse.csr_matrix((normalized[sel], (row[sel], col[sel])), shape=cm.shape)
    tic = perf_counter()
    graph = nx.from_scipy_sparse_array(new_cm, create_using=nx.DiGraph())
    comm_tmp = [ list(c) for c in nx.strongly_connected_components(graph) ]
    sccs = sorted( [c for c in comm_tmp if len(c) >= min_size], key=len, reverse=True )
    toc = perf_counter()
    print(f'Found {len(sccs)} strongly connected components in {toc-tic:.2f} seconds')
    return sccs

def jitter(arr, scale=0.2):
    return arr + np.random.normal(0, scale, size=len(arr))

def community_mean_latitude(comm, lonlat):
    return np.mean(lonlat[comm, 1])

def get_comm_size_and_latitude(comms, lonlat):
    sizes = []
    lats = []
    for c in comms:
        sizes.append(len(c))
        lats.append(community_mean_latitude(c, lonlat))
    return np.array(sizes), np.array(lats)

def plot_scc(ax, cm_total, n_seeded_total, lonlat, land_patches, reef_patches,
             wind, species, significance_threshold):
    if wind in ['ACCESS', 'BARRA', 'ERA5']:
        # --- Compute SCC ---
        comms, partition = compute_leiden_communities(cm_total, min_size=2) #compute_scc(cm_total, n_seeded_total, threshold=significance_threshold)
        
        comms_supp1 = [c for c in comms if len(c) > 1]
        print(f"Number of communities with more than 1 reef: {len(comms_supp1)}")

        # quantitative results
        # number of communities
        number_communities = len(comms_supp1)
        print(f"Number of communities for {wind} wind: {number_communities}")
        # mean size of communities
        sizes_communities = [len(c) for c in comms_supp1]
        mean_size_communities = np.mean(sizes_communities)
        print(f"Mean size of communities for {wind} wind: {mean_size_communities:.2f}")
        # size of the largest community
        largest_community_size = np.max(sizes_communities)
        print(f"Largest community size for {wind} wind: {largest_community_size}")

        # --- Base map ---
        land = PatchCollection(land_patches, fc="lightgrey", ec="k", lw=.2)
        reefs = PatchCollection(reef_patches, fc='k', ec=None, lw=.01)
        ax.add_collection(land)

        # Set axis
        ax.set_xlim(142, 154)
        ax.set_ylim(-25.5, -9.5)
        yticks = np.arange(-25.5, -9.5, 1)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{y} S" for y in yticks])
        ax.tick_params(labelleft=True, labelbottom = True)
        xticks = np.arange(142, 154, 2) 
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{x} E" for x in xticks])
        ax.set_aspect("equal")

        scalebar = AnchoredSizeBar(ax.transData,
                            2*0.909, '200 km', 'center', 
                            pad=0.2,
                            color='black',
                            frameon=False,
                            size_vertical=0.06,
                            fontproperties=ft_country,
                            sep =5,
                            label_top=True,
                            bbox_to_anchor=(0.12, 0.05),
                            bbox_transform=ax.transAxes)
                            
        ax.add_artist(scalebar)

        # N arrow
        ax.arrow(153.5, -10.5, 0, 0.3, length_includes_head=True,
                head_width=0.17, head_length=0.4, overhang=.13, facecolor='black')
        ax.annotate('N', xy=(153.5, -10.5+0.3), xytext=(153.5, -10.5+0.3+0.1),
                    ha='center', va='center', fontsize=12)

        # Ajout villes principales
        # Townsville
        ax.text(146.8169-2.9, -19.259-0.2, "Townsville", fontsize=13, horizontalalignment = 'left', verticalalignment = 'center', zorder=6,fontproperties=ft_country2)
        # Cairns
        ax.text(145.7667-1.8, -16.9167-0.2, "Cairns", fontsize=13, horizontalalignment = 'left', verticalalignment = 'center', zorder=6,fontproperties=ft_country2)
        # Cooktown
        ax.text(145.2522-3.0, -15.4722-0.1, "Cooktown", fontsize=13, horizontalalignment = 'left', verticalalignment = 'center', zorder=6,fontproperties=ft_country2)
        # Gladstone
        ax.text(151.2561-2.9, -23.8436-0.2, "Gladstone", fontsize=13, horizontalalignment = 'left', verticalalignment = 'center', zorder=6,fontproperties=ft_country2)
        # lizard island
        ax.text(145.4500+0.2, -14.6700+0.2, "Lizard Island", fontsize=13, horizontalalignment = 'left', verticalalignment = 'center', zorder=6,fontproperties=ft_country2)
        # Whitsunday
        ax.text(148.7167-3.4, -20.2833-0.2, "Whitsunday", fontsize=13, horizontalalignment = 'left', verticalalignment = 'center', zorder=6,fontproperties=ft_country2)
       
        # --- Plot SCC clusters ---
        cmap = plt.get_cmap('tab20')
        num_clusters = len(comms_supp1)
        colors = [cmap(i / num_clusters) for i in range(num_clusters)]

        for i, comm in enumerate(comms_supp1):
            comm_lonlat = lonlat[comm]
            ax.scatter(comm_lonlat[:, 0], comm_lonlat[:, 1], color=colors[i]) #s=10
        if wind == 'ACCESS':
            ax.set_title(f"(a) {wind}", fontsize=14)
        elif wind == 'BARRA':
            ax.set_title(f"(b) {wind}", fontsize=14)
        elif wind == 'ERA5':
            ax.set_title(f"(c) {wind}", fontsize=14)
    elif wind == 'scatter':
        ax.set_title(f"(d) Community size", fontsize=14)
        # ACCESS
        comms_access, part = compute_leiden_communities(cm_filtered_access, min_size=2) #compute_scc(cm_filtered_access, n_filtered_access, threshold=significance_threshold)
        comms_access_supp1 = [c for c in comms_access if len(c) > 1]
        sizes_access, lats_access = get_comm_size_and_latitude(comms_access_supp1, lonlat)
        # BARRA
        comms_barra, part = compute_leiden_communities(cm_filtered_barra, min_size=2) #compute_scc(cm_filtered_barra, n_filtered_barra, threshold=significance_threshold)
        comms_barra_supp1 = [c for c in comms_barra if len(c) > 1]
        sizes_barra,  lats_barra  = get_comm_size_and_latitude(comms_barra_supp1,  lonlat)
        # ERA5
        comms_era5, part = compute_leiden_communities(cm_filtered_era5, min_size=2) #compute_scc(cm_filtered_era5, n_filtered_era5, threshold=significance_threshold)
        comms_era5_supp1 = [c for c in comms_era5 if len(c) > 1]
        sizes_era5,   lats_era5   = get_comm_size_and_latitude(comms_era5_supp1, lonlat)

        ax.scatter(
            sizes_access, jitter(lats_access),
            label='ACCESS', alpha=0.7, s=50
        )

        ax.scatter(
            sizes_barra,jitter(lats_barra),
            label='BARRA', alpha=0.7, s=50
        )

        ax.scatter(
            sizes_era5,jitter(lats_era5),
            label='ERA5', alpha=0.7, s=50
        )
        ax.set_ylim(-25.5, -9.5)
        ax.set_xlim(0, None)
        ax.set_box_aspect(16/12)
        yticks = np.arange(-25.5, -9.5, 1)
        ax.set_yticks(yticks)
        # label x
        ax.set_xlabel('Number of meadows per communities', fontsize=14)
        ax.set_yticklabels([f"{y} S" for y in yticks])

        #ax.set_xlabel('Community size')#, fontsize=14)
        ax.set_title(f"(d) Community size", fontsize=14)

        ax.legend(fontsize=14)
        ax.grid(alpha=0.4)
        
    return ax

# Load shapefiles
land_patches = polygon_shp_to_patch(land_shp)
reef_patches = polygon_shp_to_patch(spp_shp)
lonlat = np.array(polygon_shp_centroids(spp_shp))

# Compute matrices for both winds
cm_access, n_access = compute_cm_total(species, "ACCESS", years, seasons)
cm_barra,  n_barra  = compute_cm_total(species, "BARRA",  years, seasons)
cm_era5,  n_era5  = compute_cm_total(species, "ERA5",  years, seasons)

# filter species H. ovalis
# Load shapefiles
reefs = gpd.read_file(reef_shp)
spp = gpd.read_file(spp_shp)
print('spp shape:', spp.shape)
# Ensure same CRS
spp = spp.to_crs(reefs.crs)

# Get indices of reefs containing the species
valid_indices = reefs.index[reefs["Unique_ID"].isin(spp["Unique_ID"])].to_numpy() #reefs_with_species.index.unique().values

cm_filtered_access = cm_access[valid_indices, :][:, valid_indices]
cm_filtered_barra = cm_barra[valid_indices, :][:, valid_indices]
cm_filtered_era5 = cm_era5[valid_indices, :][:, valid_indices]
print('cm_filtered shape',cm_filtered_access.shape)
n_filtered_access = n_access[valid_indices]
n_filtered_barra = n_barra[valid_indices]
n_filtered_era5 = n_era5[valid_indices]

# Create big figure
fig, axes = plt.subplots(2, 2, figsize=(10, 12), constrained_layout=True)
axes = axes.flatten()
print('ACCESS')
plot_scc(
    axes[0], cm_filtered_access, n_filtered_access, lonlat, 
    land_patches, reef_patches, "ACCESS",
    species, significance_threshold=significance_threshold
)

print('BARRA')
plot_scc(
    axes[1], cm_filtered_barra, n_filtered_barra, lonlat, 
    land_patches, reef_patches, "BARRA",
    species, significance_threshold=significance_threshold
)

print('ERA5')
plot_scc(
    axes[2], cm_filtered_era5, n_filtered_era5, lonlat,
    land_patches, reef_patches, "ERA5",
    species, significance_threshold=significance_threshold
)

print('scatter plot')
plot_scc(
    axes[3], cm_filtered_era5, n_filtered_era5, lonlat, 
    land_patches, reef_patches, "scatter",
    species, significance_threshold=significance_threshold
)

plt.tight_layout()
plt.savefig(f"/export/homes/dhanuise/GBR/Chap4_lpt_wind/impact_connectivity/figs/communities/SCC_H_ovalis_3winds_2011.png",
            dpi=300, bbox_inches='tight')
plt.savefig(f"/export/homes/dhanuise/GBR/Chap4_lpt_wind/impact_connectivity/figs/communities/SCC_H_ovalis_3winds_2011.svg",
            dpi=300, bbox_inches='tight')
plt.show()
print('Done! :-D')
