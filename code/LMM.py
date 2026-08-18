from time import perf_counter
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
import igraph as ig
from typing import List, Tuple
import leidenalg as la
from leidenalg.VertexPartition import ModularityVertexPartition
import networkx as nx
import matplotlib.font_manager as fm
import statsmodels.formula.api as smf
import pandas as pd
import geopandas as gpd
from osgeo import ogr

def find_utm_zone(lon: float, lat: float) -> str:
    """
    Find the UTM zone of a point with given longitude and latitude

    Parameters:
        lon -- longitude of the point
        lat -- latitude of the point
    """
    copy_utm_zones()
    pnt = ogr.Geometry(ogr.wkbPoint)
    pnt.AddPoint(lon,lat)
    ds = ogr.GetDriverByName('ESRI Shapefile').Open(f'{UTM_ZONES_BASENAME}.shp',0)
    lyr = ds.GetLayer()
    utm_zone = None
    for f in lyr:
        geom = f.geometry()
        if geom.Contains(pnt):
            utm_zone = f.GetField('ZONE')
            break
    ds = None
    if utm_zone is None:
        raise ValueError(f'No UTM zone found for point ({lon},{lat})')
    hemisphere = '+south' if lat < 0 else '+north'
    return f'+proj=utm +zone={utm_zone} {hemisphere} +ellps=WGS84 +units=m'

def find_utm_zone_shp(shape_file: str) -> str:
    """
    Return the UTM zone of a shapefile

    Parameters:
        shape_file -- path to the shapefile
    """
    ds = ogr.GetDriverByName('ESRI Shapefile').Open(shape_file,0)
    lyr = ds.GetLayer()
    minx, maxx, miny, maxy = lyr.GetExtent()
    return find_utm_zone((minx+maxx)/2, (miny+maxy)/2)

# %%
# Parameters
reef_shp = "path_to/shp/Group1_fragments.shp"
spp_shp = "path_to/shp/Halophila_ovalis.shp"

year = 2011 
seasons = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
wet_seasons = ['10','11','12','01','02','03']
dry_seasons = ['04','05','06','07','08','09']

winds = ['ACCESS', 'BARRA','ERA5']

cm_total_wet = {w: None for w in winds}
n_seeded_total_wet = {w: None for w in winds}
cm_total_dry = {w: None for w in winds}
n_seeded_total_dry = {w: None for w in winds}

for wind in winds:
    print(f"Processing {wind} wind...")
    for season in seasons:
        print(f"Processing season: {season}")
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

            if season in wet_seasons:
                if cm_total_wet[wind] is None:
                    cm_total_wet[wind] = cm.copy()
                    n_seeded_total_wet[wind] = n_seeded.copy()
                else:
                    cm_total_wet[wind] += cm
                    n_seeded_total_wet[wind] += n_seeded
            else:
                if cm_total_dry[wind] is None:
                    cm_total_dry[wind] = cm.copy()
                    n_seeded_total_dry[wind] = n_seeded.copy()
                else:
                    cm_total_dry[wind] += cm
                    n_seeded_total_dry[wind] += n_seeded
            
        except FileNotFoundError:
            print(f"Fichier manquant pour {year}-{season}, ignoré.")
print('loop done!')

for wind in winds:
    total_wet = np.sum(n_seeded_total_wet[wind])
    total_dry = np.sum(n_seeded_total_dry[wind])
    total = total_wet + total_dry

    print(f"{wind}:")
    print(f"  Wet season: {total_wet:,} particles")
    print(f"  Dry season: {total_dry:,} particles")
    print(f"  Total:      {total:,} particles\n")

# Load shapefiles
reefs = gpd.read_file(reef_shp)
spp = gpd.read_file(spp_shp)
print('spp shape:', spp.shape)
# Ensure same CRS
spp = spp.to_crs(reefs.crs)

# Get indices of reefs containing the species
valid_indices = reefs.index[reefs["Unique_ID"].isin(spp["Unique_ID"])].to_numpy() 
cm_filtered_dry = {}
cm_filtered_wet = {}
n_seeded_filtered_dry = {}
n_seeded_filtered_wet = {}

for wind in winds:
    cm_filtered_dry[wind] = cm_total_dry[wind][valid_indices, :][:, valid_indices]
    cm_filtered_wet[wind] = cm_total_wet[wind][valid_indices, :][:, valid_indices]

    n_seeded_filtered_dry[wind] = n_seeded_total_dry[wind][valid_indices]
    n_seeded_filtered_wet[wind] = n_seeded_total_wet[wind][valid_indices]


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
    partition = la.find_partition(g, la.ModularityVertexPartition)
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

def compute_connectivity_metrics(cm: sparse.csr_matrix, n_seeded: np.ndarray, reef_shp, wind_name, season_type):
    """
    Compute connectivity metrics and return a tidy DataFrame
    directly usable for mixed models.
    """

    reefs = gpd.read_file(reef_shp)
    utm_projection = find_utm_zone_shp(reef_shp)

    assert cm.shape[0] == n_seeded.shape[0], \
        "n_seeded should have size equal to number of rows of cm"
    assert reefs.shape[0] == cm.shape[0], \
        "connectivity matrix and reef shape file have different shapes"
    
    n_reef = cm.shape[0]
    reefs_utm = reefs.to_crs(utm_projection)
    xc = np.column_stack([reefs_utm.centroid.x, reefs_utm.centroid.y])

    eps = np.finfo(float).eps
    rows, cols = cm.nonzero()
    diag = cm.diagonal()

    # -- Indicators based on the absolute matrix
    # Local retention: fraction of particles released on a reef that settle on that same reef
    reefs['locRet'] = np.where(n_seeded > 0, diag / n_seeded, 0.0)
    # Self-recruitment: proportion of particle settling on reef that were released on that same reef
    n_incoming = np.bincount(cols, weights=cm.data, minlength=n_reef)
    reefs['selfRec'] = diag / (n_incoming + eps)
    # In-degree: number of incoming connections
    in_degree = np.bincount(cols, minlength=n_reef)
    reefs['inDeg'] = in_degree
    # Out-degree: number of outgoing connections
    out_degree = np.bincount(rows, minlength=n_reef)
    reefs['outDeg'] = out_degree
    # Proportion settles: proportion of particles released on a reef that settle on that reef or on another reef
    n_settled = np.bincount(rows, weights=cm.data, minlength=n_reef)
    reefs['propSet'] = n_settled / (n_seeded + eps)

    # -- PangeRank-like indicators that require igraph package
    for name in ('in','out'):
        nw = ig.Graph.Weighted_Adjacency(cm if name == 'in' else cm.T)
        nw.simplify(multiple=False)
        reefs[name+"PgRnk"] = nw.pagerank(directed=True, weights='weight')

    # -- Indicators based on the normalized connectivity matrix (connection probabilities)
    cm.data[:] /= n_seeded[rows]
    alpha = 0.5
    # Weighted in-degree: in-degree taking connections weights into account
    in_weights = np.bincount(cols, weights=cm.data, minlength=n_reef)
    reefs["wInDeg"] = in_degree**(1-alpha) * in_weights**alpha
    reefs['inCon'] = in_degree * in_weights
    # Weighted out-degree: out-degree taking connection weights into account
    out_weights = np.bincount(rows, weights=cm.data, minlength=n_reef)
    reefs["wOutDeg"] = out_degree**(1-alpha) * out_weights**alpha
    reefs['outCon'] = out_degree * out_weights
    # Protection index: identify reefs that are good suppliers and poor receivers of larvae
    reefs["protect"] = (reefs["outPgRnk"]-reefs["inPgRnk"]) / (reefs["inPgRnk"] + reefs["outPgRnk"])
    # Restoration index: identify reefs that are both good suppliers and receivers of larvae
    reefs['restore'] = reefs["inPgRnk"] * reefs["outPgRnk"]
    # Weighted connectivity length: averaged connection distance
    dist = np.hypot(*(xc.T[:,cols]-xc.T[:,rows])) * 1e-3
    reefs['WCL_km'] = np.bincount(rows, weights=dist*cm.data, minlength=n_reef) / (out_weights+eps)

    # ---------- GRAPH BUILDING FOR CENTRALITY ----------
    g_abs = ig.Graph.Weighted_Adjacency(cm, mode="DIRECTED", attr="weight")

    # Closeness centrality (directed)
    reefs["close_in"] = g_abs.closeness(mode="IN", weights="weight")
    reefs["close_out"] = g_abs.closeness(mode="OUT", weights="weight")

    # Eigenvector centrality
    reefs["eigen"] = g_abs.eigenvector_centrality(directed=True, weights="weight")

    # CORRECTED BETWWEENNESS 
    cost_cm = cm.copy()

    eps = 1e-15
    cost_cm.data = -np.log(np.maximum(cost_cm.data, eps))

    g_cost = ig.Graph.Weighted_Adjacency(
        cost_cm,
        mode="DIRECTED",
        attr="weight"
    )
    reefs["betweenness"] = g_cost.betweenness(
        directed=True,
        weights="weight"
    )


    tidy_rows = []

    for i in range(n_reef):
        tidy_rows.append(["selfRec", wind_name, season_type, i, reefs["selfRec"].iloc[i]])
        tidy_rows.append(["outDeg", wind_name, season_type, i, reefs["wOutDeg"].iloc[i]])
        tidy_rows.append(["inDeg", wind_name, season_type, i, reefs["wInDeg"].iloc[i]])
        tidy_rows.append(["WCL_km", wind_name, season_type, i, reefs["WCL_km"].iloc[i]])
        tidy_rows.append(["close_in", wind_name, season_type, i, reefs["close_in"].iloc[i]])
        tidy_rows.append(["close_out", wind_name, season_type, i, reefs["close_out"].iloc[i]])
        tidy_rows.append(["eigen", wind_name, season_type, i, reefs["eigen"].iloc[i]])
        tidy_rows.append(["betweenness", wind_name, season_type, i, reefs["betweenness"].iloc[i]])


    tidy_df = pd.DataFrame(tidy_rows,
                           columns=["MetricType", "Wind","Season", "ReefID", "Value"])

    return tidy_df
    

df_list = []

for wind in winds:
    # --- wet season ---
    df_wet = compute_connectivity_metrics(
        cm_filtered_wet[wind], 
        n_seeded_filtered_wet[wind], 
        spp_shp, 
        wind_name=wind,
        season_type="wet"
    )
    df_list.append(df_wet)

    # --- dry season ---
    df_dry = compute_connectivity_metrics(
        cm_filtered_dry[wind], 
        n_seeded_filtered_dry[wind],
        spp_shp,
        wind_name=wind,
        season_type="dry"
    )
    df_list.append(df_dry)

df = pd.concat(df_list, ignore_index=True)

# --- Les 3 métriques à analyser avec EMM ---
metrics = ["selfRec", "outDeg", "inDeg", "WCL_km", 'eigen', 'betweenness']

emm_results = {}   # pour stocker les EMM

for m in metrics:
    print(f"\n=== Running LMM for {m} ===")
    df_m = df[df["MetricType"] == m].copy()

    # modèle mixte
    model = smf.mixedlm("Value ~ Wind * Season", df_m, groups=df_m["ReefID"]).fit()
    print(model.summary())

    # parametres modele
    params = model.fe_params
    cov_fe = model.cov_params().loc[params.index, params.index]

    wind_levels = ["ACCESS", "BARRA", "ERA5"]
    season_levels = ["wet", "dry"]

    rows = []
    for wind in wind_levels:
        for season in season_levels:
            X = np.zeros(len(params))
            X[params.index.get_loc("Intercept")] = 1

            if wind != "ACCESS":
                X[params.index.get_loc(f"Wind[T.{wind}]")] = 1

            if season != "dry":
                X[params.index.get_loc(f"Season[T.{season}]")] = 1

            # interaction
            if wind != "ACCESS" and season != "dry":
                term = f"Wind[T.{wind}]:Season[T.{season}]"
                if term in params.index:
                    X[params.index.get_loc(term)] = 1

            emm = X @ params.values
            se  = np.sqrt(X @ cov_fe.values @ X.T)

            rows.append([wind, season, emm, se])
    emm_results[m] = pd.DataFrame(rows, columns=["Wind","Season","emmean","SE"])

# ---------------------------------------------------------
#                 PLOT DES EMMs POUR LES 4 MÉTRIQUES
# ---------------------------------------------------------
season_colors = {
    "wet": "tab:blue",
    "dry": "tab:orange"
}

season_offsets = {
    "wet": -0.1,
    "dry":  0.1
}

fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=False)
axes = axes.flatten()
for ax, (metric, res) in zip(axes, emm_results.items()):
    # res : dataframe avec mean, SE, CI…
    wind_levels = res["Wind"].unique()
    x_base = np.arange(len(wind_levels))

    label_map = {
        "wet": "Not windy",
        "dry": "Windy"
    }

    for season in ["wet", "dry"]:
        res_season = res[res["Season"] == season]
        ax.errorbar(
            x=x_base + season_offsets[season],
            y=res_season["emmean"],
            yerr=res_season["SE"],
            fmt="o",
            color=season_colors[season],
            ecolor=season_colors[season], #'lightgray',
            capsize=4,
            markersize=6,
            label=label_map[season] if ax == axes[0] else None 
        )
    
    
    metric_labels = {
        "selfRec": "(a) Local retention", 
        "outDeg": "(b) Weighted Out-Degree",
        "inDeg": "(c) Weighted In-Degree",
        "WCL_km": "(d) Weighted Connectivity Length (km)",
        "eigen": "(e) Eigenvector Centrality",
        "betweenness": "(f) Betweenness Centrality"
    }

    ax.set_title(metric_labels.get(metric, metric), fontsize=14)
    ax.set_xticks(x_base)
    ax.set_xticklabels(wind_levels, fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    if ax==axes[0] or ax==axes[3]:
        ax.set_ylabel("Estimated Marginal Mean", fontsize=13)
    ax.grid(alpha=0.3)

axes[0].legend(title="Season", fontsize=12, title_fontsize=12)
plt.tight_layout()
plt.savefig("H_ovalis_EMM_ACCESS_ERA5_BARRA_2011_from_cm_season.png", dpi=300)
plt.show()
