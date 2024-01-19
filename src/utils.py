import geopandas as gpd


def plot_winkel_tripel_map(ax):
    """
    Helper function to plot a Winkel Tripel map with a border.
    """
    border_geojson = gpd.read_file(
        'https://raw.githubusercontent.com/ALLFED/ALLFED-map-border/main/border.geojson'
    )
    border_geojson.plot(ax=ax, edgecolor='black', linewidth=0.1, facecolor='none')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])