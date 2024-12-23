"""README

This file contains all the code needed the assignment 1 for the Natural Computing
course 24/25. Please see the example at the bottom of this file on how to interact 
with the TSP object. 

Installation
------------
    Call this in a terminal: 
        pip install "geopandas<1.0.0" numpy matplotlib pandas svgpath2mpl

"""

import warnings
import math
import io
import typing

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import svgpath2mpl
import geopandas


DATA = """hckey,capital,capital_lat,capital_lng
ad,Andorra,42.5,1.5165
al,Tirana,41.3275,19.8189
at,Vienna,48.2,16.3666
ba,Sarajevo,43.85,18.383
be,Brussels,50.8333,4.3333
bg,Sofia,42.6833,23.3167
by,Minsk,53.9,27.5666
ch,Bern,46.9167,7.467
cy,Nicosia,35.1667,33.3666
cz,Prague,50.0833,14.466
de,Berlin,52.5218,13.4015
dk,Copenhagen,55.6786,12.5635
ee,Tallinn,59.4339,24.728
es,Madrid,40.4,-3.6834
fi,Helsinki,60.1756,24.9341
fr,Paris,48.8667,2.3333
gb,London,51.5072,-0.1275
gr,Athens,37.9833,23.7333
hr,Zagreb,45.8,16
hu,Budapest,47.5,19.0833
ie,Dublin,53.3331,-6.2489
is,Reykjavik,64.15,-21.95
it,Rome,41.896,12.4833
li,Vaduz,47.1337,9.5167
lt,Vilnius,54.6834,25.3166
lu,Luxembourg,49.6117,6.13
lv,Riga,56.95,24.1
mc,Monaco,43.7396,7.4069
md,Chisinau,47.005,28.8577
me,Podgorica,42.466,19.2663
mk,Skopje,42,21.4335
mt,Valletta,35.8997,14.5147
nl,Amsterdam,52.35,4.9166
no,Oslo,59.9167,10.75
pl,Warsaw,52.25,21
pt,Lisbon,38.7227,-9.1449
ro,Bucharest,44.4334,26.0999
rs,Belgrade,44.8186,20.468
ru,Moscow,55.7522,37.6155
se,Stockholm,59.3508,18.0973
si,Ljubljana,46.0553,14.515
sk,Bratislava,48.15,17.117
sm,San Marino,43.9172,12.4667
ua,Kiev,50.4334,30.5166"""

LEIDEN_SVG = """M380.52 239.51l-30.679-29.744-29.757-34.948-14.23-13.797v-13.258l-33.266-32.255-10.352 10.033-7.763-7.524-10.348 10.034 5.917 5.734-17.746 
17.203-14.047-13.62 10.721-10.395-10.351-10.033-10.718 10.395-12.197-11.827 10.717-10.395-11.452-11.107-10.721 10.391-16.266-15.767 
2.957-2.868 18.481-17.918 11.46 11.107 9.982-9.675-6.656-6.45 17.004-16.488-6.653-6.45 14.416-13.978 120.868 117.197 
29.754 31.71v18.457L403.9 216.84h20.608c2.352 0 8.24-8.424 10.436-10.142 6.777-5.306 13.24-4.346 21.31-3.43 13.316 1.514 23.955 
9.485 33.313 18.203 26.695 24.87 21.382 53.31.735 79.32.788-.988 7.349 10.174 7.555 11.301.79 4.372-.726 9.239-3.29 12.907-5.464 
7.82-15.208 8.218-24.226 8.231l-6.744-6.542c-11.747 11.39-16.922 16.034-33.659 16.037-31.782.003-53.845-10.81-66.787-40.495v-25.804l4.434-12.904 
12.936-12.543V239.51zm65.036 69.75c27.483-15.113 49.34-48.926 26.594-70.203-11.619-10.87-22.464-10.176-36.766-4.208-.032 0 .252 
30.598-4.734 30.598l-5.029 4.879-29.907.402c-14.831 14.377-7.986 36.849 11.629 44.275 11.318 4.286 25.378 1.311 38.213-5.744zm55.91 
13.359l-5.166-10.18 5.166 10.18zm-298.33-6.814l-3.457 6.814 3.458-6.814a14.054 14.054 0 0 0 1.152-3.714c.206-1.127 
6.767-12.289 7.555-11.3-20.647-26.01-25.96-54.45.735-79.32 9.358-8.72 19.997-16.69 33.312-18.205 8.07-.915 14.534-1.875 
21.31 3.431 2.197 1.718 8.085 10.142 10.437 10.142h20.608l24.304-23.565v-18.457l29.754-31.71L472.172 25.91l14.416 
13.978-6.653 6.45 17.004 16.487-6.656 6.45 9.983 9.676 11.459-11.107 18.48 17.918 2.958 2.868-16.266 15.767-10.72-10.391-11.453 
11.107 10.717 10.395-12.197 11.827-10.718-10.395-10.351 10.033 10.72 10.395-14.046 13.62-17.746-17.203 5.917-5.734-10.348-10.034-7.763 
7.524-10.352-10.033-33.266 32.255v13.258l-14.23 13.797-29.757 34.948-30.679 29.744v11.468l12.936 12.543 4.434 12.904v25.804c-12.942 
29.685-35.004 40.498-66.787 40.495-16.737-.003-21.912-4.648-33.659-16.037l-6.744 6.542c-9.018-.013-18.762-.412-24.225-8.23-1.854-2.652-3.16-5.93-3.443-9.194zm293.125-3.444L490.491 
301l5.77 11.36zm-240.672-3.102c12.835 7.055 26.895 10.03 38.213 5.744 19.615-7.426 26.46-29.898 11.63-44.275l-29.908-.402-5.029-4.879c-4.986 
0-4.702-30.598-4.734-30.598-14.302-5.968-25.147-6.662-36.766 4.208-22.745 21.277-.889 55.09 26.594 70.202z
"""
LEIDEN = 4.497010, 52.160114


def plot_europe(
    data: pd.DataFrame, fig: plt.Figure = None, ax: plt.Axes = None
) -> typing.Tuple[plt.Figure, plt.Axes]:
    """Plotting utilitly, plots a map of Europe, with Leiden explitly marked

    Parameters
    ----------
    data: pd.DataFrame
        Locations of cities on the map

    fig: mpl.Figure (optional)
    ax: plt.Axes (optional)
        Optional figure and axes for plotting

    Returns
    -------
        (mpl.Figure, plt.Axes,)
            Handles to the plot
    """

    if fig is None:
        fig, ax = plt.subplots(1, 1, figsize=(15, 8))

    keys = svgpath2mpl.parse_path(LEIDEN_SVG)
    keys.vertices -= keys.vertices.mean(axis=0)

    keys = keys.transformed(mpl.transforms.Affine2D().rotate_deg(180))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        world = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))

    world.plot(ax=ax, color="lightgray", edgecolor="black", alpha=0.5)

    ax.scatter(data["capital_lng"], data["capital_lat"], s=10, color="blue", alpha=0.5)

    ax.scatter(*LEIDEN, color="red", marker=keys, s=5_000, alpha=0.6)
    ax.scatter(*LEIDEN, color="red")
    ax.text(*LEIDEN, "Leiden", ha="right", **{"fontsize": 11})

    for _, c in data.iterrows():
        ax.text(c["capital_lng"], c["capital_lat"], c["capital"], **{"fontsize": 10})

    ax.set_xlim(-24, 40)
    ax.set_ylim(34, 70)
    plt.tight_layout()
    return fig, ax


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance in kilometers between two points
    on the earth (specified in decimal degrees)

    Parameters
    ----------
    lat1: float
        Latitude for point 1
    lon1: float
        Longtitude for point 1
    lat2: float
        Latitude for point 2
    lon2: float
        Longtitude for point 1

    Returns
    -------
        float
            The Haversine distance between point1 and point 2
    """

    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    r = 6371
    return c * r


class TSP:
    """Traveling Salesperson object, with plotting utility"""

    def __init__(self, plot: bool = True):
        """Create a Traveling Salesperson object

        Parameters
        ----------
        plot: bool = True
            Whether to create an (interactive) plot. When running this for the optimization
            its advised to turn this off, as it can take quite a bit of extra time to visualize
            the tour on every function call.
        """

        self.data = pd.read_csv(io.StringIO(DATA))
        self.plot = plot
        self.line = None
        self.dim = len(self.data)

    def __enter__(self):
        """Create a plot, i.e. figure and axes, if self.plot == True."""

        if self.plot:
            plt.ion()
            _, self.ax = plot_europe(self.data)
        return self

    def __exit__(self, *args, **kwargs):
        """Stop plotting interactively, but keep showing the plot if self.plot == True."""

        if self.plot:
            plt.ioff()
            plt.show()

    def __call__(self, path_idx: np.array) -> float:
        """Calculate the route length of the TSP problem.
        The method returns the total length of the tour, as given by the order specified
        by path_idx, caculated as the sucessive Haversine distance between cities.

        Parameters
        ----------
        path_idx: np.ndarray[int]
            Indices of the cities you will visit, i.e. the order. This should be an integer
            array of size n, where n is the number of cities. Every integer from 1 to n
            should appear in this array, in arbitrary order, but no duplicates are allowed.

        Returns
        -------
            float The length of the tour
        """
        assert len(path_idx) == len(self.data), "Make sure you visit all cities"
        assert len(set(path_idx)) == len(path_idx), "Make sure all cities are unique"

        path = self.create_path(path_idx)

        current = path[0]
        route_length = 0

        for stop in path[1:]:
            route_length += haversine(*current, *stop)
            current = stop

        return route_length

    def create_path(self, path_idx: np.ndarray) -> np.ndarray:
        """Convert an integer path to a matrix of lng, lat values, with Leiden pre- and appended"""

        return np.vstack(
            [
                LEIDEN,
                self.data.iloc[path_idx][["capital_lng", "capital_lat"]].values,
                LEIDEN,
            ]
        )

    def plot_route(self, path: np.ndarray, route_length: float = float("inf")) -> None:
        """Plot the route on the map of Europe, interactively.

        Parameters
        ----------
        path: np.ndarray
            The path of the TSP, can be either a list of indices, or a matrix
            of lng, lat values.
        route_length: float = inf (optional)
            The length of the route, to display in the plot, optional.

        """
        if self.plot:
            if len(path.shape) == 1:
                path = self.create_path(path)

            if self.line is not None:
                self.line[0].remove()
                self.legend.remove()

            self.line = self.ax.plot(
                path[:, 0],
                path[:, 1],
                color="green",
                linestyle="--",
                alpha=0.9,
                label=f"route length: {route_length:.2f} km",
            )
            self.legend = plt.legend()
            plt.draw()
            plt.pause(0.0001)


if __name__ == "__main__":
    # Below are a few examples of using the TSP class.
    # We can either use it as a context manager (i.e. with a with statement)
    # If we set plot=True, an interactive figure will appear
    #with TSP(plot=True) as tsp:
        # Sample a 10 random paths
        #for _ in range(10):
            # Sample a random path
            #random_path = np.random.permutation(tsp.dim)
            # Here we can use tsp as a function
            #path_length = tsp(random_path)
            # Display the path
            #tsp.plot_route(random_path, path_length)
            # Input something to continue
            #input()

    # Alternatively, you can use the object only for the TSP function,
    # and dont use the plot
    tsp = TSP(plot=False)
    # Get the path length
    random_path = range(tsp.dim)
    full_route = tsp.create_path(random_path) # this creates a full route with leiden in it 
    print(full_route)
    print(tsp.dim)
    print(len(full_route))
    #print(f"Random path length {route_length: .2f}km")
