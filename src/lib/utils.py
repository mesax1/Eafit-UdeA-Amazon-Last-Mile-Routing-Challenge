import numpy as np
import pandas as pd
import os, itertools
from csv import reader
from progressbar import progressbar as pbar
from joblib import Parallel, delayed
import sys, hashlib, subprocess
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from bokeh.io import output_file, output_notebook, show
from bokeh.models.mappers import ColorMapper, LinearColorMapper
from bokeh.palettes import Viridis5
from bokeh.plotting import figure, output_file, show
from bokeh.tile_providers import CARTODBPOSITRON, get_provider
from bokeh.models import ColumnDataSource, HoverTool
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import networkx as nx
import pickle
from sklearn.manifold import TSNE
import matplotlib.gridspec as gridspec


train_routes_basepath = "../data/model_build_outputs/df_routes_files"
test_routes_basepath = "../data/model_apply_outputs/df_routes_files_val"
traveltimes_basepath = "../data/model_build_outputs/travel_times_files"
grasp_routes_basepath = "../data/model_build_outputs/grasp_routes_prob"

class mParallel(Parallel):
    """
    substitutes joblib.Parallel with richer verbose progress information
    """
    def _print(self, msg, msg_args):
        if self.verbose > 10:
            fmsg = '[%s]: %s' % (self, msg % msg_args)
            sys.stdout.write('\r ' + fmsg)
            sys.stdout.flush()

def get_routes_ids(use_test_routes=False):
    base_path = train_routes_basepath
    if use_test_routes== True:
        base_path = test_routes_basepath
    files = [i.split(".")[0][3:] for i in os.listdir(base_path) if i.endswith(".csv")]
    return files

def latlon_to_meters(lat, lon):
    import math
    r_major = 6378137.000
    x = r_major * math.radians(lon)
    scale = x/lon
    y = 180.0/math.pi * math.log(math.tan(math.pi/4.0 + 
        lat * (math.pi/180.0)/2.0)) * scale
    return (y, x)

def self_distance_matrix(k):
    """
    returns a matrix with euclidean distances between all elements of k
    k must be a n x m array.
    returns and n x n matrix of distances
    """
    r = np.zeros((len(k), len(k)))
    for i in range(len(k)):
        for j in range(i+1,len(k)):
            r[i,j] = np.linalg.norm(k[i]-k[j])
            r[j,i] = r[i,j]
    return r

def distance_matrix(a,b=None):
    """
    returns a matrix with euclidean distances between all elements of a and all elements of b
    a must be a na x m array.
    b must be a nb x m array
    returns and na x nb matrix of distances
    """
    if b is None:
        return self_distance_matrix(a)
    
    r = np.ones((len(a), len(b)))
    for i in range(len(a)):
        for j in range(len(b)):
            r[i,j] = np.linalg.norm(a[i]-b[j])
    return r

def cluster_zone(zones, routes, zid, max_points_distance=5000):
    """
    breaks up a zone into smaller ones by clustering points in the convex hull
    """
    
    if not zid in zones.index:
        return None
    
    from sklearn.cluster import KMeans
    cv = zones.loc[zid].verticesm
    city = zones.loc[zid].city

    # obtain best nb of clusters for this zone
    n_clusters=1
    while True and n_clusters<len(cv):
        kp = KMeans(n_clusters=n_clusters).fit(cv).predict(cv)
        kp = {i: cv[kp==i] for i in np.unique(kp)}
        if np.max([distance_matrix(v).max() for v in kp.values()])<max_points_distance:
            break
        n_clusters += 1

    # points in the zone
    rz = routes[routes.city_zone==zid]

    # centroids for each cluster
    kc = {f'{zid}__{k}': np.mean(v, axis=0) for k,v in kp.items()}

    # new zones for route points on this zone
    nzids = np.r_[list(kc.keys())][distance_matrix(rz[['mlat', 'mlon']].values, 
                                                         np.r_[list(kc.values())]).argmin(axis=1)]

    # create new zones with split points
    nzones = {i: compute_zone(rz.iloc[nzids==i], city) for i in kc.keys()}

    return pd.DataFrame(nzones).T

def get_routes(force_recompute=False, use_test_routes=False):
    data_path = "../data/model_build_outputs/train_routes.hdf"
    if use_test_routes==True:
        data_path = "../data/model_apply_outputs/test_routes.hdf"
    if not os.path.isfile(data_path) or force_recompute:
        routes_ids = get_routes_ids(use_test_routes=use_test_routes)
        routes = [load_route(i, use_test_routes=use_test_routes, only_dropoffs=False) for i in pbar(routes_ids)]

        routes = pd.concat(routes)

        routes['city'] = [get_city(lat, lon) for lat,lon in routes[['latitude', 'longitude']].values]
        routes['city_zone'] = routes['city']+"::"+routes['zone_id']
        routes.to_hdf(data_path, "data")
    else:
        routes = pd.read_hdf(data_path)    

    return routes         

def get_aggregated_zones(force_recompute=False):
    if not os.path.isfile("../data/model_build_outputs/zones.hdf") or force_recompute:

        routes = get_routes(force_recompute=force_recompute)

        zone_ids = np.unique(routes.city_zone.dropna())
        print ("nb zones", len(zone_ids))

        zones = {}
        for zid in pbar(zone_ids):
            k = routes[routes.city_zone==zid]
            zones[zid] = compute_zone(k, city=zid.split("::")[0]) 

        zones = pd.DataFrame(zones).T
        zones.to_hdf("../data/model_build_outputs/zones.hdf", "data")
    else: 
        zones = pd.read_hdf("../data/model_build_outputs/zones.hdf")
        
    return zones

def get_routes_as_zones(force_recompute=False, use_test_routes=False):
    data_path = "../data/model_build_outputs/train_zroutes.hdf"
    if use_test_routes==True:
        data_path = "../data/model_apply_outputs/test_zroutes.hdf"
    if not os.path.isfile(data_path) or force_recompute:
        routes = get_routes(force_recompute=force_recompute, use_test_routes=use_test_routes)
        zones = []
        for route_id in pbar(np.unique(routes.route_fid)):
            route = routes[routes.route_fid==route_id]
            rzones = []
            for zid in np.unique(route.city_zone.fillna("None")):
                p = route[route.city_zone==zid]
                if len(p)==0:
                    continue
                zone = compute_zone(p, city=p.city.values[0])
                zone['order'] = np.mean(p.order)
                zone['order_min'] = np.min(p.order)
                zone['order_max'] = np.max(p.order)
                zone['order_clients'] = list(p.order.values)
                zone['route_fid'] = route_id
                zone['zone_id'] = zid
                rzones.append(zone)    
            # make sure zones are in order
            rzones = [rzones[j] for j in np.argsort([i['order'] for i in rzones])]
            zones += rzones
        zones = pd.DataFrame(zones)
        zones.to_hdf(data_path, "data")
    else:
        zones = pd.read_hdf(data_path)
    return zones


def compute_zone(dpoints, city=None):
    """
    dpoints: a dataframe with 'latitude', 'longitude' and 'route_fid'
             maybe belonging to different routes
    """
    k = dpoints
    points = k[['latitude', 'longitude']].values
    try:
        ch = ConvexHull(points)
        cv = ch.points[ch.vertices]
        area = ch.area
    except:
        cv = points
        area = 0
    tdist = travelling_distance([latlon_to_meters(*i) for i in cv])
    centroid = np.mean(points, axis=0)
    
    r = {
            "city": city, 
            "nb_points": len(points), 
            "nb_routes": len(np.unique(k.route_fid)),
            "nb_vertices": len(cv),
            "vertices": cv,
            "travelling_distance": tdist,
            "area": area,
            "centroid": centroid,
            "verticesm": np.r_[[latlon_to_meters(lat, lon) for lat,lon in cv]],
            "centroidm": latlon_to_meters(*centroid)
        }  
    
    return r

def plot_routes_as_zones(zroutes, fname=""):
    """
    plots routes as returned by get_routes_as_zones
    """
    from collections import defaultdict
    output_file(fname)

    latm_max, lonm_max = np.vstack(zroutes.verticesm.values).max(axis=0)
    latm_min, lonm_min = np.vstack(zroutes.verticesm.values).min(axis=0)

    tile_provider = get_provider(CARTODBPOSITRON)

    # range bounds supplied in web mercator coordinates
    plot = figure(x_range=(lonm_min-100000, lonm_max+100000), y_range=(latm_min-10000, latm_max+10000),
               x_axis_type="mercator", y_axis_type="mercator", plot_width=1800, plot_height=1000 )
    plot.add_tile(tile_provider)

    cr = defaultdict(list)
    
    for route_id in pbar(np.unique(zroutes.route_fid)):
        k = zroutes[zroutes.route_fid==route_id]
        centroids = np.r_[[np.r_[i] for i in k.centroidm.values]]
        cr['LON'].append(list(centroids[:,1]))
        cr['LAT'].append(list(centroids[:,0]))
        cr['route_id'].append(route_id)
                
    source = ColumnDataSource(cr)
    plot.multi_line(xs='LON', ys='LAT', 
                 line_width=1, line_color='red', line_alpha=0.6,
                 hover_line_alpha=1.0, hover_line_color="blue", hover_line_width=3,
                 source=source)        
    plot.add_tools(HoverTool(show_arrow=False, line_policy='next', tooltips=[
        ('id', '@route_id'),
    ]))
    show(plot)
    return 



def get_routes_with_reattempts():
    ra = pd.read_csv("../data/model_build_outputs/df_has_reattempts.csv")
    return [str(i) for i in ra[ra.reattempts!=0].route_id.values]

class NeighbourRoutes:
    
    def __init__(self, zroutes):
        self.zroutes = zroutes
        self.zntrs = np.r_[[np.r_[i] for i in self.zroutes.centroidm.values]]
        self.kt = KDTree(self.zntrs)
        
    def get_routes_with_reattempts(self):
        ra = pd.read_csv("../data/model_build_outputs/df_has_reattempts.csv")
        return [str(i) for i in ra[ra.reattempts!=0].route_id.values]

    def get_route(self, route_fid):        
        return self.zroutes[self.zroutes.route_fid == route_fid]
    
    def get_centroids(self, route_fid, ignore_station=True):
        """
        returns an nparray n x 2 with the ordered zone centroids of route route_fid 
        """
        zr = self.zroutes[self.zroutes.route_fid == route_fid]
        if ignore_station:
            zr = zr[~zr.zone_id.str.contains("Station")]

        zrc = np.r_[[np.r_[i] for i in zr.centroidm.values]]
        return zrc
    
    def get_points(self, p0_idx, p1_idx):
        p0 = self.zroutes.iloc[p0_idx]
        p1 = self.zroutes.iloc[p1_idx]

        z0 = np.r_[p0.centroidm]
        z1 = np.r_[p1.centroidm]
        
        assert p0.route_fid == p1.route_fid, "both points must be in the same route"

        return z0, z1, p0.route_fid

    def points_in_ball(self, p0_idx, p1_idx, r=500):
        """
        returns the points in the radius ball around p0 and p1 separately
        points that belong to the two balls are assigned to whichever of p0 or p1
        that is closest.
        """
        z0, z1, _ = self.get_points(p0_idx, p1_idx)

        # get the closest centroids from other routes to each ref point
        nn = [0,0]
        for i,z in enumerate([z0, z1]):
            nn[i] = self.kt.query_ball_point(z, r=r)

        # if there are closest centroids shared by both ref points
        # assign them to the closest ref point
        iset = np.r_[list(set(nn[0]).intersection(set(nn[1])))]

        if len(iset)>0:
            zi = self.zntrs[iset]

            closest = np.vstack([np.linalg.norm(zi - z0, axis=1), np.linalg.norm(zi - z1, axis=1)]).argmin(axis=0)
            snn0 = np.array(list(set(nn[0]).difference(iset)) + list(iset[closest==0]))
            snn1 = np.array(list(set(nn[1]).difference(iset)) + list(iset[closest==1]))

            return np.sort(snn0), np.sort(snn1)
        else:
            return None, None
        
    def get_neighbour_routes(self, p0_idx, p1_idx, r=500):
        """
        p0_idx, p1_idx: points in zroutes (must belong to the same route)
        r: max distance to consider neighbour points
        returns rc: a list of route_fids that share two points that are at 
                    most r units of distance from p0 and p1 respetively    
                snn0, snn1: list of points closest to p0 and p1 according to points_in_ball   
        """
        _, _, route_id = self.get_points(p0_idx, p1_idx)
        
        snn0, snn1 = self.points_in_ball(p0_idx, p1_idx, r=r)
        
        if snn0 is None or snn1 is None:
            rc = []
        else:
            # routes containing the closest centroids to each ref point
            r0 = np.unique(self.zroutes.iloc[snn0].route_fid)
            r1 = np.unique(self.zroutes.iloc[snn1].route_fid)

            # routes containing closest centroids to BOTH ref points
            rc = list(set(r0).intersection(set(r1)).difference(set([route_id])))

        return rc, snn0, snn1


    def get_neighbour_order(self, route_fid, p0_idx, p1_idx, snn0=None, snn1=None):
        """
        returns true if the closest point in route_fid to p0 is BEFORE
                        the closest point in route_fid to p1
                false otherwise
                
                n0, n1 the points in route_fid closest to p0 and p1
        """
        
        # get ref points
        z0, z1, _ = self.get_points(p0_idx, p1_idx)
        
        # get closest centroid in the route to each ref point
        if snn0 is None or snn1 is None:
            rz0 = self.zroutes[(self.zroutes.route_fid==route_fid)]
            rz1 = self.zroutes[(self.zroutes.route_fid==route_fid)]                             
        else:
            rz0 = self.zroutes[(self.zroutes.route_fid==route_fid)&(self.zroutes.index.isin(snn0))]
            rz1 = self.zroutes[(self.zroutes.route_fid==route_fid)&(self.zroutes.index.isin(snn1))]                             
            
        nz0 = np.r_[[i for i in rz0.centroidm.values]]
        nz1 = np.r_[[i for i in rz1.centroidm.values]]
            
        i0 = np.sum((nz0-z0)**2, axis=1).argmin()    
        i1 = np.sum((nz1-z1)**2, axis=1).argmin()

        # get order of points
        n0 = rz0.iloc[i0]
        n1 = rz1.iloc[i1]

        # record if n1 comes after n0
        return n1.order > n0.order, n0, n1

    def get_route_matrix(self, route_fid, use_pbar=False, reverse_pairs = False):
        """
        computes the probability matrix for this route. the matrix contains an entry for
        each pair of route points with the probability of being traversed in that
        order in the history (zroutes).
        """
        route = self.get_route(route_fid)

        pairs = []

        for i in range(len(route)):
            for j in range(i+1, len(route)):
                pairs.append((route.index[i], route.index[j]))
                
        probs = np.zeros((len(route), len(route)))*np.nan
        probs = pd.DataFrame(probs, columns = route.index, index = route.index)
        scores = probs.copy()
        nroutes = probs.copy()
        
        pb = pbar if use_pbar else (lambda x: x)
        for pair in pb(pairs):
            if reverse_pairs:
                pair = pair[::-1]
            rc, snn0, snn1 = self.get_neighbour_routes(*pair)        
            orders = [self.get_neighbour_order(i, *pair, snn0, snn1)[0] for i in rc]
            if len(orders)>0:
                probs.loc[pair]  = np.mean(orders)        
                scores.loc[pair] = self.get_neighbour_order(route_fid, *pair, snn0, snn1)[0]*1
                nroutes.loc[pair] = len(orders)

        valids = ~pd.isna(scores.values.flatten())
        acc = np.mean((scores.values.flatten()[valids]>=.5)==(probs.values.flatten()[valids]>=.5))
        pairs_covered = sum(valids)/len(pairs)

        station = route[route.zone_id.str.endswith("_Station")]
        if len(station)==1:
            station_id = station.iloc[0].name
            if reverse_pairs:
                probs.loc[[i for i in probs.columns if i!=station_id], station_id] = 0
            else:
                probs.loc[station_id, [i for i in probs.columns if i!=station_id]] = 1

        return probs, nroutes, acc, pairs_covered

    def sample_pair(self, route_fid, r=500, max_tries=100):
        """
        tries to sample 2 centroids on route_fid that are present in some
        neighbour route.
        return: - list of neighbour routes
                - the idxs of the two points

                None, None, None if could not find any
        """
        zr = self.zroutes[self.zroutes.route_fid == route_fid]
        # zrc = np.r_[[np.r_[i] for i in zr.centroidm.values]]
        c = 0
        while True:
            c+=1
            # select two ref points randomly
            s0 = zr.sample(1).iloc[0]
            s1 = zr.sample(1).iloc[0]
            p0_idx, p1_idx = s0.name, s1.name
            rc, snn0, snn1 = self.get_neighbour_routes(p0_idx, p1_idx, r=r)

            if len(rc)>0:
                break

            if c>max_tries:
                break

        if len(rc)==0:
            return None, None, None, None, None
        else:
            return rc, s0.name, s1.name, snn0, snn1

        
    def plot_paths(self, route_fid, paths, ignore_station=False, lw=10, alpha=.5):
        route = self.zroutes[self.zroutes.route_fid==route_fid]
        if ignore_station:
            route = route[~route.zone_id.str.contains("Station")]
            
        zr = self.get_centroids(route_fid, ignore_station=ignore_station)
        plt.plot(zr[:,1], zr[:,0], lw=lw, alpha=alpha)
        for _,i in route.iterrows():
            #if not i.name in probs.index:
            #    continue
            plt.text(*i.centroidm[::-1], str(i.name))    

        for p in paths:
            p = np.r_[[i for i in route.loc[p].centroidm]]
            plt.plot(p[:,1], p[:,0], color="black")
            plt.scatter(p[0,1], p[0,0], color="black")
            plt.scatter(p[-1,1], p[-1,0], color="black", marker="x")
        
        
    def plot_sample_points_and_neighbours(self, route_fid=None, p0_idx=None, p1_idx=None, r=500):
        colors = ['orange', 'steelblue']

        if route_fid is None:
            route_fid = np.random.choice(np.unique(self.zroutes.route_fid))
            rc, p0_idx, p1_idx, snn0, snn1 = self.sample_pair(route_fid, r=r)
        elif p0_idx is None or p1_idx is None:
            rc, p0_idx, p1_idx, snn0, snn1 = self.sample_pair(route_fid, r=r)
        else:
            rc, snn0, snn1 = self.get_neighbour_routes(p0_idx, p1_idx, r=r)

        zrc = self.get_centroids(route_fid)

        s0 = self.zroutes.iloc[p0_idx]
        s1 = self.zroutes.iloc[p1_idx]

        z0 = np.r_[s0.centroidm]
        z1 = np.r_[s1.centroidm]

        plt.figure(figsize=(10,10))
        plt.plot(zrc[:,1], zrc[:,0], color="black", label="ref", lw=3)
        for i,z in enumerate([z0, z1]):
            plt.scatter(*z[::-1], s=400, color=colors[i], alpha=.3, label="p%d"%i)

        zr1_gt_zr0 = []    
        for kk in range(len(rc)):
            rid = rc[kk]
            # select any nearest route with BOTH ref points
            order, nz0, nz1 = self.get_neighbour_order(rid, p0_idx, p1_idx, snn0, snn1)
            zr1_gt_zr0.append(order)

            ztc = self.get_centroids(rid)
            plt.plot(ztc[:,1], ztc[:,0], color="gray", label="neighbour" if kk==0 else None)
            for i, nz in enumerate([nz0, nz1]):
                plt.scatter(*nz.centroidm[::-1], s=100, marker="x", color=colors[i], 
                            label="assigned to p%d"%i if kk==0 else None)

        plt.grid()
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.axis("equal")
        print ("nb neighbour routes:  ", len(zr1_gt_zr0))
        print ("prob p1 after p0:     ", "%.4f"%np.mean(zr1_gt_zr0))
        print ("p1 is really after p0:", s1.order > s0.order)
        plt.show();
        
    def sample_seq_probabilities(self, n=1000, r=500):

        rx = []
        for _ in pbar(range(n)):
            route_fid = np.random.choice(np.unique(self.zroutes.route_fid))

            rc, p0_idx, p1_idx, snn0, snn1 = self.sample_pair(route_fid, r=r)
            if rc is None or len(rc)==0:
                continue
                
            s0 = self.zroutes.iloc[p0_idx]
            s1 = self.zroutes.iloc[p1_idx]
            zr1_gt_zr0 = []   
            for kk in range(len(rc)):
                rid = rc[kk]
                # select any nearest route with BOTH ref points
                order, _, _ = self.get_neighbour_order(rid, p0_idx, p1_idx, snn0, snn1)
                zr1_gt_zr0.append(order)

            rx.append([route_fid, p0_idx, p1_idx, len(rc), np.mean(zr1_gt_zr0), s1.order > s0.order])
            
        rx = pd.DataFrame(rx, columns=['route_fid', 'p0_idx', 'p1_idx', 'nb_routes_near', 'prob(p1|p0)', 'p1|p0'])
        return rx
                
    def plot_sample_seq_probabilities(self, n=1000, r=500):
        sno = self.sample_seq_probabilities(n=n, r=r)
        
        plt.figure(figsize=(15,3))
        plt.subplot(131)
        k = sno[sno["p1|p0"]]
        plt.scatter(k['prob(p1|p0)'], k['nb_routes_near'], color="steelblue", alpha=.1)
        plt.xlabel("prob p1|p0 in similar routes"); plt.title("p1|p0"); plt.ylabel("nb similar routes")
        plt.grid()

        plt.subplot(132)
        k = sno[~sno["p1|p0"]]
        plt.scatter(k['prob(p1|p0)'], k['nb_routes_near'], color="orange", alpha=.1)
        plt.xlabel("prob p1|p0 in similar routes"); plt.title("p0|p1"); plt.ylabel("nb similar routes")
        plt.grid()

        plt.subplot(133)
        plt.hist(sno[sno["p1|p0"]]['prob(p1|p0)'].values, density=True,  bins=30, alpha=.5, color="steelblue", label="p1|p0");
        plt.hist(sno[~sno["p1|p0"]]['prob(p1|p0)'].values, density=True, bins=30, alpha=.5, color="orange", label="p0|p1");
        plt.xlabel("prob p1|p0 in similar routes"); plt.title("histogram");
        plt.grid(); plt.legend();

        
        A_accuracy = sno[sno['prob(p1|p0)']>.5]['p1|p0'].mean()
        B_accuracy = 1-sno[sno['prob(p1|p0)']<.5]['p1|p0'].mean()
        p = len(sno)/n
        
        print (f'accuracy on seq+ {A_accuracy:.3f}')
        print (f'accuracy on seq- {B_accuracy:.3f}')
        print (f'pct similar routes found {p:.3f}')
        
        return sno


def get_city(lat, lon):
    if lon>-85:
        return "BOS"
    if lon>-95:
        return "CHI"
    if lat>45:
        return "SEA"
    if lon<-115:
        return "LAX"
    
    return "AUS"

def travelling_distance(points):
    """
    computes the distance of travelling through the points
    coming back to the beginning
    """
    if len(points)==0:
        return 0
    cv = [i for i in points]
    cv.append(cv[0])
    cv = np.r_[cv]
    return np.sum(np.sqrt(np.sum((cv[1:]-cv[:-1])**2, axis=1)))

def load_route(route_id, use_test_routes=False, drop_nozone_customers=False, only_dropoffs=True):
    base_path=train_routes_basepath
    if use_test_routes==True:
        base_path=test_routes_basepath
        
    k = pd.read_csv(base_path+"/df_"+str(route_id)+".csv")
    kcoords = np.r_[[latlon_to_meters(i.latitude,i.longitude) for _, i in k.iterrows()]]
    k['mlat'] = kcoords[:,0]
    k['mlon'] = kcoords[:,1]
    k['route_fid'] = route_id
    if only_dropoffs:
        k = k[k['type']=='Dropoff']
    k.loc[k.type=='Station', 'zone_id'] = "R"+k[k.type=='Station'].route_fid+"_Station"
    try:
        k.sort_values(by=['order'], inplace=True)
    except:
        k.sort_values(by=['zone_id'], inplace=True)
    k.customer_id.fillna("None", inplace=True)
    if drop_nozone_customers:
        k = k[(~pd.isna(k.zone_id))|(k.type=='Station')]
        k.order = np.arange(len(k))
    return k
    
def load_route_df(route_id, base_path=train_routes_basepath, drop_nozone_customers=False):
    na_values = [""]
    df = pd.read_csv(base_path+"/df_"+str(route_id)+".csv", na_values=na_values, keep_default_na=False)
    return df
    
def load_travel_times_df(route_id):
    na_values = [""]
    tt = pd.read_csv(f"{traveltimes_basepath}/travel_times_route_{str(route_id)}.csv", na_values=na_values, keep_default_na=False)
    return tt

def load_grasp_route(route_id, base_path=grasp_routes_basepath):
    with open(base_path+"/solution_route_"+str(route_id), 'r') as read_obj:
        csv_reader = reader(read_obj, skipinitialspace=True,)
        for row in csv_reader:
            grasp_route = row
    return grasp_route

def load_grasp_route_alpha_beta(route_id, base_path=grasp_routes_basepath, normalized="normalized",alpha=0, beta=1, grasp_type="normal_grasp"):
    with open(base_path+"/solution_route_"+str(route_id)+"_"+normalized+"_alpha_"+str(alpha)+"_beta_"+str(beta)+"_"+grasp_type, 'r') as read_obj:
        csv_reader = reader(read_obj, skipinitialspace=True,)
        for row in csv_reader:
            grasp_route = row
    return grasp_route
    
def load_travel_times(route_id):
    tt = pd.read_csv(f"{traveltimes_basepath}/travel_times_route_{str(route_id)}.csv")
    tt = tt.set_index('key').to_dict()
    return tt

def has_zone_transgression(route_df):
    k = route_df
    return pd.Series({zid: k[k['zone_id']==zid].order.diff().dropna().max()>1 for zid in np.unique(k.zone_id)}).dropna().max()

def score(actual,sub,cost_mat,g=1000):
    '''
    Scores individual routes.

    Parameters
    ----------
    actual : list
        Actual route.
    sub : list
        Submitted route.
    cost_mat : dict
        Cost matrix.
    g : int/float, optional
        ERP gap penalty. Irrelevant if large and len(actual)==len(sub). The
        default is 1000.

    Returns
    -------
    float
        Accuracy score from comparing sub to actual.

    '''
    norm_mat=normalize_matrix(cost_mat)
    return seq_dev(actual,sub)*erp_per_edit(actual,sub,norm_mat,g)

def erp_per_edit(actual,sub,matrix,g=1000):
    '''
    Outputs ERP of comparing sub to actual divided by the number of edits involved
    in the ERP. If there are 0 edits, returns 0 instead.

    Parameters
    ----------
    actual : list
        Actual route.
    sub : list
        Submitted route.
    matrix : dict
        Normalized cost matrix.
    g : int/float, optional
        ERP gap penalty. The default is 1000.

    Returns
    -------
    int/float
        ERP divided by number of ERP edits or 0 if there are 0 edits.

    '''
    total,count=erp_per_edit_helper(actual,sub,matrix,g)
    if count==0:
        return 0
    else:
        return total/count

def erp_per_edit_helper(actual,sub,matrix,g=1000,memo=None):
    '''
    Calculates ERP and counts number of edits in the process.

    Parameters
    ----------
    actual : list
        Actual route.
    sub : list
        Submitted route.
    matrix : dict
        Normalized cost matrix.
    g : int/float, optional
        Gap penalty. The default is 1000.
    memo : dict, optional
        For memoization. The default is None.

    Returns
    -------
    d : float
        ERP from comparing sub to actual.
    count : int
        Number of edits in ERP.

    '''
    if memo==None:
        memo={}
    actual_tuple=tuple(actual)
    sub_tuple=tuple(sub)
    if (actual_tuple,sub_tuple) in memo:
        d,count=memo[(actual_tuple,sub_tuple)]
        return d,count
    if len(sub)==0:
        d=gap_sum(actual,g)
        count=len(actual)
    elif len(actual)==0:
        d=gap_sum(sub,g)
        count=len(sub)
    else:
        head_actual=actual[0]
        head_sub=sub[0]
        rest_actual=actual[1:]
        rest_sub=sub[1:]
        score1,count1=erp_per_edit_helper(rest_actual,rest_sub,matrix,g,memo)
        score2,count2=erp_per_edit_helper(rest_actual,sub,matrix,g,memo)
        score3,count3=erp_per_edit_helper(actual,rest_sub,matrix,g,memo)
        option_1=score1+dist_erp(head_actual,head_sub,matrix,g)
        option_2=score2+dist_erp(head_actual,'gap',matrix,g)
        option_3=score3+dist_erp(head_sub,'gap',matrix,g)
        d=min(option_1,option_2,option_3)
        if d==option_1:
            if head_actual==head_sub:
                count=count1
            else:
                count=count1+1
        elif d==option_2:
            count=count2+1
        else:
            count=count3+1
    memo[(actual_tuple,sub_tuple)]=(d,count)
    return d,count

def normalize_matrix(mat):
    '''
    Normalizes cost matrix.

    Parameters
    ----------
    mat : dict
        Cost matrix.

    Returns
    -------
    new_mat : dict
        Normalized cost matrix.

    '''
    new_mat=mat.copy()
    time_list=[]
    for origin in mat:
        for destination in mat[origin]:
            time_list.append(mat[origin][destination])
    avg_time=np.mean(time_list)
    std_time=np.std(time_list)
    min_new_time=np.inf
    for origin in mat:
        for destination in mat[origin]:
            old_time=mat[origin][destination]
            new_time=(old_time-avg_time)/std_time
            if new_time<min_new_time:
                min_new_time=new_time
            new_mat[origin][destination]=new_time
    for origin in new_mat:
        for destination in new_mat[origin]:
            new_time=new_mat[origin][destination]
            shifted_time=new_time-min_new_time
            new_mat[origin][destination]=shifted_time
    return new_mat

def gap_sum(path,g):
    '''
    Calculates ERP between two sequences when at least one is empty.

    Parameters
    ----------
    path : list
        Sequence that is being compared to an empty sequence.
    g : int/float
        Gap penalty.

    Returns
    -------
    res : int/float
        ERP between path and an empty sequence.

    '''
    res=0
    for p in path:
        res+=g
    return res

def dist_erp(p_1,p_2,mat,g=1000):
    '''
    Finds cost between two points. Outputs g if either point is a gap.

    Parameters
    ----------
    p_1 : str
        ID of point.
    p_2 : str
        ID of other point.
    mat : dict
        Normalized cost matrix.
    g : int/float, optional
        Gap penalty. The default is 1000.

    Returns
    -------
    dist : int/float
        Cost of substituting one point for the other.

    '''
    if p_1=='gap' or p_2=='gap':
        dist=g
    else:
        dist=mat[p_1][p_2]
    return dist

def seq_dev(actual,sub):
    '''
    Calculates sequence deviation.

    Parameters
    ----------
    actual : list
        Actual route.
    sub : list
        Submitted route.

    Returns
    -------
    float
        Sequence deviation.

    '''
    actual=actual[1:]
    sub=sub[1:-1]
    comp_list=[]
    for i in sub:
        comp_list.append(actual.index(i))
        comp_sum=0
    for ind in range(1,len(comp_list)):
        comp_sum+=abs(comp_list[ind]-comp_list[ind-1])-1
    n=len(actual)
    return (2/(n*(n-1)))*comp_sum


class ProbabilityMatrix:
    
    def __init__(self, probs, zroutes, ignore_station=False):
        assert list(probs.index)==list(probs.columns), "probs matrix must have the same rows and columns"
        if ignore_station:
            non_stations = np.r_[[i for i in probs.index if not "Station" in zroutes.loc[i].zone_id]]
            self.probs = probs.loc[non_stations, non_stations]
        else:
            self.probs = probs
        
        start_nodes = (~pd.isna(self.probs)).astype(int).sum(axis=0)
        end_nodes   = (~pd.isna(self.probs)).astype(int).sum(axis=1)

        self.start_nodes = list(start_nodes[(start_nodes==0)&(end_nodes!=0)].index)
        self.end_nodes   = list(end_nodes[(end_nodes==0)&(start_nodes!=0)].index)        
        
        self.G = nx.DiGraph()
        self.G.add_nodes_from(self.probs.index)        
        
        for col, row in itertools.product(self.probs.columns, self.probs.index):
            if self.probs.loc[row][col]>0.5:
                self.G.add_edge(row, col)
            if self.probs.loc[row][col]<0.5:
                self.G.add_edge(col, row)
        
    def get_next_nodes(self, node):
        r = self.probs.loc[node]
        r = r[r>0.5]
        return dict(r) if len(r)>0 else {}

    def get_prev_nodes(self, node):
        r = self.probs[node]
        r = 1-r[r<0.5]
        return dict(r) if len(r)>0 else {}
    
    def add_next(self, paths=None):
        if paths is None:
            paths = [[i] for i in self.start_nodes]
            
        r = []
        for path in paths:
            last_node = path[-1]
            next_nodes = self.get_next_nodes(last_node)
            if len(next_nodes)>0:
                for next_node in next_nodes.keys():
                    r.append(path+[next_node])
            else:
                r.append(path)
                
        return r
    
    def add_prev(self, paths=None):
        if paths is None:
            paths = [[i] for i in self.end_nodes]
            
        r = []
        for path in paths:
            last_node = path[-1]
            next_nodes = self.get_prev_nodes(last_node)
            if len(next_nodes)>=0:
                for next_node in next_nodes.keys():
                    r.append([next_node]+path)
            else:
                r.append(path)
                
        return r    

    def get_directed_paths(self, direction="fwd"):
        assert direction in ['fwd', 'bckw'], "direction must be 'fwd', 'bckw'"
        r = self.add_next() if direction=='fwd' else self.add_prev()
        while True:
            next_r = self.add_next(r) if direction=='fwd' else self.add_prev()
            if r==next_r:
                break
            r = next_r
            
            # remove paths included in others
            rr = []
            for i in r:
                if np.sum([is_included(i,j) for j in r if j!=i]) == 0:
                    rr.append(i)
            r = rr
    
        r = [i for i in r if len(i)>1]
        
        # remove paths included in others
        rr = []
        for i in r:
            if np.sum([is_included(i,j) for j in r if j!=i]) == 0:
                rr.append(i)
        return rr

    
    def get_fwd_paths(self):
        r = self.add_next()
        while True:
            next_r = self.add_next(r)
            if r==next_r:
                break
            r = next_r
            
            # remove paths included in others
            rr = []
            for i in r:
                if np.sum([is_included(i,j) for j in r if j!=i]) == 0:
                    rr.append(i)
            r = rr
    
        r = [i for i in r if len(i)>1]
        
        # remove paths included in others
        rr = []
        for i in r:
            if np.sum([is_included(i,j) for j in r if j!=i]) == 0:
                rr.append(i)
        return rr
    
is_included = lambda a,b: np.alltrue([i in b for i in a])
    

class ZrouteField:
    
    def __init__(self, zroutes, max_distance=100):
        self.zroutes = zroutes
        self.max_distance = max_distance

    def set_kdistance(self, dwk=1e-3):
        self.dwk = dwk
        
        # weight function proportional to distance
        self.dweight = lambda x: np.exp(-x*self.dwk)        
        
        # max distance to consider for neighbours (distances with dweight < 1e-3  and thus discarded)
        drange = np.linspace(10,1000000,10000)
        self.nmaxd = drange[np.argwhere(self.dweight(drange)<1e-3)[0][0]]

        return self
        
    def compute_field(self, dwk=1e-3, use_pbar=True):

        self.set_kdistance(dwk)
        
        zroutes_augmented = []

        fbar = pbar if use_pbar else lambda x: x

        for route_fid in fbar(np.unique(self.zroutes.route_fid)):
            
            route = self.zroutes[self.zroutes.route_fid==route_fid]
            
            # route centroids
            centroids = np.r_[[list(i) for i in route.centroidm.values]][1:]
            
            # vectors between centroids
            vcentroids = centroids[1:,:] - centroids[:-1,]
            
            # distances between centroids
            vclens = np.sqrt((vcentroids**2).sum(axis=1))
            
            # unit vectors between centroids
            ucentroids = vcentroids/(vclens.reshape(-1,1)) 

            # create step vectors between centroids at most maxd meters from each other
            vacentroids = None
            for i in range(len(vcentroids)):
                if vclens[i]<self.max_distance:
                    continue

                step_size = self.max_distance / ((vclens[i]/self.max_distance)/(vclens[i]//self.max_distance))
                step_size = vclens[i] if step_size==0 else step_size
                
                acentroids = np.r_[[ucentroids[i]*j*step_size for j in np.arange(0,int(vclens[i]//step_size)+1)]] + centroids[i]
                k = np.hstack([acentroids, ucentroids[i].reshape(-1,1).repeat(len(acentroids), axis=1).T])[:-1]
                vacentroids = k if vacentroids is None else np.vstack([vacentroids, k])

            r = pd.DataFrame(vacentroids, columns = ['mlat', 'mlon', 'udelta_lat', 'udelta_lon'])
            r['route_fid'] = route_fid
            zroutes_augmented.append(r)
            
        self.azroutes = pd.concat(zroutes_augmented)

        self.kt = KDTree(self.azroutes[['mlat', 'mlon']].values)
        return self
    
    def plot_route(self, route_fids, nscale=1, **kwargs):
        for route_fid in route_fids:
            k = self.azroutes[self.azroutes.route_fid==route_fid]
            for _,i in k.iterrows():
                l = self.max_distance*.8
                plt.arrow(i.mlon, i.mlat, nscale*i.udelta_lon*l, nscale*i.udelta_lat*l, 
                          width=8, length_includes_head = True, **kwargs)

            route = self.zroutes[self.zroutes.route_fid==route_fid]
            centroids = np.r_[[list(i) for i in route.centroidm.values]][1:]
            plt.scatter(centroids[:,1], centroids[:,0], s=200, **kwargs)
        
        k = self.azroutes[self.azroutes.route_fid.isin(route_fids)]
        plt.ylim(np.min(k.mlat)-100, np.max(k.mlat)+100)
        plt.xlim(np.min(k.mlon)-100, np.max(k.mlon)+100)
    
    
    def estimate_heading(self, p, exclude_route_fid=None):
        """
        estimates heading at point p, weighing in observed headings around p

        returns:
           - rv: estimated heading, computed averaging all nv with wd
           - nc: observed points around p being considered
           - nv: observed headings at each nc
           - wd: weight for each nv
        """
        # nearest centroids
        nn = self.kt.query_radius(np.r_[p].reshape(1,-1), self.nmaxd)[0]


        # from other routes
        if exclude_route_fid is not None:
            nn = nn[(self.azroutes.iloc[nn].route_fid!=exclude_route_fid).values]

        # nearest routes
        nn_route_fids = np.unique(self.azroutes.iloc[nn].route_fid)

        # distances from each vector
        nc = self.azroutes.iloc[nn][['mlon', 'mlat']].values
        nv = self.azroutes.iloc[nn][['udelta_lon', 'udelta_lat']].values
        nd = np.sqrt(((nc-p[::-1])**2).sum(axis=1))

        if len(nd)>0:
            wd = self.dweight(nd)
            rv = (nv*(wd.reshape(-1,1))).mean(axis=0)    
        else:
            wd = 0
            rv = [0,0]
            nv = [0,0]

        return rv, nc, nv, wd

    def get_route_estimated_headings(self, route_fid, zroutes=None):
        """
        zroutes: use this zroutes to get the points instead of self.zroutes
                 still, the headings will be computed wrt self.zroutes
                 this is done like this to allow computing head estimations for
                 routes not used to compute the ZrouteField
        """
        rvs = []

        zroutes = self.zroutes if zroutes is None else zroutes

        route = zroutes[zroutes.route_fid==route_fid]

        for p in route.centroidm.values[1:]:
            rv, nc, nv, wd = self.estimate_heading(p, exclude_route_fid=route_fid)
            rvs.append(rv)

        rvs = np.r_[[[0,0]] + rvs]

        k = pd.DataFrame(rvs, columns=['estimated_heading_lon', 'estimated_heading_lat'])
        kc = np.r_[[list(i) for i in route.centroidm.values]]

        headings = kc[1:]-kc[:-1]
        headings = headings / np.sqrt((headings**2).sum(axis=1).reshape(-1,1))
        headings = np.vstack((headings,[0,0]))

        k.index = route.index
        k['city'] = route.city
        k['route_fid'] = route_fid
        k['zone_id'] = route.zone_id
        k['centroid_mlat'] = kc[:,0]
        k['centroid_mlon'] = kc[:,1]
        k['observed_heading_lat'] = headings[:,0]
        k['observed_heading_lon'] = headings[:,1]
        k = k.iloc[1:]
        return k

    def plot_heading_estimation(self, p, rv, nc=None, nv=None, wd=None, nscale=1, hscale=1):
        plt.scatter(*p[::-1], s=10, marker="o", color="red")
        if np.alltrue([i is not None for i in [rv, nv, wd]]):
            for i in range(len(nc)):

                plt.arrow(*nc[i], *(nscale*nv[i]*self.max_distance*wd[i]), 
                          color="blue", lw=1,alpha = 0.7,
                          length_includes_head=False,
                          head_width = self.max_distance*wd[i])

        plt.arrow(*p[::-1], *(hscale*self.max_distance*rv), 
                  lw=2, color="red", 
                  length_includes_head=True, head_width = 0.2*self.max_distance)   
        
    def plot_route_heading_estimations(self, route_fid, figsize=(18,18), nscale=1, hscale=1, min_rlen=0):
        plt.figure(figsize=figsize)
        self.plot_route([route_fid], nscale=nscale, color="black", alpha=.5)

        h = self.get_route_estimated_headings(route_fid)
        rvs = h[['estimated_heading_lon', 'estimated_heading_lat']].values

        ps = h[['centroid_mlat', 'centroid_mlon']].values
        plt.scatter(*ps[0][::-1], color="blue", marker="o", s=200)

        # normalize heading vectors for plotting
        rlens = np.sqrt((rvs**2).sum(axis=1))
        nrvs = rvs/(rlens.max())
        for p, nrv in zip(ps, nrvs):
            self.plot_heading_estimation(p, nrv, nc=None, nv=None, wd=None, nscale=nscale, hscale=hscale)
            
        plt.axis("equal")

    def get_estimated_headings(self, use_pbar=True, zroutes=None):
        """
        zroutes: use this zroutes instead of self.zroutes to get the zones to estimate the headings for
        return estimated headings for all routes
        """
        fbar = pbar if use_pbar else lambda x: x

        zroutes = self.zroutes if zroutes is None else zroutes

        k = pd.concat([self.get_route_estimated_headings(route_fid, zroutes)\
                    for route_fid in fbar(np.unique(zroutes.route_fid))])

        ## apply some normalization across all data
        klens = np.sqrt((k[['estimated_heading_lon', 'estimated_heading_lon']].values**2).sum(axis=1))
        k['estimated_heading_lon'] /= klens.mean()
        k['estimated_heading_lat'] /= klens.mean()

        # include cos distance between estimated direction and observed direction
        he = k[['estimated_heading_lon', 'estimated_heading_lat']].values
        he = normalize_vectors(he)
        oe = k[['observed_heading_lon', 'observed_heading_lat']].values
        oe = normalize_vectors(oe)
        k['cos_distance'] = (oe*he).sum(axis=1)
        
        return k

    def heading_estimations_cosdistance(self, h):
        # h: as returned by get_estimated_headings
        # computes cosdistance removing centroids with no estimation, or route endpoints

        he = h[['estimated_heading_lon', 'estimated_heading_lat']].values
        he = normalize_vectors(he)
        oe = h[['observed_heading_lon', 'observed_heading_lat']].values
        oe = normalize_vectors(oe)
        ok = ((he[:,0]!=0)|(he[:,1]!=0))&((oe[:,0]!=0)|(oe[:,1]!=0))
        cos_distance = (oe[ok]*he[ok]).sum(axis=1)
        
        p10, p50, p90 = np.percentile(cos_distance, [10,50,90])
        return {'cos_distance_mean': np.mean(cos_distance), 
                'cos_distance_p10': p10,
                'cos_distance_p50': p50,
                'cos_distance_p90': p90}

def normalize_vectors(v):
    vlens = np.sqrt((v**2).sum(axis=1)).reshape(-1,1)
    vlens[vlens==0] = 1
    return v/vlens

def plot_heading_estimations(h, route_fid, size_estimated=150, size_observed=200, figsize=(8,8),
                            plot_observed = True, plot_route=True):
    if figsize is not None:
        plt.figure(figsize=figsize)

    kr = h[h.route_fid==route_fid]

    if plot_route:
        plt.plot(kr.centroid_mlon, kr.centroid_mlat, color="steelblue", marker="o", alpha=0.5 if plot_observed else 0.7)
    else:
        plt.scatter(kr.centroid_mlon, kr.centroid_mlat, color="steelblue", marker="o", alpha=.7)

    plt.scatter(kr.centroid_mlon.values[0], kr.centroid_mlat.values[0], color="blue", s=80, label="start")

    cc = (kr.centroid_mlon, kr.centroid_mlat, 
          kr.estimated_heading_lon*size_estimated, kr.estimated_heading_lat*size_estimated)

    co = (kr.centroid_mlon, kr.centroid_mlat, 
          kr.observed_heading_lon*size_observed, kr.observed_heading_lat*size_observed)

    cc = np.r_[[i.values for i in cc]].T
    co = np.r_[[i.values for i in co]].T

    if plot_observed:
        for n,i in enumerate(co):
            plt.arrow(*i, color="steelblue", length_includes_head = True, lw=3, alpha=.5, head_width=60, label="observed" if n==0 else None)
            plt.scatter(*i[:2], color="steelblue", alpha=.1, label="observed" if n==0 else None)

    for n,i in enumerate(cc):
        plt.arrow(*i, color="red", length_includes_head = True, head_width=60)
        plt.scatter(*i[:2], color="red", label="estimated" if n==0 else None)

    plt.legend()
    plt.grid()
    plt.title(f"route {route_fid}")
    plt.axis("equal");
    

def get_heading_based_probmatrix(h, route_fid, how="combined"):

    """
    probability matrix:
    
    - compute cosine difference between centroid i,j between estimated vector at i, and direction from i to j (j-i)
    - compute euclidean difference between i,j
    - den: normalize all euclidean differences so that they are between 0.5 and 1
    - dcn: normalize all cosine differences so that they are between 0 and 1
    - compute probability: den*0.5 + (1-den)*dcn
    
    this way, probability:
    - tends to 0.5 when euclidean difference is high
    - tends to cosine distance when euclidean difference is low
    
    params:
        - h as returned by get_estimated_headings
        - route_fid: the route file id

    returns a dataframe: probability of column AFTER row
    
    """
    
    how_values  = ['combined', 'cosinedistance']
    assert how in how_values, f"only 'how' in {how_values} is allowed"

    hr = h[h.route_fid==route_fid]

    # euclidean distance matrix
    de = np.eye(len(hr))

    # cosine distance matrix
    dc = np.eye(len(hr))

    z = hr[['centroid_mlat', 'centroid_mlon']].values

    for i in range(len(hr)):
        for j in range(i+1, len(hr)):
            vij = z[j]-z[i]
            de[i,j] = de[j,i] = np.sqrt( ((vij)**2).sum())

    for i in range(len(hr)):
        ei = hr.iloc[i][['estimated_heading_lat', 'estimated_heading_lon']].values.astype(float)
        ei = normalize_vectors(ei.reshape(1,-1))[0]

        for j in range(len(hr)):
            vij = z[j]-z[i]
            vij = normalize_vectors((vij).reshape(1,-1))[0]
            dc[i,j] = np.sum(ei*vij)

    # normalize cosine distance as probability (cosd=-1 --> prob after = 0)
    dcn = dc*.5+.5
    den = de/np.max(de)
    
    # build probability matrix
    if how=='combined':
        p = den*0.5 + (1-den)*dcn
    elif how=='cosinedistance':
        p = dcn
    p = pd.DataFrame(p, columns = hr.zone_id, index=hr.zone_id)
    return p
    
def get_angle_based_probmatrix(h, route_fid):

    """
    probability matrix:
    
    - compute angular difference between centroid i,j between estimated vector at i, and direction from i to j (j-i)
    - compute probability: angle/180
    
    get the angle in radians between the two unit vectors. convert it to degrees, then divide by 180
    
    params:
        - h as returned by get_estimated_headings
        - route_fid: the route file id

    returns a dataframe: probability of column AFTER row
    
    """
    import math
    
    hr = h[h.route_fid==route_fid]


    # angle matrix
    dc = np.eye(len(hr))

    z = hr[['centroid_mlat', 'centroid_mlon']].values

    for i in range(len(hr)):
        ei = hr.iloc[i][['estimated_heading_lat', 'estimated_heading_lon']].values.astype(float)
        ei = normalize_vectors(ei.reshape(1,-1))[0]
        
        
        
        for j in range(len(hr)):
            vij = z[j]-z[i]
            vij = normalize_vectors((vij).reshape(1,-1))[0]
            
            unit_vector_1 = ei / np.linalg.norm(ei)
            unit_vector_2 = vij / np.linalg.norm(vij)
            dot_product = np.dot(unit_vector_1, unit_vector_2)
            angle = np.arccos(dot_product)  
            dc[i,j] = math.degrees(angle)/180

    # normalize cosine distance as probability (cosd=-1 --> prob after = 0)
    
    
    p = dc
    p = pd.DataFrame(p, columns = hr.zone_id, index=hr.zone_id)
    return p


class Route_Solutions:
    
    def __init__(self, route_id, sols, routes):
        self.route_id = route_id
        self.route = routes[routes.route_fid==route_id].sort_values(by='order')
        self.route_seq = [i if i!='None' else 'NA' for i in self.route.customer_id.values]
        self.sols = sols[sols.route_id==route_id]
        self.sols_seqs = [eval(i)[0][:-1] for i in self.sols.route_sequence.values]    
        assert np.alltrue(np.r_[[len(i) for i in self.sols_seqs]] == len(routes[routes.route_fid==route_id].customer_id.values))        

        self.sols_seqs = self.sols_seqs + [self.route_seq]
        
        self.best_cost_sol = self.sols.total_cost.argmin()
        
    def compute_eval_scores(self):
        
        seqs = self.sols_seqs    
        
        self.cost_mat = utils.load_travel_times_df(route_id).set_index('key').to_dict()
        
        sc = np.zeros((len(seqs), len(seqs)))
        for i in pbar(range(len(seqs))):
            for j in range(i+1, len(seqs)):
                #rd[i,j] = rd[j,i] = Levenshtein.distance(cseqs[i], cseqs[j])
                sc[i,j] = sc[j,i] = utils.score(self.sols_seqs[i], self.sols_seqs[j], self.cost_mat)
                
        self.scores = sc
        
        self.score_to_ref = [utils.score(list(self.route_seq), i, self.cost_mat) for i in seqs]        
        self.best_sol = self.sols.evaluation_score.argmin()
        
        self.tscores = TSNE(n_components=2, perplexity=5, learning_rate=200).fit_transform(self.scores)
        return self
        
    def compute_lev_distances(self):
        seqs = self.sols_seqs    

        lc = np.zeros((len(seqs), len(seqs)))
        for i in range(len(seqs)):
            for j in range(i+1, len(seqs)):
                #rd[i,j] = rd[j,i] = Levenshtein.distance(cseqs[i], cseqs[j])
                lc[i,j] = lc[j,i] = Levenshtein.distance("".join(self.sols_seqs[i]), "".join(self.sols_seqs[j]))

        self.lev_scores = lc
        self.lev_score_to_ref = [Levenshtein.distance("".join(list(self.route_seq)), "".join(i)) for i in seqs]
        self.lev_best_sol = np.r_[self.lev_score_to_ref[:-1]].argmin()
        self.lev_tscores = TSNE(n_components=2, perplexity=5, learning_rate=200).fit_transform(self.lev_scores)
        return self
        
    def plot(self, figsize=(20,3), show=True, what = 'eval_score'):
        assert what in ['eval_score', 'lev_score']
        
        fig = plt.figure(figsize=figsize)
        grid = gridspec.GridSpec(ncols=10, nrows=1, figure=fig)        
        
        if what == 'eval_score':
            tscores = self.tscores
            scores  = self.scores
            best_sol = self.best_sol
            score_to_ref = self.score_to_ref
        else:
            tscores = self.lev_tscores
            scores  = self.lev_scores
            best_sol = self.lev_best_sol
            score_to_ref = self.lev_score_to_ref
        
        plt.subplot(grid[0,:2])
        plt.scatter(tscores[:,0], tscores[:,1], alpha=.5)
        plt.scatter(tscores[-1,0], tscores[-1,1], color="green", marker="x", alpha=.5, s=200, label="ref")
        plt.scatter(tscores[best_sol,0], tscores[best_sol,1], color="red", marker="x", s=200)
        plt.scatter(tscores[self.best_cost_sol,0], tscores[self.best_cost_sol,1], color="blue", marker="x", s=200)
        plt.legend() # loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid(); plt.title("tSNE") 
        
        plt.subplot(grid[0,2:])
        for i in range(len(scores)):
            s = np.r_[[scores[i,j] for j in range(len(scores)) if i!=j]]
            if i==0:
                labels = ['score to perfect route', 'score to best solution', 'score to best cost', 'score to other solutions']
            else:
                labels = [None, None, None, None]
                
            baseline = 0# scores[i,self.best_cost_sol]
            plt.scatter([i]*len(s), s-baseline, color="gray", s=5, alpha=.5, label=labels[3], zorder=3)
            if i!=len(scores)-1:
                plt.scatter([i], score_to_ref[i]-baseline, color="green", s=20, label=labels[0], zorder=3)
            if i!=best_sol:
                plt.scatter([i], scores[i,best_sol]-baseline, color="red", s=20, label=labels[1], zorder=3)
            if i!=self.best_cost_sol:
                plt.scatter([i], scores[i,self.best_cost_sol]-baseline, color="blue", s=20, label=labels[2], zorder=3)

        """
        plt.scatter(best_sol, self.sols.iloc[best_sol].evaluation_score, color="red", marker="x", 
                    alpha=1, s=150, zorder=3, label="best solution");    
        plt.scatter(self.best_cost_sol, self.sols.iloc[self.best_cost_sol].evaluation_score, 
                    color="blue", marker="x", alpha=1, s=150, zorder=3, label="solution with best cost");    
        """
        
        plt.scatter(best_sol, scores[best_sol, -1]-baseline, color="red", marker="x", 
                    alpha=1, s=150, zorder=3, label="best solution");    
        plt.scatter(self.best_cost_sol, scores[self.best_cost_sol, -1]-baseline, 
                    color="blue", marker="x", alpha=1, s=150, zorder=3, label="solution with best cost");    
        
        plt.title("route "+self.route_id)
        plt.legend();
        plt.xticks(range(len(scores)), list(range(len(scores)-1))+['ref'])
        plt.grid(zorder=0)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        if show:
            plt.show()        
            
    def __getstate__(self):
        return self.__dict__
    def __setstate__(self, d):
        self.__dict__ = d

    @staticmethod
    def load_from_pickle(route_id, fdir="../data/tmp/cross_scores"):

        file = f"{fdir}/{route_id}.pkl" 
        with open(file, "br") as f:
            kk = pickle.load(f)    
        
        kk.compute_lev_distances()
        return kk
        