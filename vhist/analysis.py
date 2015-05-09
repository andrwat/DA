'''
Mouse dynamics classifier

What could be improved
    . don't mix data selection/filtering/split with analysis class
        better to have small composable operators
    . add type description? it's really hard to remember it all or figure it out again every time


# TODO (for code quality):
#   - convert stroke data to numpy like orig_features to be able to share more code?

'''
import os
import sys
import pdb
import csv
import copy
import math
import random
import logging
import platform

import pprint

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import scipy
import scipy.stats

from matplotlib.patches import Ellipse

import pandas as pd


# our dataset descriptors
dataset1_sessions = {'1423470172': 'Roman', '1423470700': 'Joe', '1423475777': 'Andrew', '1423471523': 'Roman'}
dataset2_sessions = {'1424664435': 'Roman', '1424767500': 'Andrew', '1424770997': 'Joe', '1428470230': 'Iv', '1428883647': 'Si'}
duncan_sessions = {'1428471956': 'Duncan', '1428487071': 'Duncan'}


def p_stats(dat):
    return pprint.pformat({k: len(dat[k]) for k in dat.keys()})

def dictmap(fn, m):
    return {k: fn(k, m[k]) for k in m.keys()}


def large_log(v):
    res = np.log(v)
    res[v < 0] = -np.log(np.abs(v[v < 0]))
    s = (v < 1) & (v > -1)
    res[s] = 0
    return res

def partition(data_all, PARTITIONS):
    cnts = {}
    for i in data_all.ids:
        c = cnts.setdefault(i, 0)
        c += 1
        cnts[i] = c


    subdiv_ids = []
    sess_start_ix = 0
    last_sess = 0
    last_sess_subdiv = -PARTITIONS
    for ix in range(len(data_all.ids)):
        sess = data_all.ids[ix]
        if sess != last_sess:
            last_sess = sess
            sess_start_ix = ix
            last_sess_subdiv += PARTITIONS

        total = cnts[sess]

        subdiv_ids.append( int(last_sess_subdiv + ((ix - sess_start_ix) / ((total / PARTITIONS) +1 ))) )

    data_all.subdiv_ids = np.array(subdiv_ids)
    return data_all


# TODO: subclass dict to have convenience methods?
class Dummy(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return 'Dummy(**' + str(self) + ')'



class DataSet(Dummy):
    '''
    Attributes:
        sessions: { session key (timestamp): person }
        ids: [session key]  for each row of nums
        nums: the raw data
    '''

    def cbyname(self, nm):
        'returns the column data'
        return self.nums[:,list(self.headers).index(nm)]

    def partition(self, count):
        parts = partition(self, count)

        arr = []
        for ix in range(count):
            res = copy.copy(self)
            sel = parts.subdiv_ids % count == ix
            res.nums = self.nums[sel,:]
            res.ids = self.ids[sel]
            res.slabel = self.slabel[sel]
            arr.append(res)

        return arr

    def user(self, sess_id):
        'Return a DataSet with rows for session id'
        if not isinstance(sess_id, list): sess_id = [sess_id]

        res = copy.copy(self)
        sel = np.array([(x in sess_id) for x in res.ids])
        return self.select(sel)

    def person(self, p):
        'Return rows for person p'
        return self.user(self.sk_by_person()[p])

    def sk_by_person(self):
        'session keys by person'
        return {p: [sk for sk in self.sessions.keys() if self.sessions[sk]==p] for p in self.sessions.values()}

    def random_partition(self, count):
        assignment = [random.choice(range(count)) for _ in self.ids]
        arr = []
        for ix in range(count):
            res = copy.copy(self)
            sel = np.array([(x == ix) for x in assignment])
            res.nums = self.nums[sel,:]
            res.ids = self.ids[sel]
            res.slabel = self.slabel[sel]
            arr.append(res)
        return arr

    def copy(self):
        return copy.copy(self)

    def select(self, barr):
        'barr - boolean array'
        res = self.copy()
        res.nums = res.nums[barr,:]
        res.ids = res.ids[barr]
        res.slabel = res.slabel[barr]
        return res

    def by_slabel(self, slabels):
        slabels = set(slabels)
        sel = np.array([x in slabels for x in self.slabel])
        return self.select(sel)

    def remove_std_outliers(self, sds):
        'Remove anything further than sds*SD from the mean'
        sel = np.array([True for _ in range(self.nums.shape[0])])
        for col in range(self.nums.shape[1]):
            D = self.nums[:,col]

            # stats different for each column (cos each column has different range / units)
            M = np.median(D)
            sd = np.std(D)

            x = np.abs(D - M) < (sds * sd)
            sel = sel & x

        # debug
        return self.select(sel)

    @classmethod
    def regularize_column(cls, vals, cut_low=False, cut_high=False, log=False):
        'Do some data massaging'
        # OK this is nice and all but if I want to do it for all features,
        # it's going to remove a lot of rows
        isnum = lambda x: isinstance(x, float) or (type(x) == int)   # avoid bool
        a = np.percentile(vals, cut_low if isnum(cut_low) else 2)
        b = np.percentile(vals, cut_high if isnum(cut_high) else 98)
        sel_a = np.ones_like(vals)
        sel_b = np.ones_like(vals)
        if cut_low:
            sel_a = vals > a

        if cut_high:
            sel_b = vals < b

        sel = np.logical_and(sel_a, sel_b)
        vals = vals[sel]

        if log:
            if log == True: log = large_log
            vals = log(vals)

        return vals, sel

    def massage_all(self, instr, o=False):
        instr = instr.fillna(value=' ')
        # TODO:
        #  - cut_low should ignore a long stretch of 0 because we are only interested in cutting off extremes
        headers = list(self.headers)

        cnt = 0
        removed_set = set()

        glob_sel = np.array([True for x in range(len(self.nums))])
        for col in instr.columns:
            col_instr = instr[col]
            col_ix = headers.index(col)


            def choice_cut(v):
                if v == ' ': return False
                elif v == 'x': return True
                elif v == 'o': return o
                else: return float(v)

            cut_low = choice_cut(col_instr.cut_low)
            cut_high = choice_cut(col_instr.cut_high)
            log_fn = {' ': None, 'x': large_log, 'log': np.log, 'o': large_log if o else None}[col_instr.log]

            _, sel = self.regularize_column(self.nums[:,col_ix], cut_low, cut_high, log=log_fn)

            removed_here = np.count_nonzero( np.logical_and(np.logical_not(sel), glob_sel) )
            here_set = np.where(np.logical_not(sel))[0]

            cnt += removed_here

            removed_set.update(here_set)
            glob_sel = np.logical_and(sel, glob_sel)

        # stats
        print 'removed', np.count_nonzero(np.logical_not(glob_sel))
        return self.select(glob_sel)


    def to_pandas(self):
        return pd.DataFrame(self.nums, index=self.ids, columns=self.headers)



def identify_outliers(d):
    result = d
    for col in range(d.shape[1]):
        max_ix = d[:,col].argmax()
        min_ix = d[:,col].argmin()

        print col, min_ix, max_ix
        #result = np.delete(result, [max_ix, min_ix], 0)

    return result

def remove_outliers(rich_data, iters=1):
    cnt = 0

    result = copy.copy(rich_data)

    def remove_fn(ix):
        result.nums = np.delete(result.nums, ix, 0)
        result.ids = np.delete(result.ids, ix, 0)
        result.slabel = np.delete(result.slabel, ix, 0)

    for i in range(iters):
        for col in range(result.nums.shape[1]):
            cold = result.nums[:,col]
            sd = cold.std()
            mean = cold.mean()

            if cold.max() > mean + sd:
                remove_fn(cold.argmax())
                cnt += 1

            if cold.min() < mean - sd:
                remove_fn(cold.argmin())
                cnt += 1

    print 'removed', cnt
    return result

def select_columns(rich_data, col_ixs):
    result = copy.copy(rich_data)
    result.nums = rich_data.nums[:,col_ixs]
    result.headers = rich_data.headers[col_ixs]
    return result

def inv_dict(dct):
    inv_map = {}
    for k, v in dct.iteritems():
        inv_map[v] = inv_map.get(v, [])
        inv_map[v].append(k)
    return inv_map

def gauss_normalize(vec):
    std = vec.std()
    if std == 0:
        std = 1
    return (vec - vec.mean()) / std

def normalize_columns(table):
    'Each column is one feature; change each column to standard normal distribution'
    return np.vstack([gauss_normalize(x) for x in table.T]).T

def xisq_pt(fi, gi):
    'The Xi^2 statistic between a point in two distributions'
    if (fi == 0) and (gi == 0): return 0
    v = ((fi - gi)/2)**2 / ((fi + gi)/2)
    #if v == float('nan'): print fi - gi, ' -> ', v

    # apparently need to do some bounds check otherwise sum(...) will bomb out on tiny numbers
    if v < 1.0e-200: return 0
    if v != v: return 0
    return v


def tight_subplots(fig):
    fig.subplots_adjust(0.01, 0.01, 0.99, 0.99, 0.01, 0.01)

def omin(arr):
    'Optimistic min, ignores None'
    return min(x for x in arr if x != None)

def omax(arr):
    'Optimistic max, ignores None'
    return max(x for x in arr if x != None)

ppl_colors = {'Roman': 'red', 'Joe': 'blue', 'Andrew': 'green', 'Iv': 'black', 'Si': 'brown'}

class HistUserData(object):
    def __init__(self, parent):
        self.X = []
        self.Y = []

        # one entry for each stroke, some info about that stroke
        # must be same order as self.samples to be able to plot things
        self.lengths = []
        self.maxv = []

        # corresponds to samples - which stroke label it was
        self.slabels = []

        # for each sampling point, the list of velocities at that point
        self.samples = []


def same_ranges(axes):
    autorange = [1000, -1000]

    for ax in axes:
        r = ax.get_ylim()
        autorange[0] = min(autorange[0], r[0])
        autorange[1] = max(autorange[1], r[1])

    for ax in axes:
        ax.set_ylim(autorange)

def log_colormap(cm_name):
    cm = mpl.cm.get_cmap(cm_name)
    cm = copy.copy(cm)
    cm.set_bad(cm(0.0))
    return cm


class DataSelectorBase(object):

    def split(self, is_random, ratio):
        if is_random:
            return self.split_random(ratio)
        else:
            return self.split_ratio(ratio)


class StrokeOperation(object):
    def __init__(self, strokes, sess_mapping, dat_type):
        self.strokes = strokes
        self.sess_mapping = sess_mapping
        self.dat = {k:dat_type() for k in self.sess_mapping.values()}

    def enum_ppl(self):
        return enumerate(self.dat.keys())


class SpeedsHistogram(object):
    # tried to use log scale for calculating Ymeans but it was incorrect
    # I don't see an easy math transformation to do what LogNorm does to the histogram
    # and don't see the benefit anyway so let's skip it
    def __init__(self):
        self.key = 'v'
        self.width = 100
        self.samples = 100
        self.height = 100
        self.yrange_scale = 1

    def create_dat(self, sess_mapping):
        self.dat = {k:HistUserData(self) for k in self.sess_mapping.values()}

    def enum_ppl(self):
        return enumerate(self.dat.keys())

    def interpolate(self, stroke):
        targ_dat = self.dat[self.sess_mapping[str(stroke['sess'])]]

        v = stroke[self.key]
        points = range(self.width)
        intrp = list(np.interp(np.linspace(0, len(v), self.width), range(len(v)), v))
        targ_dat.X += points
        targ_dat.Y += intrp
        targ_dat.lengths.append(len(v))
        targ_dat.maxv.append(np.array(v).max())
        targ_dat.slabels.append(stroke['slabel'])

        # sampling
        this_sample = np.interp(np.linspace(0, len(v), self.samples), range(len(v)), v)
        assert len(this_sample) == self.samples
        targ_dat.samples.append(intrp)

    def finish(self):
        for k in self.dat.keys():
            self.dat[k].samples = np.array(self.dat[k].samples)

        self.Range = [[0, self.width], [self.yrange_scale * min([min(a.Y) for a in self.dat.values()]),
                                        self.yrange_scale * max([max(a.Y) for a in self.dat.values()])]]

    def load_dct(self, dct, sess_mapping):
        """Loads an existing split into persons"""
        self.sess_mapping = sess_mapping
        self.create_dat(sess_mapping)
        for p in dct.keys():
            for s in dct[p]:
                self.interpolate(s)

        self.finish()


    def load_list(self, lst):
        for s in lst:
            self.interpolate(s)

        self.finish()

    def calc_means(self):
        'Means and other stats'
        for ix, p in self.enum_ppl():
            dat = self.dat[p].samples
            per_sample_fn = lambda agg: np.array([agg(dat[:,si]) for si in range(self.samples)])

            Ymean = per_sample_fn(np.mean)
            Ymed = per_sample_fn(np.median)
            Ysd_plus = per_sample_fn(lambda v: np.mean(v) + np.std(v)/2)
            Ysd_minus = per_sample_fn(lambda v: np.mean(v) - np.std(v)/2)
            Ysd = per_sample_fn(np.std)
            Ymax = per_sample_fn(np.max)
            Y25 = per_sample_fn(lambda v: np.percentile(v, 25))
            Y5 = per_sample_fn(lambda v: np.percentile(v, 5))

            X = np.linspace(0, self.width, self.samples)

            self.dat[p].smp_stats = Dummy(X=X, Ymean=Ymean, Ymed=Ymed,
                                          Ysd_plus=Ysd_plus, Ysd_minus=Ysd_minus,
                                          Ysd=Ysd, Y25=Y25, Y5=Y5)

            half = len(Ymed) / 2
            calc_asy = lambda x: (np.sum(x[:half]) - np.sum(x[half:])) / np.sum(x)
            self.dat[p].smp_stats.median_asymetry = calc_asy(Ymed)
            self.dat[p].smp_stats.mean_asymetry = calc_asy(Ymean)
            self.dat[p].smp_stats.y5_asymetry = calc_asy(Y5)

    def prepare_plot(self, figsize=None):
        fig, axes = plt.subplots(1, len(self.dat))
        self.fig = fig
        if figsize: self.fig.set_size_inches(*figsize)
        self.axes = axes if len(self.dat) > 1 else [axes]

    def plot_kde_hist(self):
        pts = np.linspace(0, self.Range[1][1], self.width)

        fig, ax = plt.subplots(1, len(self.dat))
        for ix, p in enumerate(self.dat.keys()):
            dat = self.dat[p]
            cols = []
            for s in range(dat.samples.shape[1]):
                kde = scipy.stats.gaussian_kde(dat.samples[:,s])
                vals = kde(pts)
                cols.append(vals)

            ax[ix].imshow(np.rot90(cols), extent=[0, self.width, 0, self.width*1.8])


    def plot_samples_lines(self):
        for ix, p in self.enum_ppl():
            dat = self.dat[p]
            ax = self.axes[ix]
            X = range(self.width)

            for i in dat.samples:
                ax.plot(X, i, alpha=0.15, c=(0.3, 0.7, 0.4))

            ax.set_xlim(self.Range[0])
            ax.set_ylim(self.Range[1])

    def plot_hist(self):
        Norm = mpl.colors.LogNorm(clip=False)

        for ix, p in enumerate(self.dat.keys()):
            dat = self.dat[p]
            Weights = np.zeros_like(dat.X) + 1.0 / len(dat.X)
            self.axes[ix].hist2d(dat.X, dat.Y, bins=self.width,
                                 weights=Weights,
                                 range=self.Range, norm=Norm, cmap=log_colormap('jet'))


    def plot_line(self):
        for ix, p in self.enum_ppl():
            st = self.dat[p].smp_stats
            self.axes[ix].plot(st.X, st.Ymean, 'violet', linewidth=3)
            self.axes[ix].plot(st.X, st.Ymed, 'white', linewidth=3)
            self.axes[ix].plot(st.X, st.Ysd_minus, 'black')
            self.axes[ix].plot(st.X, st.Ysd_plus, 'black')
            self.axes[ix].plot(st.X, st.Y25, '--', linewidth=2)
            self.axes[ix].plot(st.X, st.Y5, '-.', linewidth=2)

    def compare_sampling(self, my_p, other, other_p):
        """Compares the mean, sd of this sampled with other sampled"""
        assert self.samples == other.samples
        assert self.dat[my_p].smp_stats
        assert other.dat[other_p].smp_stats

        my_dat = self.dat[my_p]
        other_dat = other.dat[other_p]

        # TODO: find out what theoretical distribution it is actually and use a more specific metric
        d = np.average( np.abs(other_dat.smp_stats.Ymed - my_dat.smp_stats.Ymed) )
        # TODO: is this right?
        #d += np.average(other_dat.smp_stats.Ysd - my_dat.smp_stats.Ysd)
        return d


    def plot_corr(self, p, sample, orig_features):
        '''Plot the dependence between speeds at position *sample* for person *p*
        and some other variable to see if there's any correlation.'''
        sl = self.dat[p].slabels
        X = self.dat[p].samples[:,sample]
        #Y = orig_features.cbyname('s')
        Y = self.dat[p].maxv


        assert all(orig_features.slabel == sl)

        # TODO: try correlation with stroke length in pixels

        print 'correlation', scipy.stats.stats.pearsonr(X, Y)

        plt.scatter(X, X / Y, marker='x')


class SelectStrokes(StrokeOperation, DataSelectorBase):
    def __init__(self, strokes, sess_mapping):
        # param strokes as read from strokes.csv
        #   [ {header: value (list usually)} ]
        #
        # self.dat
        #   { person: [ {header: value} ] }
        StrokeOperation.__init__(self, strokes, sess_mapping, list)
        self.reset()

        # skip first N strokes because that's where the feature extractor is still learning about pause length
        # now using fixed pause length, no need to skip any more
        self.skip = 0

        # actually found that most (overwhelmingly!) strokes are shorter than 20
        # also found that I was doing the CombinedSetup() with these values which are not so
        # nice on the graph as the ones in the notebook
        self.min_len = 20
        # and also, most strokes are slower than this
        self.min_speed = 500
        self.max_len = 40
        self.max_speed = 4000
        # for the filter
        self.dragging = 0
        self.custom_filter = None
        # if an array or set, only take strokes with slabels from here
        self.slabels = None

    def reset(self):
        self.dat = { p: [] for p in self.sess_mapping.values() }
        self.filtered_cnt = { p: 0 for p in self.dat.keys() }

    def load_dct(self, inp):
        assert all(x == [] for x in self.dat.values())
        assert all(x == 0 for x in self.filtered_cnt.values())

        # loads into self and filters data already split by person
        for p in inp.keys():
            for s in inp[p]:
                if not self.filter(s):
                    self.filtered_cnt[p] += 1
                    continue

                self.dat[p].append(s)


    def disable_filter(self):
        # when you only want to use the 'divide_to_sessions' thing
        self.filter = lambda s: True

    def filter(self, stroke):
        v = stroke['v']
        if len(v) < self.min_len: return False
        if len(v) >= self.max_len: return False
        avg = sum(v) / len(v)
        if avg < self.min_speed: return False
        if avg >= self.max_speed: return False
        if stroke['dragging'] != self.dragging: return False
        if self.slabels != None:
            if not stroke['slabel'] in self.slabels: return False
        if self.custom_filter: return self.custom_filter(stroke)
        return True

    def divide_to_sessions(self):
        for s in self.strokes[self.skip:]:
            p = self.sess_mapping[str(s['sess'])]
            if not self.filter(s):
                self.filtered_cnt[p] += 1
                continue

            target_dat = self.dat[p]
            target_dat.append(s)

        self.verify()

    def verify(self):
        # also it's dumb because divide_to_sessions() is really straightforward
        assert len(self.strokes) == (sum(len(x) for x in self.dat.values()) + sum(self.filtered_cnt.values())) + self.skip
        for p in self.dat.keys():
            assert all(p == self.sess_mapping[str(x['sess'])] for x in self.dat[p])

    def split_ratio(self, ratio):
        # in this case, should filter it first
        first = {}
        second = {}
        for ix, p in self.enum_ppl():
            split = int(len(self.dat[p]) * ratio)

            first[p] = self.dat[p][:split]
            second[p] = self.dat[p][split:]


        return first, second

    def split_random(self, ratio):
        first = {}
        second = {}

        for ix, p in self.enum_ppl():
            cnt = int(len(self.dat[p]) * ratio)
            first[p] = random.sample(self.dat[p], cnt)
            second[p] = [x for x in self.dat[p] if x not in first[p]]

        return first, second


class EvaluateStrokeSpeedsMethod(object):
    def __init__(self, sel, csds):
        '''Take data from csds:`CombinedSetup` (because that handles the split, globally) and
        run both datasets through sel:`SelectStrokes` to filter the strokes.'''

        self.selector = sel
        self.selector.sess_mapping = csds.sessions
        self.csds = csds
        self.sess_mapping = self.csds.sessions

        # load testing
        sel.reset()
        sel.load_dct( csds.first['strokes'] )
        self.testing = sel.dat
        self.hs_testing = SpeedsHistogram()

        # load learning
        sel.reset()
        sel.load_dct( csds.second['strokes'] )
        self.learning = sel.dat
        self.hs_learning = SpeedsHistogram()

    def calc(self):
        self.hs_learning.load_dct(self.learning, self.sess_mapping)
        self.hs_learning.calc_means()

        self.hs_testing.load_dct(self.testing, self.sess_mapping)
        self.hs_testing.calc_means()

        self.cmp_results = {}
        for tst_p in self.hs_testing.dat.keys():
            for learn_p in self.hs_learning.dat.keys():
                d = self.hs_testing.compare_sampling(tst_p, self.hs_learning, learn_p)
                self.cmp_results.setdefault(tst_p, {})
                self.cmp_results[tst_p][learn_p] = d
                #print tst_p, learn_p, d

    def plot_comparisons(self):
        fig, axes = plt.subplots(1, len(self.learning))
        fig.set_size_inches(12, 10)
        axes = axes if len(self.learning) > 1 else [axes]

        for ix, p in enumerate(self.learning.keys()):
            axes[ix].plot(self.hs_learning.dat[p].smp_stats.Ymed, c='blue', linewidth=3)
            axes[ix].plot(self.hs_testing.dat[p].smp_stats.Ymed, c='red', linewidth=3)


            if False:
                le_color = (0.3, 0.3, 1)
                te_color = (1, 0.3, 0.3)
                axes[ix].plot(self.hs_learning.dat[p].smp_stats.Ysd_minus, c=le_color)
                axes[ix].plot(self.hs_testing.dat[p].smp_stats.Ysd_minus, c=te_color)
                axes[ix].plot(self.hs_learning.dat[p].smp_stats.Ysd_plus, c=le_color)
                axes[ix].plot(self.hs_testing.dat[p].smp_stats.Ysd_plus, c=te_color)


            for o in self.learning.keys():
                if o == p: continue
                axes[ix].plot(self.hs_learning.dat[o].smp_stats.Ymed, c='gray')

            #axes[ix].plot(self.hs_learning.dat[p].smp_stats.Ysd, c='blue', marker='x')
            #axes[ix].plot(self.hs_testing.dat[p].smp_stats.Ysd, c='red', marker='x')

        same_ranges(axes)

    def plot_comparison_hists(self):
        # do not call prepare_plot and instead do it from here (hacky hacky warning)
        fig, axes = plt.subplots(2, len(self.learning), squeeze=False)
        fig.set_size_inches(12, 10)
        self.hs_learning.fig = fig
        self.hs_testing.fig = fig

        self.hs_learning.axes = axes[1]
        self.hs_testing.axes = axes[0]

        self.hs_learning.plot_hist()
        self.hs_learning.plot_line()

        self.hs_testing.plot_hist()
        self.hs_testing.plot_line()


class FeaturesPCA(object):
    # Doesn't work imho because features have wildly different ranges and this puts them all in the same space
    def __init__(self):
        self.columns = [
            'vxmin', 'vxmax', 'vxrange', 'vxavg', 'vxsd',
            'vymin', 'vymax', 'vyrange', 'vyavg', 'vysd',
            'vmin', 'vmax', 'vrange', 'vavg', 'vsd'
        ]

    def selftest(self):
        sel1 = [False for x in self.data1.nums]
        self.data1 = DataSet()
        self.data1.headers = self.datasrc.headers
        self.data1.sessions = self.datasrc.sessions
        self.data1.ids = np.array(['1424664435', '1424664435', '1424664435', '1424767500'])
        self.data2.slabel = np.array([6, 7, 8, 9])
        self.data1.nums = np.array([[1.0 / random.randint(1, 100) for _ in self.h_ix] for _ in range(4)])

        self.data2 = DataSet()
        self.data2.headers = self.datasrc.headers
        self.data2.sessions = self.datasrc.sessions
        self.data2.ids = np.array(['1424664435', '1424767500', '1424767500', '1424770997'])
        self.data2.slabel = np.array([1, 2, 3, 4])
        self.data2.nums = np.array([[1.0 / random.randint(1, 100) for _ in self.h_ix] for _ in range(4)])

    def load(self, datasrc):
        self.datasrc = datasrc
        self.h_ix = [list(datasrc.headers).index(x) for x in self.columns]

        '''
        'vdmax', 'vdrange', 'vdavg', 'vdsd',
        'wmax', 'wrange', 'wavg', 'straight', 'wsd']]
        '''

        self.data1 = remove_outliers(select_columns(datasrc, self.h_ix), 2)


    def pca_first(self):
        self.pca = mlab.PCA(self.data1.nums)
        #self.pca2 = self.pca.project(self.data2.nums)


    def plot_scatter(self, pca_data, datasrc):
        from mpl_toolkits.mplot3d import Axes3D

        D3 = False

        fig = plt.figure()
        ax = fig.add_subplot(111 , projection=('3d' if D3 else None))
        X = pca_data[:,0]
        Y = pca_data[:,1]
        Z = pca_data[:,2] if D3 else None

        ax.scatter(X, Y,  c=[ppl_colors[datasrc.sessions[x]] for x in datasrc.ids], marker='x')

    def plot_hist(self, pca_data, datasrc):
        X = pca_data[:,0]
        Y = pca_data[:,1]

        Bins = 60
        sess_filter = lambda s: datasrc.ids == s

        fig, axes = plt.subplots(1, len(datasrc.sessions.keys()))

        for ix, s in enumerate(datasrc.sessions.keys()):
            sel = sess_filter(s)
            axes[ix].hist2d(X[sel], Y[sel], bins=Bins, norm=mpl.colors.LogNorm(),
                            range=self.Range, cmap=log_colormap('jet'))


    def plot_all(self):
        ax_range = lambda d,ix: [d[:,ix].min(), d[:,ix].max()]
        r1 = [ax_range(self.pca.Y, 0), ax_range(self.pca.Y, 1)]
        r2 = [ax_range(self.pca2, 0), ax_range(self.pca2, 1)]
        self.Range = [[min(r1[0][0], r2[0][0]), max(r1[0][1], r2[0][1])], [min(r1[1][0], r2[1][0]), max(r1[1][1], r2[1][1])]]
        self.Range[1][0] = -10
        self.plot_hist(self.pca.Y, self.data1)
        self.plot_hist(self.pca2, self.data2)

        #self.plot_scatter(self.pca.Y, self.data1)
        #self.plot_scatter(self.pca2, self.data2)


class RangeGen(object):
    def __init__(self, data, it=None, proj=None):
        self.data = data
        if it: self.it = it
        else: self.it = self.it_dict

        if proj: self.proj = proj
        else: self.proj = lambda x: x

    def it_dict(self, k):
        return self.data[k]

    def it_list(self, k):
        return k

    def bound(self, direction):
        return direction(direction(self.proj(self.it(k))) for k in self.data)

    def rng(self):
        return np.array([self.bound(min), self.bound(max)])


class OrigFeatureDirectAnalysis(object):
    def __init__(self, dataset):
        # param dataset is DataSet from features.csv
        self.dataset = dataset

        # split and join by person
        #   { person: DataSet() }
        psk = dataset.sk_by_person()
        self.dat = {p: dataset.user(psk[p]) for p in psk}
        self.result = {k: Dummy() for k in self.dat.keys()}
        self.rgitfn = lambda agg, col: agg([agg(getattr(self.result[k], col)) for k in self.dat.keys()])
        self.rgfn = lambda col: [self.rgitfn(min, col), self.rgitfn(max, col)]

        assert sum([x.nums.shape[0] for x in self.dat.values()]) == self.dataset.nums.shape[0]

    def velocity_analysis(self):
        for k in self.dat.keys():
            x = self.result[k].vxavg = self.dat[k].cbyname('vxavg')
            y = self.result[k].vyavg = self.dat[k].cbyname('vyavg')

            self.result[k].vx_mean = x.mean()
            self.result[k].vy_mean = y.mean()
            self.result[k].vx_sd = x.std()
            self.result[k].vy_sd = y.std()
            self.result[k].vx_kurt = scipy.stats.kurtosis(x)
            self.result[k].vy_kurt = scipy.stats.kurtosis(y)

            r = self.result[k]
            r.vxrange = self.dat[k].cbyname('vxrange')
            r.vyrange = self.dat[k].cbyname('vyrange')
            r.vxr_sd = r.vxrange.std()
            r.vyr_sd = r.vyrange.std()
            r.vxr_mean = r.vxrange.mean()
            r.vyr_mean = r.vyrange.mean()
            r.vxr_q75 = np.percentile(r.vxrange, 75)
            r.vyr_q75 = np.percentile(r.vyrange, 75)

    def plot_velocity(self):
        fig, axes = plt.subplots(1, len(self.dat))
        x_range = self.rgfn('vxavg')
        y_range = self.rgfn('vxavg')

        for ix, k in enumerate(self.dat.keys()):
            r = self.result[k]

            X = r.vxavg
            Y = r.vyavg

            axes[ix].set_xlim(x_range)
            axes[ix].set_ylim(y_range)

            axes[ix].scatter(X, Y, marker='x')
            axes[ix].scatter([self.result[k].vx_mean], [self.result[k].vy_mean],
                             marker='x', c='red')

            el = Ellipse(xy=[r.vx_mean, r.vy_mean],
                         width=r.vx_sd * 2, height=r.vy_sd * 2)
            el.set_facecolor((0.2, 0.9, 0.7))
            el.set_alpha(0.3)
            axes[ix].add_artist(el)

    def plot_hist_inone(self):
        'Plots the histogram of some feature to compare difference in distribution'
        # conclusion is that these features are not very interesting
        fig, ax = plt.subplots(1)
        col = 'wsd'
        Range = RangeGen(self.dat, proj=lambda p: p.cbyname(col)).rng()
        for ix, k in enumerate(self.dat.keys()):
            V = self.dat[k].cbyname(col)
            Bins = 40
            n, bins, patches = ax.hist(V, bins=Bins, range=Range,
                    weights=np.zeros_like(V) + 1. / V.size,
                    alpha=0.1, color=ppl_colors[k])

            # plots the PDF line, more useful for comparison
            ax.plot(bins[1:], n)

    def clean(self):
        for k in self.result.keys():
            del self.result[k].vxrange
            del self.result[k].vyrange
            del self.result[k].vxavg
            del self.result[k].vyavg


class PerSessionAnalysis(object):
    def __init__(self, dataset):
        # param dataset comes from SplitSessionData
        #   { person: [ {hdr: value} ] }
        self.dataset = dataset

        #   { person: Dummy{median, mean, sd} }
        self.result = {k: Dummy() for k in dataset.keys()}

    def run(self):
        for person in self.result.keys():
            T = np.array(self.dataset[person]['ttc'])
            H = np.array(self.dataset[person]['hold'])

            self.run_one(person, T, 'ttc')
            self.run_one(person, H, 'hold')

    def run_one(self, sk, V, name):
        l1 = len(V)

        # remove outliers
        V = V[V < (V.mean() + V.std() * 3)]
        l2 = len(V)

        assert float(l2 - l1)/l1 < 0.1

        r = self.result[sk]
        robj = Dummy()
        robj.median = np.median(V)
        robj.mean = np.mean(V)
        robj.sd = np.std(V)
        setattr(r, name, robj)


class SplitSessionData(DataSelectorBase):
    def __init__(self, dataset, sessions):
        # param dataset as it comes from the CSV
        #   [ {hdr: value (usually list)} ]
        #
        # self.dataset divided into dicts by person
        #   { person: {hdr: value (usually list)} }
        self.dataset = {k:{} for k in sessions.values()}
        self.sessions = sessions

        # split into dicts
        for i in dataset:
            person = self.sessions[str(i['sess'])]
            b = dict(i)
            del b['sess']
            self.append( self.dataset[person], b )

        self.verify(dataset)

    def append(self, a, b):
        'Adds items from lists in *b* to lists in *a*'
        for k in b.keys():
            if not (k in a): a[k] = []
            a[k] = a[k] + b[k]

    def verify(self, orig):
        'Verifies self consistency using a simple sum of counts of all values'
        orig_cnt = 0
        for it in orig:
            for l in it.values():
                if isinstance(l, list):
                    orig_cnt += len(l)

        new_cnt = 0
        for pv in self.dataset.values():
            for l in pv.values():
                if isinstance(l, list):
                    new_cnt += len(l)

        assert orig_cnt == new_cnt

    @classmethod
    def test(cls):
        sess = {'a': 'a', 'b': 'b'}
        dat = [{'sess': 'a', 'x': [1, 2]}, {'sess': 'b', 'x': [3, 3]}]
        sel = SplitSessionData(dat, sess)
        assert sel.dataset['a']['x'] == [1, 2]

    def split_ratio(self, ratio):
        first = {}
        second = {}

        for k in self.dataset.keys():
            first[k] = {}
            second[k] = {}
            for ak in self.dataset[k].keys():
                line = self.dataset[k][ak]
                cnt = int(len(line) * ratio)

                first[k][ak] = line[:cnt]
                second[k][ak] = line[cnt:]

        return first, second


    def split_random(self, ratio):
        first = {}
        second = {}

        for k in self.dataset.keys():
            first[k] = {}
            second[k] = {}
            for ak in self.dataset[k].keys():
                line = self.dataset[k][ak]
                cnt = int(len(line) * ratio)

                first[k][ak] = random.sample(line, cnt)
                second[k][ak] = [x for x in line if x not in first[k][ak]]

        return first, second


class VHistSegment(Dummy):
    def __init__(self, strokes, sessions, params):
        self.params = params

        sel = SelectStrokes([], sessions)
        sel.min_len = params['min_len']
        sel.max_len = params['max_len']
        sel.min_speed = 100
        sel.max_speed = 10000
        sel.dragging = 0
        sel.slabels = params['slabels']
        sel.load_dct(strokes)
        self.sel = sel

        #print p_stats(sel.dat)

        self.vhist = SpeedsHistogram()
        self.vhist.load_dct(sel.dat, sel.sess_mapping)
        self.vhist.calc_means()


class CombinedSetup(object):
    def __init__(self):
        self.first = {}
        self.second = {}
        self.all_data = {}

        # config on how to split data for testing
        self.ratio = 0.5
        self.random = False

    @classmethod
    def load_bymachine(cls, ds=2, massage=False):
        paths = {'2Topinka': r'c:\Transfer\cmss\dataset{}',
                 'default': 'data/dataset{}'}

        csds = cls()
        csds.load_folder(paths.get(platform.node(), paths['default']).format(ds), cls.sess_by_ix(ds))
        if massage: csds.massage(csds.feature_massage)
        return csds


    @classmethod
    def sess_by_ix(cls, ds):
        return [0, dataset1_sessions, dataset2_sessions, duncan_sessions][ds]


    def load_folder(self, path, sess_map):
        self.sessions = sess_map
        self.load_orig_features(os.path.join(path, 'features.csv'))
        self.load_strokes(os.path.join(path, 'strokes.csv'))
        self.load_sess_data(os.path.join(path, 'sessions.csv'))
        self.load_hlact(os.path.join(path, 'hlact.csv'))

        massage_path = os.path.join(path, 'features massage.csv')
        if os.path.exists(massage_path):
            self.feature_massage = pd.read_csv(massage_path, sep=';', index_col=0)[:3]
        else:
            print 'did not load features massage.csv'

    def load_orig_features(self, fname):
        def convert_line(ln):
            # strip '' at the end
            # everything except session is a float
            def exc_float(x):
                try:
                    return float(x)
                except:
                    return 0.0
            return [ln[0], ln[1]] + map(exc_float, ln[2:-1])

        with open(fname, 'rb') as inf:
            rdr = csv.reader(inf, delimiter=';')
            all_data = list(rdr)
            data = np.array([convert_line(x) for x in all_data[1:]])
            headers = all_data[0][:-1]

        data_all = DataSet()
        data_all.headers_ix = [x for x in range(1, data.shape[1]) if not (('count' in headers[x]) or ('slabel' == headers[x]))]
        data_all.nums = np.asarray(data[:,tuple(data_all.headers_ix)], float)
        data_all.ids = data[:,0]
        data_all.slabel = np.asarray(data[:,1], int)
        data_all.headers = np.array(headers)[data_all.headers_ix]
        data_all.sessions = self.sessions
        self.all_data['orig'] = data_all

    def load_3d_csv(self, fname):
        sta = []
        with open(fname, 'rb') as inf:
            rdr = csv.reader(inf, delimiter=';')

            for ix, row in enumerate(rdr):
                try:
                    if ix == 0: header = row
                    else: sta.append({header[y]: eval(row[y]) for y in range(len(row)) if row[y] != ''})
                except SyntaxError as e:
                    logging.error('Problem reading {} at line {}'.format(fname, ix))
                    raise e

        return sta

    def load_strokes(self, fname):
        self.all_data['strokes'] = self.load_3d_csv(fname)

    def load_sess_data(self, fname):
        data = self.load_3d_csv(fname)
        if data[-1]['sess'] == 0:
            del data[-1]
        self.all_data['sess_data'] = data

    def load_hlact(self, fname):
        self.all_data['hlact'] = self.load_3d_csv(fname)
        self.calc_PaC_ttc()

    def slabels_by(self, pred=None):
        if pred == None: pred = lambda _: True
        return [x['parts'][0][2] for x in self.all_data['hlact'] if pred(x)]

    def calc_PaC_ttc(self):
        '''Calculate time-to-click values from PaC actions, may be different from
        sess_data.ttc'''
        dat = {}
        for i in self.all_data['hlact']:
            ttc = i['parts'][1][1] - i['parts'][0][1]
            sess = i['sess']
            dat.setdefault(sess, [])
            dat[sess].append(ttc)

        self.all_data['PaC_ttc'] = dat

    def partition(self):
        '''Divides all data into testing and learning halves'''
        # strokes
        sel = SelectStrokes(self.all_data['strokes'], self.sessions)
        sel.disable_filter()
        sel.divide_to_sessions()
        self.sel = sel
        first, second = sel.split(self.random, self.ratio)

        self.first['strokes'] = first
        self.second['strokes'] = second

        # orig features
        if self.random:
            p = self.all_data['orig'].random_partition(2)
        else:
            p = self.all_data['orig'].partition(2)
        self.first['orig'] = p[0]
        self.second['orig'] = p[1]

        sks = self.first['orig'].sk_by_person()
        self.first['orsplit'] = {p: self.first['orig'].user(sks[p]) for p in sks}
        self.second['orsplit'] = {p: self.second['orig'].user(sks[p]) for p in sks}

        # sessions
        sel = SplitSessionData(self.all_data['sess_data'], self.sessions)
        first, second = sel.split(self.random, self.ratio)
        self.first['sess_data'] = first
        self.second['sess_data'] = second

        # PaC_ttc
        self.split_PaC_ttc()

    def split_PaC_ttc(self):
        fst = self.first['PaC_ttc'] = {}
        snd = self.second['PaC_ttc'] = {}

        for k in self.all_data['PaC_ttc'].keys():
            vals = self.all_data['PaC_ttc'][k]

            if not self.random:
                cnt = int(len(vals) * self.ratio)
                fst[k] = vals[:cnt]
                snd[k] = vals[cnt:]
            else:
                print 'random split of PaC ttc not implemented'
                #raise NotImplementedError('tough luck')



    def calc(self, data):
        '''Calculates all features using different analysis methods'''
        vhist = SpeedsHistogram()
        vhist.load_dct(data['strokes'], self.sessions)
        vhist.calc_means()

        psa = PerSessionAnalysis(data['sess_data'])
        psa.run()

        ofda = OrigFeatureDirectAnalysis(data['orig'])
        ofda.velocity_analysis()
        ofda.clean()

        return {'vhist': {k: vhist.dat[k].smp_stats for k in vhist.dat.keys()},
                'psa': psa.result,
                'ofda': ofda.result
                }


    @classmethod
    def dist(cls, a, b):
        compare = {}
        for ak in a['psa'].keys():
            compare[ak] = {}
            for bk in b['psa'].keys():
                # removed vhist stuff

                # PSA
                psa_hold_d = np.abs(a['psa'][ak].hold.median - b['psa'][bk].hold.median)
                psa_hold_d += np.abs(a['psa'][ak].hold.mean - b['psa'][bk].hold.mean)
                psa_holdsd_d = np.abs(a['psa'][ak].hold.sd - b['psa'][bk].hold.sd)

                psa_ttc_d = np.abs(a['psa'][ak].ttc.median - b['psa'][bk].ttc.median)
                psa_ttc_d += np.abs(a['psa'][ak].ttc.mean - b['psa'][bk].ttc.mean)
                psa_ttc_sd_d = np.abs(a['psa'][ak].ttc.sd - b['psa'][bk].ttc.sd)

                # OFDA
                ao = a['ofda'][ak]
                bo = b['ofda'][bk]
                of_mean_d = np.abs(ao.vx_mean - bo.vx_mean)
                of_mean_d += np.abs(ao.vy_mean - bo.vy_mean)

                of_sd_d = np.abs(ao.vx_sd - bo.vx_sd)
                of_sd_d += np.abs(ao.vy_sd - bo.vy_sd)

                of_kurt_d = np.abs(ao.vx_kurt - bo.vx_kurt)
                of_kurt_d += np.abs(ao.vy_kurt - bo.vy_kurt)

                of_rng_d = np.abs(ao.vxr_mean - bo.vxr_mean)
                of_rng_d += np.abs(ao.vyr_mean - bo.vyr_mean)
                of_rng_d += np.abs(ao.vxr_q75 - bo.vxr_q75)
                of_rng_d += np.abs(ao.vyr_q75 - bo.vyr_q75)

                of_rngsd_d = np.abs(ao.vxr_sd - bo.vxr_sd)
                of_rngsd_d += np.abs(ao.vyr_sd - bo.vyr_sd)


                compare[ak][bk] = (psa_hold_d, psa_holdsd_d, psa_ttc_d, psa_ttc_sd_d,
                                   of_mean_d, of_sd_d, of_kurt_d, of_rng_d, of_rngsd_d)

        return compare

    def simple_evaluate_dist(self, c):
        'Feed me the result of self.dist()'
        tuple_len = len(c.values()[0].values()[0])
        # for each session in training
        for k in c.keys():
            matches = 0
            # over tuple
            for i in range(tuple_len):
                # which learning session matches this tuple?
                min_v = None
                min_lk = None
                for lk in c[k].keys():
                    v = c[k][lk][i]
                    if (min_v == None) or (v < min_v):
                        min_v = v
                        min_lk = lk

                if min_lk == k: matches += 1

            # outputs what ratio of features in distance tuple matches the same person best
            print k, float(matches) / tuple_len

    def massage(self, instr):
        'Massages the orig feats data'
        self.all_data['orig'] = self.all_data['orig'].massage_all(instr)

class ParzenMethod(object):
# ranges
#   must normalize the individual columns because the kernel function has the same scale in each
#   dimension and the kernel bandwidth may be otherwise too big/small for some dimensions
#
#   OTOH, normalization does seem to remove interesting information, KDE implementation should support
#   different bandwidth for each dimension
#
#
# data count:
#   here we must have the same number of strokes (which we dont) or normalize
#   also if any log() returns -inf, it means it's 1 less count
#
# KDE at a given point:
#   can be higher than 1, just in a very small area, integral still comes to 1
#   MUST always integrate over an area, PDF for a continuous variable cannot be used in a single point
#   for distant point, intuitively, P(X|w_i) = 1/n of users, ie. each user is equally likely and we have no idea
#       in the book they have some equation to implement this
#
#
# TODO (method):
#   - split by 'dragging' or not
    def __init__(self):
        self.columns = ['vxrange', 'vyrange']

        # we don't have normalization for a different number of strokes so let's simply just limit it
        self.count = 400
        self.normalize = True
        self.pointsize = 0.1
        self.xi_resolution = 80
        self.sds_outliers = 3
        self.plot_points = True


        # TODO: move data selection and preparation out? or make it
        # reusable for other data

    def build(self, dataset, fac=None):
        # TODO: use same amount of rows or normalize for that
        #assert isinstance(dataset, DataSet)

        # select columns
        self.cix = [list(dataset.headers).index(x) for x in self.columns]
        subdata = select_columns(dataset, self.cix)

        if self.sds_outliers:
            subdata = subdata.remove_std_outliers(self.sds_outliers)

        if self.normalize:
            # normalize all users together but only the selected columns
            subdata.nums = normalize_columns(subdata.nums)


        ksp = dataset.sk_by_person()
        self.dat = {p: Dummy(dataset=subdata.user(ksp[p])) for p in ksp}

        for p in self.dat.keys():
            n = self.dat[p].dataset.copy()
            n.nums = n.nums[:self.count]
            n.ids = n.ids[:self.count]
            n.slabel = n.slabel[:self.count]

            self.dat[p].normalized = n
            self.dat[p].kde = scipy.stats.gaussian_kde(n.nums.T, fac)

    def rng(self):
        r = None
        for p in self.dat.keys():
            d = self.dat[p].normalized.nums
            X = d[:,0]
            Y = d[:,1]
            if r == None:
                r = [X.min(), X.max(), Y.min(), Y.max()]
            r[0] = min(r[0], X.min())
            r[1] = min(r[1], X.max())
            r[2] = min(r[2], Y.min())
            r[3] = min(r[3], Y.max())
        return np.array(r)


    def match_all(self, testing_dat, method, rng):
        '''testing_dat: { person: Dummy(kde, normalized,...) }'''
        maps = {}
        result = {}
        for tp in testing_dat.keys():
            result[tp] = {}
            maps[tp] = {}
            for lp in self.dat.keys():
                score, mapping = method(testing_dat[tp], self.dat[lp], rng)
                result[tp][lp] = score
                maps[tp][lp] = mapping

        self.xi_maps = maps
        return result

    def match_self(self, testing_dat, method, rng):
        return [method(testing_dat[p], self.dat[p], rng) for p in self.dat.keys()]

    def method_xi_pointbased(self, t, l, _):
        pts = np.vstack([
                t.normalized.nums,
                l.normalized.nums
        ])

        ndim = len(self.columns)
        bounds = np.ones(ndim) * 0.05

        def integrate(src_kde, x):
            if ndim == 3:
                print '.',
            return src_kde.integrate_box(x - bounds, x + bounds)

        tv = [integrate(t.kde, x) for x in pts]
        lv = [integrate(l.kde, x) for x in pts]

        vals = np.array([xisq_pt(*x) for x in zip(tv, lv)])
        result = np.sum(vals) / len(vals)
        return result, vals

    def method_xi_pt_noint(self, t, l, _):
        pts = np.vstack([
                t.normalized.nums,
                l.normalized.nums
        ]).T

        ndim = len(self.columns)

        tv = t.kde.evaluate(pts)
        lv = l.kde.evaluate(pts)

        vals = np.array([xisq_pt(*x) for x in zip(tv, lv)])
        result = np.sum(vals) / len(vals)
        return result, vals

    def method_xi(self, t, l, rng):
        '''t - testing Dummy(kde, normalized)
        l - learned Dummy(kde, normalized)
        rng - distribution range'''

        R = self.xi_resolution
        x = np.linspace(rng[0], rng[1], R)
        y = np.linspace(rng[2], rng[3], R)
        pts = np.dstack(np.meshgrid(x, y)).reshape(-1, 2).T

        # actually pts is always the same, can save it

        tv = t.kde.evaluate(pts)
        lv = l.kde.evaluate(pts)

        #print t.kde
        #print tv
        #print '-- l:'
        #print l.kde
        #print lv

        vals = np.array([xisq_pt(tv[i], lv[i]) for i in range(len(tv))])
        result = np.sum(vals) #/ len(tv)
        vals = vals.reshape(R, R)


        if result != result:
            # debug NaN
            #print vals
            print 'any nan', any(x != x for x in vals)
            print 'len', len(tv)
            print 'sum', np.sum(vals)

        return result, vals

    @staticmethod
    def eval_match(match_d):
        '''Give a score on how much the testing person (tp) matches learning person (lp)
        and discriminates others.
        match_d: { tp: { lp: score } }'''
        score = 0
        max_ = float(max(map(max, [x.values() for x in match_d.values()])))

        side = float(len(match_d.keys()))

        # it's just a small proportion, give it a larger weight
        #diag_weight = (side * side - side) / side
        diag_weight = (side - 1) / side
        #print side, diag_weight, max_

        for tp in match_d.keys():
            for lp in match_d.keys():
                # normalize all now
                v = match_d[tp][lp] / max_

                if lp == tp:
                    # here it's good when it's small
                    score -= diag_weight *  v

                else:
                    # here it's good when it's large
                    score += v

        return score


    @staticmethod
    def show_match(scores):
        # needs improvement to avoid being misleading
        #   -> added colorbar, that provides good scale

        # convert dict to array
        # scores = tp -> lp -> int
        k = scores.keys()
        ar = np.array([[scores[t][l] for l in k] for t in k])

        fig, ax = plt.subplots()
        img = ax.imshow(ar, interpolation='none', cmap=mpl.cm.get_cmap('PuBuGn'))
        ax.set_xticks(range(len(k)))
        ax.set_yticks(range(len(k)))
        ax.set_xticklabels(k)
        ax.set_yticklabels(k)
        fig.colorbar(img)
        fig.show()
        return fig, ax

    def plot(self, p):
        fig, ax = plt.subplots(1)
        return self.plot_internal(ax)

    def plot_internal(self, p, ax, Extent):
        d = self.dat[p]

        X = d.normalized.nums[:,0]
        Y = d.normalized.nums[:,1]
        D = 80

        #print 'X from', X.min(), 'to', X.max()
        #print 'Y from', Y.min(), 'to', Y.max()

        def do(x, y):
            v = d.kde.evaluate([x, y])[0]
            return v

        Z = np.array([
            [do(x, y) for x in np.linspace(Extent[0], Extent[1], D)]
            for y in np.linspace(Extent[2], Extent[3], D)]).T

        d.kde_Z = Z

        ax.set_xlim([Extent[0], Extent[1]])
        ax.set_ylim([Extent[2], Extent[3]])
        if self.plot_points:
            ax.scatter(X, Y, color=(1.0, 1.0, 0.3, 0.5), marker='.')
        ax.imshow(np.rot90(Z), extent=Extent)
        ax.set_title(p)

    def plot_testing(self, other, p, ax, Extent):
        if self.plot_points:
            d = other.dat[p].normalized.nums
            X = d[:,0]
            Y = d[:,1]
            ax.scatter(X, Y, color=(0, 0, 0, 0.5), marker='x')

    def tight_rng(self, stds=1):
        # one [min, max] entry for each dimension
        r = []
        for dim in range(len(self.columns)):
            r.append([None, None])

            for p in self.dat.keys():
                nums = self.dat[p].normalized.nums[:,dim]
                me = [nums.mean() - stds*nums.std(), nums.mean() + stds*nums.std()]

                r[-1][0] = omin([me[0], r[-1][0]])
                r[-1][1] = omax([me[1], r[-1][1]])

        return r



class SplitGrid(object):
    def __init__(self, first, second, plotter):
        # first: { p: some data }
        self.data = [first, second]
        self.persons = first.keys()
        self.plotter = plotter
        self.fig = None

    def subplots(self):
        if self.fig == None:
            self.fig, self.axes = plt.subplots( 2, len(self.persons), squeeze=False )
            tight_subplots(self.fig)
        return self.fig, self.axes


    def plot_all(self):
        self.subplots()

        # find shared range
        for d_ix, d in enumerate(self.data):
            for p_ix, p in enumerate(self.persons):
                self.plotter.adjust_range(d[p])

        # plot all at once
        for d_ix, d in enumerate(self.data):
            for p_ix, p in enumerate(self.persons):
                opt = Dummy()
                opt.p = p
                opt.p_ix = p_ix
                opt.other = self.data[1-d_ix]
                opt.owner = d
                opt.d_ix = d_ix

                self.plotter.plot( d[p], self.axes[d_ix, p_ix], opt )


class KdePlot(object):
    def __init__(self):
        self.resolution = 80
        self.extent = [-5, 5, -5, 5]

    def adjust_range(self, dat):
        logging.warning('adjust_range not implemented, ignoring')

    def plot(self, dat, ax, opt):
        # dat: Dummy(kde)
        xspace = np.linspace(self.extent[0], self.extent[1], self.resolution)
        yspace = np.linspace(self.extent[2], self.extent[3], self.resolution)

        self.Z = np.array([[dat.kde.evaluate([x, y])[0] for x in xspace] for y in yspace])

        ax.set_xlim(self.extent[1:3])
        ax.set_ylim(self.extent[2:4])
        ax.imshow(self.Z, extent=self.extent)


class AllHistograms(object):
    '''Plot histograms of (almost) all features in one big figure.'''
    def __init__(self, dat):
        self.dat = dat

    def plot(self):
        fig, ax = plt.subplots(6, 7)
        tight_subplots(fig)
        c = ['blue', 'green']

        Headers = self.dat[0].headers

        for c_ix, cname in enumerate(Headers):
            if c_ix > 20: break

            Range = [None, None]
            for ix in range(2):
                D = self.dat[ix].cbyname(cname)

                Range[0] = omin([np.percentile(D, 5), Range[0]])
                Range[1] = omax([np.percentile(D, 95), Range[1]])

            bins = np.linspace(Range[0], Range[1], 40)
            for ix in range(2):
                D = self.dat[ix].cbyname(cname)

                order = c_ix * 2 + ix
                this_ax = ax[order % 6, order / 6]
                this_ax.hist(D, bins, color=c[ix])
                this_ax.set_title(cname)


def binize(ds, c_ix):
    '''For doing the X2 test of independence or feature selection.'''
    N = 40

    X = ds.nums[:,c_ix]
    bins = np.linspace(np.percentile(X, 10), np.percentile(X, 90), N)

    sks = ds.sk_by_person()
    tbl = np.vstack( np.histogram(ds.person(p).nums[:,c_ix], bins)[0] for p in sks.keys() )
    return pd.DataFrame(
        tbl,
        index=sks.keys(),
        columns=range(N-1))


class CompareParzenMethod(object):
    def __init__(self, datasrc, columns):
        '''datasrc: [testing orig feats, learning orig feats]'''
        self.datasrc = datasrc
        self.columns = columns
        self.plot = False

    def build_parzen(self):
        pm = []
        for d_ix, d in enumerate(self.datasrc):
            parzen = ParzenMethod()
            parzen.columns = self.columns
            parzen.sds_outliers = None
            parzen.build(d, 0.18)
            pm.append(parzen)

        self.pm = pm
        return pm

    def run(self):
        pm = self.build_parzen()

        rng = [None, None, None, None]
        for x in pm:
            tr = x.tight_rng(stds=1.5)
            rng[0] = omin([rng[0], tr[0][0]])
            rng[1] = omax([rng[1], tr[0][1]])
            rng[2] = omin([rng[2], tr[1][0]])
            rng[3] = omax([rng[3], tr[1][1]])


        class ParzenPlotter(KdePlot):
            def adjust_range(self, x):
                pass

            def plot(self, dat, ax, opt):
                pm[opt.d_ix].plot_internal(opt.p, ax, rng)
                pm[opt.d_ix].plot_testing(pm[1-opt.d_ix], opt.p, ax, rng)

        match_d = pm[0].match_all(pm[1].dat, pm[0].method_xi, rng)
        if self.plot:
            sg = SplitGrid(pm[0].dat, pm[1].dat, ParzenPlotter())
            sg.plot_all()
            ParzenMethod.show_match(match_d)

        # plotting
        # pm[0] -- learn, pm[1] -- test


        XiRange = [0, 1]
        if self.plot:
            N = len(pm[0].dat)
            xifig, xiax = plt.subplots(N, N)
            tight_subplots(xifig)
            Extent = rng
            for l_ix, lp in enumerate(pm[0].dat.keys()):
                for t_ix, tp in enumerate(pm[1].dat.keys()):
                    Z = pm[0].xi_maps[tp][lp]
                    xiax[t_ix][l_ix].set_xlim([rng[0], rng[1]])
                    xiax[t_ix][l_ix].set_ylim([rng[2], rng[3]])
                    # here data was upside down
                    xiax[t_ix][l_ix].imshow(np.flipud(Z), extent=Extent, vmin=XiRange[0], vmax=XiRange[1])
                    xiax[t_ix][l_ix].set_title('t:{}/l:{}'.format(tp, lp))

            # TODO: vmin, vmax imho nefunguje tak jak bych chtel
            # aby to vynechalo fakt maly hodnoty ... a odpovidalo tomu 3x3 teplotnimu grafu
            for p_ix, p in enumerate(pm[0].dat.keys()):
                for d_ix in range(2):
                    CS = sg.axes[d_ix, p_ix].contour(pm[0].xi_maps[p][p],
                                                extent=Extent, vmin=XiRange[0], vmax=XiRange[1],
                                                cmap=mpl.cm.get_cmap('PuBuGn'))
                    sg.axes[d_ix, p_ix].clabel(CS, inline=1)

        return match_d

    def run_multid(self):
        'Runs the multidimensional comparison using Xi2 over points only'
        pm = self.build_parzen()

        # range is not used, just go according to the points

        match_d = pm[0].match_all(pm[1].dat, pm[0].method_xi_pt_noint, None)

        return match_d

    def run_self_similarity(self):
        pm = self.build_parzen()

        parts = [x[0] for x in pm[0].match_self(pm[1].dat, pm[0].method_xi_pointbased, None)]
        return {'parts': parts,
                'summed': sum(parts),
                'feats': self.columns}

    @classmethod
    def try_self_similarity(cls, datasrc):
        Headers = ['vavg', 'crange', 'cssd']

        results = []

        # 1D
        print '-----    1 D   -----'
        for i in Headers:
            cpm = CompareParzenMethod(datasrc, [i])
            r = cpm.run_self_similarity()
            results.append(r)

        # 2D
        print '-----    22 D   -----'
        for i in Headers[1:]:
            cpm = CompareParzenMethod(datasrc, [Headers[0], i])
            r = cpm.run_self_similarity()
            results.append(r)


        # 3D
        print '-----    333 D   -----'
        cpm = CompareParzenMethod(datasrc, Headers)
        r = cpm.run_self_similarity()
        results.append(r)

        return results

    @classmethod
    def rate_1feat_selfsim(cls, datasrc):
        Headers = list(datasrc[0].headers[:73])

        results = []
        for i in Headers:
            cpm = CompareParzenMethod(datasrc, [i])
            r = cpm.run_self_similarity()
            results.append(r)

        results.sort(key=lambda z: z['summed'])
        return results





class TimeStepsAnalysis(object):
    'For seeing what are the usual intervals between points'

    @staticmethod
    def load_time_steps(datadir):
        fname = os.path.join(datadir, 'time_steps.csv')
        with open(fname) as inf:
            dat = inf.read()

        return [float(x) for x in dat.split(', ') if x != ''][1:]

    def hist(self, ts):
        h = plt.hist(ts, range=[0, 0.032], bins=250, histtype='bar')
        print zip(*h)



class OrigFeatureSelection(object):
    def __init__(self, csds):
        self.csds = csds
        self.splits = [csds.first['orig'], csds.second['orig']]

    def calc_kde_selfsim(self):
        ratings = CompareParzenMethod.rate_1feat_selfsim(self.splits)
        return ratings

    def adjust_kde_selfsim(self, ratings):
        self.selfsim_ratings = ratings
        vals = np.array([z['summed'] for z in ratings])
        min_, max_ = np.min(vals), np.max(vals)
        range_ = max_ - min_

        self.selfsim_ratings_dict = {}
        for ix, _ in enumerate(ratings):
            d = ratings[ix]
            d['norm'] = (ratings[ix]['summed'] - min_) / range_
            self.selfsim_ratings_dict[ratings[ix]['feats'][0]] = d


    def calc_xi2(self):
        x = self.csds.all_data['orig']

        def elem_fn(x, c_ix):
            try:
                return scipy.stats.chi2_contingency(binize(x, c_ix))[0]
            except ValueError as e:
                return 1

        feat_rank = [(x.headers[c_ix], elem_fn(x, c_ix)) for c_ix in range(len(x.headers))]
        feat_rank = [x for x in feat_rank if x[1] != 1]
        feat_rank.sort(key=lambda (a,b): b, reverse=False)

        vals = [x[1] for x in feat_rank]
        min_, max_ = np.min(vals), np.max(vals)
        range_ = max_ - min_

        self.xi2 = {x[0]: (x[1], (x[1] - min_) / range_) for x in feat_rank}

    def rate_both(self):
        # min of selfsim_ratings.norm
        # max of xi2

        def rating_fn(ft):
            try:
                ss = self.selfsim_ratings_dict[ft]['norm']
                xi2 = self.xi2[ft][1]

                return ss - xi2
                #if pv > 0.05: return 10
                #return ss + (pv / 0.05)
            except KeyError:
                return 10

        vals = [(ft, rating_fn(ft)) for ft in self.csds.all_data['orig'].headers]
        # small values first, matches the rating_fn output
        vals.sort(key=lambda (a,b): b, reverse=False)
        self.both_rating = vals
        return vals


    def copy_from(self, ofs):
        self.selfsim_ratings_dict = ofs.selfsim_ratings_dict
        self.xi2 = ofs.xi2

    def random_feature_selection(self):
        Headers = self.csds.all_data['orig'].headers[:72]

        result = []
        for i in range(500):
            n = random.choice(range(2, 7))
            h = random.sample(Headers, n)

            cpm = CompareParzenMethod(self.splits, h)
            cpm.plot = False
            match_d = cpm.run_multid()
            score = ParzenMethod.eval_match(match_d)

            result.append(Dummy(n=n, h=h, match_d=match_d, score=score))
            print '.',

        result.sort(key=lambda f: f.score, reverse=True)
        print 'done'
        return result


class OrigFeatureSkL(object):
    def __init__(self, learn, test):
        '''Trying to apply scikit-learn implemented algorithms on the original features.
        learn, test: Dataset()
        '''
        self.learn = learn
        self.test = test


    def select_columns(self, colnames):
        self.cix = [list(self.learn.headers).index(x) for x in colnames]
        self.columns = colnames

        self.learn = select_columns(self.learn, self.cix)
        self.test = select_columns(self.test, self.cix)

    def normalize(self):
        '''Normalize each column independently, remember the
        coefficients and normalize self.test the same way'''
        # first remove outliers
        sds_outliers = 4
        self.learn = self.learn.remove_std_outliers(sds_outliers)
        self.test = self.test.remove_std_outliers(sds_outliers)

        # do the normalization itself
        self.norm_params = [Dummy(mean=0.0, std=0.0) for _ in self.columns]
        arr = []
        for i in range(len(self.columns)):
            params = self.norm_params[i]
            vals = self.learn.nums[:,i]
            params.mean = vals.mean()
            params.std = vals.std()

            vals = (vals - params.mean) / params.std
            arr.append(vals)

        self.learn.nums = np.vstack(arr).T

        # now normalize test the same way
        arr = []
        for i in range(len(self.columns)):
            params = self.norm_params[i]
            vals = self.test.nums[:,i]

            vals = (vals - params.mean) / params.std
            arr.append(vals)

        self.test.nums = np.vstack(arr).T

    def run_svm(self):
        from sklearn import svm
        self.clf = svm.SVC() #kernel='poly', degree=3)
        self.clf.fit(self.learn.nums, self.learn.ids)

    def run_rforest(self):
        from sklearn.ensemble import RandomForestClassifier
        self.clf = RandomForestClassifier(n_estimators=90)
        self.clf.fit(self.learn.nums, self.learn.ids)


    def run(self):
        self.run_svm()

        # eval
        ok_cnt = 0
        for row in range(len(self.test.nums)):
            predicted = self.clf.predict(self.test.nums[row])[0]
            real = self.test.ids[row]

            if predicted == real:
                ok_cnt += 1

        return float(ok_cnt) / len(self.test.nums)


class SegmentedVHistCompare(object):
    def __init__(self, sessions):
        self.sessions = sessions

    def segment_vhist(self, strokes_partitioned, slabels):
        '''Segments strokes by speed / length into several groups so that we can
        do stats on comparable strokes only.

        strokes_partitioned: { p: [ stroke ] }
            assumed already divided into testing/learning
            and split by person
            but not filtered by length or anything
        '''

        params = [{'min_len': 5, 'max_len': 15, 'slabels': None},
                  {'min_len': 16, 'max_len': 40, 'slabels': None},
                  {'min_len': 41, 'max_len': 200, 'slabels': None}]

        if slabels:
            # there are not many, split in 2 groups only
            slabels = set(slabels)
            params += [{'min_len': 10, 'max_len': 150, 'slabels': slabels}]

        segs = [VHistSegment(strokes_partitioned, self.sessions, p) for p in params]
        return segs

    @classmethod
    def dist_vhist(cls, test, learn, tp, lp):
        '''Calculates a normalized distance vector between segmented datasets learn, test.

        learn, test: [ VHistSegment() ]
        lp, tp: persons that we are comparing from the learn dataset and test dataset respectively

        normalization:
            0 means very similar
            1 means completely different

            try to normalize using some knowledge about the values
            ideally so that it can reach both 0 or 1 to use the full range
        '''

        dist_res = {}

        # TODO: check if this works with negative values (th, cs, ...)
        for sg_ix in range(len(learn)):
            def store(k, v):
                dist_res['{}_{}'.format(sg_ix, k)] = v

            sg_l = learn[sg_ix].vhist.dat[lp].smp_stats
            sg_t = test[sg_ix].vhist.dat[tp].smp_stats

            # also tried mean square error but it doesn't make much difference or improvement

            med_d = np.abs(sg_l.Ymed - sg_t.Ymed)
            med_max = max(np.max(sg_l.Ymed), np.max(sg_t.Ymed))
            store('median_d_avg', np.average(med_d) / med_max)
            store('median_d_max', np.max(med_d) / med_max)

            y5_d = np.abs(sg_l.Y5 - sg_t.Y5)
            y5_max = max(np.max(sg_l.Y5), np.max(sg_t.Y5))
            store('y5_d_avg', np.average(y5_d) / y5_max)
            store('y5_d_max', np.max(y5_d) / y5_max)

            store('median_asymetry_d', np.abs(sg_l.median_asymetry - sg_t.median_asymetry))
            store('y5_asymetry_d', np.abs(sg_l.y5_asymetry - sg_t.y5_asymetry))

            # TODO: find more shape parameters

        return dist_res

    def compare_new(self, test, learn, slabels):
        test_segs = self.segment_vhist(test['strokes'], slabels)
        learn_segs = self.segment_vhist(learn['strokes'], slabels)

        compare = {}
        people = self.sessions.values()
        for tp in people:
            compare[tp] = {}
            for lp in people:
                compare[tp][lp] = self.dist_vhist(test_segs, learn_segs, tp, lp)

        return compare

    def dist_by_seg(self, comparison, sg_ix):
        '''comparison is what you get from self.compare_new
        comparison: { tp: { lp: { result of self.dist_vhist } } }'''
        result = {}
        for tp in comparison.keys():
            result[tp] = {}
            for lp in comparison[tp].keys():
                val = comparison[tp][lp]
                seg_keys = [i for i in val.keys() if i.split('_')[0] == str(sg_ix)]

                result[tp][lp] = sum(val[i] for i in seg_keys) / len(seg_keys)

        return result

    def sum_results(self, comp):
        sum2_fn = lambda k, v: sum(v.values())
        sum1_fn = lambda k, v: dictmap(sum2_fn, v)

        return dictmap(sum1_fn, comp)
