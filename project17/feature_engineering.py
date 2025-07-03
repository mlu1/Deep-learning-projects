#!/usr/bin/env python3
"""
Build enriched user profiles — **SVD‑200 + DeepFM‑64** with extra requested features
=====================================================================================
Rich handcrafted features plus:
  • Time since last purchase per category
  • Peak activity hours (mode hour sin/cos)
  • Seasonal purchase mix (quarter fractions)
  • Cart-abandonment rate
  • Search-to-buy conversion rate
  • Browsing diversity
  • Cumulative spend per category (sum)
  • Discount sensitivity (std/mean of normalized price)
  • Session interval regularity (std gap)
  • Median session duration
  • Page visits before purchase
  • Interaction sequence depth (max & avg order)
  • Event-type diversity
200‑dim SVD + 64‑dim DeepFM ⇒ 264‑dim embeddings saved.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD

# ─── CONFIG ───────────────────────────────────────────────────────────────
DATA_DIR   = Path("data")
EVENT_FILES = [
    "product_buy.parquet","add_to_cart.parquet","remove_from_cart.parquet",
    "page_visit.parquet","search_query.parquet",
]
PROD_PROPS = DATA_DIR / "product_properties.parquet"
RELEVANT   = DATA_DIR / "input/relevant_clients.npy"
OUTPUT_DIR = Path("my_submission"); OUTPUT_DIR.mkdir(exist_ok=True)
TIME_WINDOWS = (1,7,14,30,60,90)
SVD_DIM     = 200

SAVE_DIR = Path("my_save");
SAVE_DIR.mkdir(exist_ok=True)
FEAT_PARQ  = SAVE_DIR / "user_feats.parquet"
FEAT_NPY   = SAVE_DIR / "user_feats.npy"
FEAT_IDX   = SAVE_DIR / "user_feats_idx.npy"


# ─── HELPERS --------------------------------------------------------------
parse_emb = lambda s: np.fromstring(s.strip("[]"), sep=" ").astype(np.float32)
one_hot   = lambda s,p: pd.get_dummies(s,prefix=p)

# ─── LOAD EVENTS ----------------------------------------------------------
clients = set(np.load(RELEVANT))
frames = []
for fn in EVENT_FILES:
    df = pd.read_parquet(DATA_DIR/fn)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df[df['client_id'].isin(clients)]
    df['event_type'] = fn.replace('.parquet','')
    frames.append(df)
ev = pd.concat(frames,ignore_index=True)
MAX_TS = ev['timestamp'].max()
print('Events loaded:', len(ev))

# ─── BASE COUNTS & TIME WINDOWS -----------------------------------------
user_evt = ev.groupby(['client_id','event_type']).size().unstack(fill_value=0)
for w in TIME_WINDOWS:
    tmp = ev[ev['timestamp']>=MAX_TS-pd.Timedelta(days=w)]
    cnt = tmp.groupby(['client_id','event_type']).size().unstack(fill_value=0)
    user_evt = user_evt.join(cnt.add_prefix(f'evt_last{w}d_'),how='left').fillna(0)

rate30 = (
    ev[ev['timestamp'] >= MAX_TS - pd.Timedelta(days=30)]
      .groupby('client_id').size().div(30).to_frame('event_rate_30d')
)
# ─── CROSS-FEATURE INTERACTIONS & NON-LINEAR TRANSFORMS
# Create pairwise interaction features between key event types
interaction_cols = ['product_buy', 'page_visit', 'add_to_cart']
from itertools import combinations
for a, b in combinations(interaction_cols, 2):
    if a in user_evt.columns and b in user_evt.columns:
        user_evt[f'{a}_x_{b}'] = user_evt[a] * user_evt[b]
# Add non-linear kernels: square and log1p of counts
for c in interaction_cols:
    if c in user_evt.columns:
        user_evt[f'{c}_sq'] = user_evt[c] ** 2
        user_evt.replace([np.inf,-np.inf],0, inplace=True)
        user_evt.fillna(0, inplace=True)
        user_evt.replace([np.inf,-np.inf],0, inplace=True)
        user_evt.fillna(0, inplace=True)
        user_evt[f'{c}_log1p'] = np.log1p(user_evt[c])
        

# ─── TEMPORAL FEATURES ----------------------------------------------------
ev['dow'] = ev['timestamp'].dt.dayofweek
user_dow   = one_hot(ev['dow'],'dow').assign(client_id=ev['client_id']).groupby('client_id').sum()
user_hr    = one_hot(ev['timestamp'].dt.hour,'hr').assign(client_id=ev['client_id']).groupby('client_id').sum()
ev['part_of_day']=pd.cut(ev['timestamp'].dt.hour,[0,6,12,18,24],labels=['night','morning','afternoon','evening'],right=False)
user_pod   = one_hot(ev['part_of_day'],'pod').assign(client_id=ev['client_id']).groupby('client_id').sum()
ev['hr_sin']=np.sin(2*np.pi*ev['timestamp'].dt.hour/24); ev['hr_cos']=np.cos(2*np.pi*ev['timestamp'].dt.hour/24)
user_hour_cycle = ev.groupby('client_id')[['hr_sin','hr_cos']].mean()
# month/quarter/week/day-of-year
ev['month']=ev['timestamp'].dt.month; ev['quarter']=ev['timestamp'].dt.quarter
user_month   = one_hot(ev['month'],'m').assign(client_id=ev['client_id']).groupby('client_id').sum()
user_quarter = one_hot(ev['quarter'],'q').assign(client_id=ev['client_id']).groupby('client_id').sum()
ev['week']=ev['timestamp'].dt.isocalendar().week.astype(int)
user_week    = one_hot(ev['week'],'w').assign(client_id=ev['client_id']).groupby('client_id').sum()
ev['doy']=ev['timestamp'].dt.dayofyear
user_doy_cycle = pd.DataFrame({
    'doy_sin':np.sin(2*np.pi*ev['doy']/365.25),
    'doy_cos':np.cos(2*np.pi*ev['doy']/365.25)
}, index=ev.index)
user_doy_cycle = user_doy_cycle.assign(client_id=ev['client_id']).groupby('client_id').mean()
# peak activity hour
peak = ev.groupby('client_id')['timestamp'].apply(lambda x: x.dt.hour.mode()[0]).to_frame('peak_hr')
peak['peak_sin']=np.sin(2*np.pi*peak['peak_hr']/24); peak['peak_cos']=np.cos(2*np.pi*peak['peak_hr']/24)
peak.drop(columns='peak_hr',inplace=True)
# seasonal buy mix
buys=ev[ev['event_type']=='product_buy']
buy_q = buys.groupby(['client_id',buys['timestamp'].dt.quarter]).size().unstack(fill_value=0)
buy_q_frac = buy_q.div(buy_q.sum(axis=1),axis=0).add_prefix('buy_q').fillna(0)
buy_weekend_frac = buys.groupby('client_id')['dow'].apply(lambda x: (x>=5).mean()).to_frame('buy_weekend_frac')
last_buy_days = (MAX_TS - buys.groupby('client_id')['timestamp'].max()).dt.days.to_frame('last_buy_days')


# ─── SESSION FEATURES -----------------------------------------------------
seq = ev.sort_values(['client_id','timestamp'])
seq['gap'] = seq.groupby('client_id')['timestamp'].diff().dt.total_seconds()/60
seq['new_sess'] = (seq['gap']>30)|seq['gap'].isna()
seq['sess_id'] = seq.groupby('client_id')['new_sess'].cumsum()
sl = seq.groupby(['client_id','sess_id']).size().groupby('client_id')
user_sess = pd.DataFrame({'session_count':sl.size(),'avg_sess_len':sl.mean()})
sess_std = seq.groupby('client_id')['gap'].std().fillna(0).to_frame('sess_gap_std')
med_len  = sl.apply(np.median).to_frame('median_sess_len')
sess_cnt = seq.groupby('client_id')['sess_id'].nunique().to_frame('session_count')
queries_per_session = (
    user_evt.get('search_query', pd.Series(0, index=user_evt.index)) /
    (sess_cnt['session_count']+1e-6)
).to_frame('queries_per_session')

sess_dur = (
    seq.groupby(['client_id','sess_id'])['timestamp']
       .agg(session_start='min', session_end='max')
)
sess_dur['session_duration'] = (
    (sess_dur['session_end'] - sess_dur['session_start'])
    .dt.total_seconds() / 60
)

# Aggregate per user
user_sess_dur = sess_dur.reset_index().groupby('client_id')['session_duration'].agg(
    avg_sess_dur='mean',
    median_sess_dur='median',
    max_sess_dur='max',
    min_sess_dur='min'
)

sess_times = seq.groupby(['client_id','sess_id'])['timestamp'].agg(
    start='min', end='max')
# shift per user to compute inter-session gap
sess_times = sess_times.sort_index(level=[0,1])
sess_times['prev_end'] = sess_times.groupby(level=0)['end'].shift(1)
# gap in hours or days
sess_times['inter_sess_gap_h'] = (
    (sess_times['start'] - sess_times['prev_end'])
    .dt.total_seconds() / 3600
)
# per user aggregates
user_inter_gap = sess_times.groupby('client_id')['inter_sess_gap_h'].agg(
    avg_inter_gap='mean',
    median_inter_gap='median',
    max_inter_gap='max'
).fillna(0)


sess_times = seq.groupby(['client_id','sess_id'])['timestamp'].agg(
    start='min', end='max')
# shift per user to compute inter-session gap
sess_times = sess_times.sort_index(level=[0,1])
sess_times['prev_end'] = sess_times.groupby(level=0)['end'].shift(1)
# gap in hours or days
sess_times['inter_sess_gap_h'] = (
    (sess_times['start'] - sess_times['prev_end'])
    .dt.total_seconds() / 3600
)
# per user aggregates
user_inter_gap = sess_times.groupby('client_id')['inter_sess_gap_h'].agg(
    avg_inter_gap='mean',
    median_inter_gap='median',
    max_inter_gap='max'
).fillna(0)


sess_times['hour'] = sess_times['start'].dt.hour
bins = [0,6,12,18,24]
labels = ['night','morning','afternoon','evening']
sess_times['pod'] = pd.cut(sess_times['hour'], bins, labels=labels, right=False)

# count per user
pod_counts = (
    pd.get_dummies(sess_times['pod'])
      .groupby(sess_times.index.get_level_values('client_id'))
      .sum()
)
# and normalize
pod_frac = pod_counts.div(pod_counts.sum(axis=1), axis=0).add_prefix('frac_sess_')


sess_times['date'] = sess_times['start'].dt.date
daily_sess = sess_times.reset_index().groupby(['client_id','date']).size().to_frame('sess_per_day')
# then per user
user_daily = daily_sess.groupby('client_id')['sess_per_day'].agg(
    avg_sess_per_day='mean',
    max_sess_per_day='max'
)


has_purchase = seq.assign(
    purchase_flag = (seq['event_type']=='purchase').astype(int)
).groupby(['client_id','sess_id'])['purchase_flag'].max().to_frame()
# fraction of sessions with a purchase
frac_purchase_sess = has_purchase.groupby('client_id')['purchase_flag'].mean().to_frame('frac_sess_with_purchase')


last_session = sess_times.groupby('client_id')['end'].max()
ref_date = pd.Timestamp('now')
recency = (ref_date - last_session).dt.days.to_frame('days_since_last_sess')


# ─── PRODUCT & PRICE STATS -----------------------------------------------
pp = pd.read_parquet(PROD_PROPS)
sku = ev[ev['event_type'].isin(['product_buy','add_to_cart','remove_from_cart'])].merge(pp,on='sku',how='left')
sku['name_vec']=sku['name'].apply(parse_emb)
sku['mon']=sku['timestamp'].dt.to_period('M')
med=sku.groupby('mon')['price'].median().to_dict(); sku['price_n']=sku.apply(lambda r:r['price']/med[r['mon']],axis=1)
price_stats = sku.groupby('client_id')['price_n'].quantile([0,.25,.5,.75,1]).unstack()
price_stats.columns=[f'pnorm_q{int(q*100)}' for q in (0,25,50,75,100)]
user_cat = one_hot(sku['category'],'cat').assign(client_id=sku['client_id']).groupby('client_id').sum()
cat30    = sku[sku['timestamp']>=MAX_TS-pd.Timedelta(days=30)].groupby('client_id')['category'].nunique().to_frame('distinct_cat_30d')
sku['decay']=np.exp(-((MAX_TS-sku['timestamp']).dt.days)/14)
rec_div  = sku.groupby(['client_id','category'])['decay'].max().groupby('client_id').sum().to_frame('recency_cat_div')
ndim=sku['name_vec'].iloc[0].shape[0]
ndf=pd.DataFrame(sku['name_vec'].tolist(),columns=[f'name_emb_{i}' for i in range(ndim)])
ndf['client_id']=sku['client_id']; user_name=ndf.groupby('client_id').mean()
# spend & discount
cum    = sku.groupby('client_id')['price_n'].sum().to_frame('cum_spend')
var    = sku.groupby('client_id')['price_n'].var().fillna(0).to_frame('price_var')
mean_n = sku.groupby('client_id')['price_n'].mean().to_frame('price_n_mean')
disc   = (var['price_var']/(mean_n['price_n_mean']+1e-6)).to_frame('discount_sensitivity')

evt_counts = (
    sku.groupby(['client_id','event_type'])
       .size()
       .unstack(fill_value=0)
)
# buy/add and remove/add ratios
evt_counts['buy_to_add']    = evt_counts['product_buy'] / (evt_counts['add_to_cart']+1e-6)
evt_counts['remove_to_add'] = evt_counts['remove_from_cart'] / (evt_counts['add_to_cart']+1e-6)
distinct_sku   = sku.groupby('client_id')['sku'].nunique().to_frame('distinct_skus')
distinct_sku30 = sku[sku['timestamp']>=MAX_TS-pd.Timedelta(30,'d')] \
                   .groupby('client_id')['sku'] \
                   .nunique() \
                   .to_frame('distinct_skus_30d')

from scipy.stats import linregress

def price_trend(df):
    # convert timestamps to ordinal (days since epoch)
    x = (df['timestamp'].view('int64') // 10**9 / 86400).values
    y = df['price_n'].values
    if len(x)>1:
        return linregress(x,y).slope
    else:
        return 0


spend_30d = sku[sku['timestamp']>=MAX_TS-pd.Timedelta(30,'d')] \
               .groupby('client_id')['price_n'].sum() \
               .to_frame('spend_30d')
# historic avg monthly spend
mon_spend = sku.groupby(['client_id','mon'])['price_n'].sum().groupby('client_id').mean().to_frame('avg_monthly_spend')
# ratio
spend_ratio = (spend_30d['spend_30d'] / (mon_spend['avg_monthly_spend']+1e-6)).to_frame('spend_30d_to_avg_month')


first_evt = sku.groupby('client_id')['timestamp'].min().to_frame('first_evt')
last_evt  = sku.groupby('client_id')['timestamp'].max().to_frame('last_evt')

recency_freq = (
    last_evt
    .assign(
        days_since_last = (MAX_TS - last_evt['last_evt']).dt.days,
        days_active     = (last_evt['last_evt'] - first_evt['first_evt']).dt.days,
        total_days      = ((last_evt['last_evt'] - first_evt['first_evt']).dt.days+1)
    )
)
recency_freq['avg_events_per_day'] = sku.groupby('client_id').size() / (recency_freq['total_days']+1e-6)


sku = sku.sort_values(['client_id','timestamp'])
sku['price_n_roll_var_5'] = sku.groupby('client_id')['price_n'].transform(lambda x: x.rolling(5, min_periods=1).var())
# then aggregate
roll_var = sku.groupby('client_id')['price_n_roll_var_5'].mean().to_frame('avg_roll_var_5')

sku['hour'] = sku['timestamp'].dt.hour
hour_feats = (
    pd.get_dummies(sku['hour'], prefix='hour')
      .groupby(sku['client_id'])
      .mean()
)

# day of week
sku['dow'] = sku['timestamp'].dt.dayofweek
dow_feats = (
    pd.get_dummies(sku['dow'], prefix='dow')
      .groupby(sku['client_id'])
      .mean()
)

from scipy.stats import entropy

def cat_entropy(series):
    p = series.value_counts(normalize=True)
    return entropy(p)

cat_ent = sku.groupby('client_id')['category'].apply(cat_entropy).to_frame('cat_entropy')


# ─── BEHAVIORAL & DIVERSITY ----------------------------------------------
sku_div       = ev.groupby('client_id')['sku'].nunique().to_frame('distinct_sku_count')
avg_gap       = seq.groupby('client_id')['gap'].mean().to_frame('avg_event_gap')
price_var_df  = sku.groupby('client_id')['price_n'].var().fillna(0).to_frame('price_var')
revisit_freq  = sku.groupby(['client_id','sku']).size().groupby('client_id').mean().to_frame('sku_revisit_freq')
weekend_ratio = ev.groupby('client_id')['dow'].apply(lambda x: x.isin([5,6]).mean()).to_frame('weekend_event_ratio')
sku_pop = ev.groupby('sku').size(); ev['pop']=ev['sku'].map(sku_pop)
user_pop = ev.groupby('client_id')['pop'].mean().to_frame('avg_item_popularity')

# 1. Event‐type entropy: how evenly a user’s events are spread across types
from scipy.stats import entropy

def evt_entropy(df):
    p = df.value_counts(normalize=True)
    return entropy(p)

evt_ent = (
    ev.groupby('client_id')['event_type']
      .apply(evt_entropy)
      .to_frame('event_type_entropy')
)

# 2. Category Gini (inequality) — high if user concentrates on few categories
def gini(arr):
    # arr: counts per category
    arr = np.array(arr, dtype=float)
    if arr.sum() == 0:
        return 0.
    arr = np.sort(arr)
    n = len(arr)
    cum = np.cumsum(arr)
    return (n + 1 - 2*(cum/arr.sum()).sum())/n

cat_gini = (
    sku.groupby(['client_id','category']).size()
       .groupby(level=0)
       .apply(lambda s: gini(s.values))
       .to_frame('category_gini')
)

# 3. Novelty rate: fraction of SKUs in the last 7 days that the user has never seen before
recent = sku[sku['timestamp'] >= MAX_TS - pd.Timedelta(7,'d')]
first_seen = sku.groupby('sku')['timestamp'].min().to_dict()
novelty = (
    recent.assign(
       is_new = recent['timestamp'].eq(recent['sku'].map(first_seen))
    )
    .groupby('client_id')['is_new']
    .mean()
    .to_frame('novel_sku_ratio_7d')
)

# 4. Daily activity span: average number of active hours per day
hours_per_day = (
    ev.assign(hour=ev['timestamp'].dt.hour)
      .groupby(['client_id', ev['timestamp'].dt.date])['hour']
      .nunique()
      .groupby('client_id')
      .mean()
      .to_frame('avg_active_hours_per_day')
)

# 5. Time‐of‐day entropy: does the user concentrate events into specific hours or spread out?
def time_entropy(hours):
    p = hours.value_counts(normalize=True)
    return entropy(p)

tod_ent = (
    ev.groupby('client_id')['timestamp']
      .apply(lambda ts: time_entropy(ts.dt.hour))
      .to_frame('hour_of_day_entropy')
)

# 6. Recency‐weighted SKU diversity: distinct SKUs but weighting recent ones more
sku_rec_wt = (
    sku.assign(wt = np.exp(-((MAX_TS - sku['timestamp']).dt.days)/7))
       .groupby('client_id')
       .apply(lambda df: df.groupby('sku')['wt'].sum().gt(0).sum())
       .to_frame('recency_wt_distinct_sku')
)

# 7. Price skewness & kurtosis — gives sense of tails in spend distribution
from scipy.stats import skew, kurtosis

price_moments = (
    sku.groupby('client_id')['price_n']
       .agg(price_skew=lambda x: skew(x, bias=False) if len(x)>2 else 0,
            price_kurt=lambda x: kurtosis(x, bias=False) if len(x)>3 else 0)
)

# 8. Session burst ratio: max events in any session ÷ avg events/session
sess_sizes = seq.groupby(['client_id','sess_id']).size().to_frame('size')
burst_ratio = (
    sess_sizes.groupby('client_id')['size']
              .agg(lambda x: x.max()/ (x.mean()+1e-6))
              .to_frame('session_burst_ratio')
)

# 9. Weekday vs. weekend spend ratio: compare normalized spend on weekends vs weekdays
sku = sku.assign(is_weekend=sku['timestamp'].dt.weekday.isin([5,6]))
spend_wd_we = (
    sku.groupby(['client_id','is_weekend'])['price_n']
       .sum()
       .unstack(fill_value=0)
       .assign(
         weekend_to_weekday = lambda df: df[True] / (df[False]+1e-6)
       )
       [['weekend_to_weekday']]
       .rename(columns={'weekend_to_weekday':'spend_wknd_wd_ratio'})
)

# 10. Event‐rate trend: slope of events/day over the last N days
def rate_trend(df, days=30):
    daily = df['timestamp'].dt.floor('D').value_counts().sort_index()
    if len(daily)>1:
        x = (daily.index.astype(np.int64)//1e9/86400).astype(float)
        y = daily.values.astype(float)
        return linregress(x,y).slope
    return 0

rate_tr = (
    ev.groupby('client_id')
      .apply(lambda df: rate_trend(df, days=30))
      .to_frame('event_rate_trend_30d')
)

# cart abandon & search conv
ac     = user_evt.get('add_to_cart',pd.Series(0,index=user_evt.index))
pb     = user_evt.get('product_buy',pd.Series(0,index=user_evt.index))
cart_abandon = ((ac-pb)/(ac+1)).to_frame('cart_abandon')
sc     = user_evt.get('search_query',pd.Series(0,index=user_evt.index))
search_conv  = (pb/(sc+1)).to_frame('search_conv')


sku_ac = sku[sku['event_type']=='add_to_cart']
repeat_add = sku_ac.groupby(['client_id','sku']).size().groupby('client_id').mean().to_frame('repeat_add_freq')

cart_buys = sku[sku['event_type'].isin(['add_to_cart','product_buy'])]
cart_buys['sess_id'] = cart_buys.groupby('client_id')['timestamp'].diff().dt.total_seconds().gt(1800).cumsum()

immediate_conv = (
    cart_buys.groupby(['client_id','sku','sess_id'])['event_type']
    .agg(lambda x: set(x)=={'add_to_cart','product_buy'})
    .groupby('client_id').mean()
    .to_frame('immediate_conv_rate')
)


searches = ev[ev['event_type']=='search_query'][['client_id','timestamp']]
purchases = ev[ev['event_type']=='product_buy'][['client_id','timestamp']]

search_to_buy = searches.merge(purchases, on='client_id', suffixes=('_search','_buy'))
search_to_buy = search_to_buy[search_to_buy['timestamp_buy'] > search_to_buy['timestamp_search']]
search_to_buy['lag_minutes'] = (search_to_buy['timestamp_buy'] - search_to_buy['timestamp_search']).dt.total_seconds() / 60

query_purchase_speed = search_to_buy.groupby('client_id')['lag_minutes'].median().to_frame('median_query_buy_lag')

ac_skus = set(sku_ac['sku'])
pb_skus = set(sku[sku['event_type']=='product_buy']['sku'])

abandoned_skus = sku_ac[~sku_ac['sku'].isin(pb_skus)]
abandoned_sku_diversity = abandoned_skus.groupby('client_id')['sku'].nunique().to_frame('abandoned_sku_diversity')

search_per_day = (
    ev[ev['event_type']=='search_query']
    .groupby(['client_id', ev['timestamp'].dt.date])
    .size().groupby('client_id').mean().to_frame('searches_per_active_day')
)


sku_ac['price_quartile'] = pd.qcut(sku_ac['price_n'], 4, labels=False)
sku_pb_set = set(sku[sku['event_type']=='product_buy'][['client_id','sku']].itertuples(index=False, name=None))

sku_ac['abandoned'] = ~sku_ac.set_index(['client_id','sku']).index.isin(sku_pb_set)
price_abandon_rate = sku_ac.groupby(['client_id','price_quartile'])['abandoned'].mean().unstack().add_prefix('abandon_rate_price_q')


purchase_rate = (pb / (ac + 1)).to_frame('purchase_rate')

# 2. Adds per search (how many adds each search yields)
carts_per_search = (ac / (sc + 1)).to_frame('carts_per_search')

# 3. Buys per search
buys_per_search = (pb / (sc + 1)).to_frame('buys_per_search')

# 4. Search abandonment: fraction of searches that led to no add
search_abandon = ((sc - ac) / (sc + 1)).to_frame('search_abandon_rate')

# 5. Log‐scaled counts to tame heavy tails
log_counts = pd.DataFrame({
    'log_add_to_cart':   np.log1p(ac),
    'log_product_buy':   np.log1p(pb),
    'log_search_query':  np.log1p(sc),
}, index=user_evt.index)

# 6. Binary conversion flags
conversion_flags = pd.DataFrame({
    'made_any_purchase':       (pb > 0).astype(int),
    'abandoned_any_cart':      ((ac - pb) > 0).astype(int),
    'searched_and_purchased':  ((sc > 0) & (pb > 0)).astype(int),
}, index=user_evt.index)

# 7. Simple counts
count_feats = pd.DataFrame({
    'add_to_cart_count': ac,
    'product_buy_count': pb,
    'search_count':      sc,
}, index=user_evt.index)


# browsing diversity
browse_div = sku.groupby('client_id')['category'].nunique().to_frame('browse_div')
# page visits per buy & event diversity
page_visit_per_buy = (user_evt.get('page_visit',0)/(pb+1)).to_frame('page_visit_per_buy')
event_div         = ev.groupby('client_id')['event_type'].nunique().to_frame('event_div')
# interaction depth
ev_sorted = ev.sort_values(['client_id','timestamp'])
ev_sorted['order'] = ev_sorted.groupby('client_id').cumcount()+1
depth_max = ev_sorted.groupby('client_id')['order'].max().to_frame('max_depth')
depth_avg = ev_sorted.groupby('client_id')['order'].mean().to_frame('avg_depth')
# recency & frequency
t_last    = ev.groupby('client_id')['timestamp'].max()
rec       = (MAX_TS-t_last).dt.days.to_frame('days_since_last')
t_first   = ev.groupby('client_id')['timestamp'].min()
lifetime  = (MAX_TS-t_first).dt.days.to_frame('lifetime_days')
freq      = ev.groupby('client_id').agg(total_events=('event_type','size'),active_days=('timestamp',lambda x: x.dt.date.nunique()))
freq['events_per_day']=freq['total_events']/(freq['active_days']+1e-6)
# decay counts
ev['decay_wt']=np.exp(-((MAX_TS-ev['timestamp']).dt.days)/7)
user_dec=ev.groupby(['client_id','event_type'])['decay_wt'].sum().unstack(fill_value=0).add_prefix('decay_')


daily_ev = ev.groupby(['client_id', ev['timestamp'].dt.date]).size()
activity_consistency = daily_ev.groupby('client_id').agg(lambda x: x.std() / (x.mean() + 1e-6)).to_frame('activity_consistency')
recent_window = 30
recent_freq = ev[ev['timestamp']>=MAX_TS-pd.Timedelta(days=recent_window)].groupby('client_id').size()
avg_daily_freq = freq['events_per_day']
recent_intensity = (recent_freq/recent_window) / (avg_daily_freq+1e-6)
recent_intensity = recent_intensity.to_frame('recent_activity_surge')
current_dormancy = rec['days_since_last']
dormancy_ratio = (current_dormancy / (lifetime['lifetime_days']+1e-6)).to_frame('dormancy_ratio')

from scipy.stats import linregress

def daily_activity_trend(df):
    daily_counts = df['timestamp'].dt.floor('D').value_counts().sort_index()
    if len(daily_counts)>1:
        x = daily_counts.index.astype(np.int64) // 1e9 / 86400  # days
        y = daily_counts.values
        return linregress(x,y).slope
    else:
        return 0

activity_momentum = ev.groupby('client_id').apply(daily_activity_trend).to_frame('daily_activity_slope')
peak_day_activity = daily_ev.groupby('client_id').max().to_frame('peak_day_activity')
from scipy.stats import skew, kurtosis

activity_distribution = daily_ev.groupby('client_id').agg(
    daily_activity_skew=lambda x: skew(x,bias=False) if len(x)>2 else 0,
    daily_activity_kurtosis=lambda x: kurtosis(x,bias=False) if len(x)>3 else 0
)

dow_ev = ev.groupby(['client_id', ev['timestamp'].dt.dayofweek]).size().unstack(fill_value=0)
weekly_pattern_strength = (dow_ev.max(axis=1)/dow_ev.sum(axis=1)).to_frame('weekly_pattern_strength')



# ─── CONCAT & LOG --------------------------------------------------------
blocks=[user_evt,user_sess_dur,
    user_inter_gap,       # inter-session gaps
    pod_frac,# time-of-day fractions
    evt_counts,
    distinct_sku, distinct_sku30,
    spend_30d, mon_spend, spend_ratio,
    recency_freq[['days_since_last','avg_events_per_day']],
    roll_var,
    hour_feats, dow_feats,
    cat_ent,
    sku_div,         # distinct_sku_count
    avg_gap,         # avg_event_gap
    price_var_df,    # price_var
    revisit_freq,    # sku_revisit_freq
    weekend_ratio,   # weekend_event_r,  # avg_item_popularity
    cart_abandon,       # your original cart‐abandonment rate
    search_conv,        # your original search‐to‐buy conversion
    purchase_rate,
    carts_per_search,
    buys_per_search,
    search_abandon,
    log_counts,
    conversion_flags,
    count_feats,
    cart_abandon,
    search_conv,
    repeat_add,
    immediate_conv,
    query_purchase_speed,
    abandoned_sku_diversity,
    search_per_day,
    price_abandon_rate,
    abandoned_sku_diversity,
    rec,
    lifetime,
    freq,
    activity_consistency,
    recent_intensity,
    dormancy_ratio,
    activity_momentum,
    peak_day_activity,
    activity_distribution,
    weekly_pattern_strength,
    user_daily,user_dow,queries_per_session,rate30,last_buy_days,user_hr,user_hour_cycle,user_pod,user_month,user_quarter,user_week,user_doy_cycle,peak,buy_q_frac,user_sess,sess_std,med_len,price_stats,user_cat,cat30,rec_div,user_name,cum,var,disc,sku_div,avg_gap,price_var_df,revisit_freq,weekend_ratio,user_pop,cart_abandon,search_conv,browse_div,page_visit_per_buy,event_div,depth_max,depth_avg,rec,lifetime,freq[['total_events','events_per_day']],user_dec]
user_feats=pd.concat(blocks,axis=1).fillna(0)
user_feats.replace([np.inf,-np.inf],0, inplace=True)
user_feats= user_feats.apply(np.log1p)
user_feats.replace([np.inf,-np.inf],0, inplace=True)
user_feats.fillna(0, inplace=True)
print('Final feature matrix:',user_feats.shape)


client_ids = user_feats.index.to_numpy(np.int64)
X = user_feats.values.astype(np.float32)
# Save compressed
np.savez_compressed('user_features.npz', X=X, client_ids=client_ids)
print("Saved compressed feature matrix to user_features.npz")

# ─── FINAL SAVE --------------------------------------------------------
# Only SVD embeddings (removing DeepFM)
np.save(OUTPUT_DIR/'client_ids.npy', user_feats.index.to_numpy(np.int64))
np.save(OUTPUT_DIR/'embeddings.npy', emb_svd)
print('Saved embeddings', emb_svd.shape, '→', OUTPUT_DIR)

# ─── SCALE & SVD --------------------------------------------------------
X_s = StandardScaler(with_mean=False).fit_transform(sparse.csr_matrix(user_feats.values)).astype(np.float32)
emb_svd = TruncatedSVD(n_components=SVD_DIM,random_state=42).fit_transform(X_s).astype(np.float16)
print('SVD embedding:',emb_svd.shape)

