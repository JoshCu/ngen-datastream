import pandas as pd
import concurrent.futures as cf
import json
import os, time

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset

def get_weight_json(catchments,jproc):
    
    weight_data = {}
    ncatch = len(catchments)

    w = pa.dataset.dataset(
        's3://lynker-spatial/v20.1/forcing_weights.parquet', format='parquet'
    ).filter(
        pc.field('divide_id').isin(catchments)
    ).to_batches()
    batch: pa.RecordBatch
    for batch in w:
        tbl = batch.to_pandas()
        if tbl.empty:
            continue    
    
        for j, jcatch in enumerate(catchments):
            t0 = time.perf_counter()
            df_jcatch = tbl.loc[tbl['divide_id'] == jcatch]
            if df_jcatch.empty:
                continue  
            weight_data[jcatch] = [[int(x) for x in list(df_jcatch['cell'])],list(df_jcatch['coverage_fraction'])]
            if len(list(df_jcatch['cell'])) == 0: print(f'{j} {jcatch} {df_jcatch} {tbl.empty}')
            t1 = time.perf_counter() - t0
            if j % 100 == 0 and jproc == 0: print(f'{j} {jcatch} {100*j/ncatch:.2f}% {t1*ncatch/3600:.2f}')

    return (weight_data)

uri = "s3://lynker-spatial/v20.1/forcing_weights.parquet"
weights_df = pd.read_parquet(uri)
catchment_list = list(weights_df.divide_id.unique())
del weights_df

nprocs = os.cpu_count() - 2
print(nprocs)
catchment_list_list = []
ncatchments = len(catchment_list)
nper = ncatchments // nprocs
nleft = ncatchments - (nper * nprocs)
i = 0
k = 0
for _ in range(nprocs):
    k = nper + i + nleft      
    catchment_list_list.append(catchment_list[i:k])
    i = k
    
with cf.ProcessPoolExecutor(max_workers=nprocs) as pool:
    results = pool.map(get_weight_json, catchment_list_list,[x for x in range(nprocs)])

# Aggregate results
weights = {}
for jweights in results:
    weights = weights | jweights

print(len(weights))

data = json.dumps(weights)
with open('./weights_conus_v21.json','w') as fp:
    fp.write(data)
    