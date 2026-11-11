"""Spawn-safe regression worker; never imports or touches Tk widgets."""
import multiprocessing as mp
import queue
import time


def regression_worker(path,initial_limit,updates):
    try:
        from model import read_excel,train
        last=0.0
        def report(done,total,message):
            nonlocal last
            now=time.monotonic()
            if done<total and now-last<.05:return
            last=now
            try:updates.put_nowait(('progress',done,total,message))
            except queue.Full:pass
        data=read_excel(path)
        bundle=train(data,progress=report,wape_limit_percent=initial_limit,auto_relax_wape=True)
        updates.put(('result',data,bundle))
    except Exception as e:
        updates.put(('error',str(e)))


def start_regression(path,initial_limit):
    ctx=mp.get_context('spawn');updates=ctx.Queue(maxsize=64)
    process=ctx.Process(target=regression_worker,args=(str(path),initial_limit,updates),daemon=True)
    process.start()
    return process,updates


def verify_worker(output):
    """Source and frozen smoke test for worker startup, IPC and complete regression."""
    from pathlib import Path
    import tempfile
    import numpy as np
    import pandas as pd
    with tempfile.TemporaryDirectory() as tmp:
        rng=np.random.default_rng(83);n=30
        df=pd.DataFrame({'순번':[str(i+1) for i in range(n)],'함종':['호위함']*n,'함유형':['FFG']*n,
                         '취역연도':rng.integers(1980,2020,n),'길이':rng.uniform(80,150,n),'폭':rng.uniform(10,20,n),
                         '깊이':rng.uniform(6,12,n),'소요마력':rng.uniform(10000,50000,n),'소요전력':rng.uniform(1000,9000,n),
                         '순항속력':rng.uniform(12,20,n)})
        for i in range(100,800,100):df[f'SWBS{i}']=i+(df['길이']*df['폭']*.1)
        df['총중량']=df[[f'SWBS{i}' for i in range(100,800,100)]].sum(axis=1)
        patrol=df.iloc[[0]].copy();patrol['함종']='초계함';patrol['순번']='P1'
        df=pd.concat([df,patrol],ignore_index=True)
        path=Path(tmp)/'fleet.xlsx';df.to_excel(path,index=False)
        process,updates=start_regression(path,10);deadline=time.monotonic()+180;progress=False
        try:
            while time.monotonic()<deadline:
                try:msg=updates.get(timeout=.2)
                except queue.Empty:
                    if not process.is_alive():raise RuntimeError(f'Worker exited without result: {process.exitcode}')
                    continue
                if msg[0]=='progress':progress=True
                elif msg[0]=='error':raise RuntimeError(msg[1])
                elif msg[0]=='result':
                    if not progress or len(msg[2]['models'])!=7:raise RuntimeError('Incomplete regression output')
                    if any(m['class_counts'].get('초계함')!=1 or abs(m['diagnostics'][-1]['오차율(%)'])>1e-7 for m in msg[2]['models'].values()):raise RuntimeError('Patrol model did not survive worker transfer')
                    Path(output).write_text('spawned regression worker verified',encoding='utf-8');return
            raise TimeoutError('Regression worker timeout')
        finally:
            process.join(timeout=5)
            if process.is_alive():process.terminate();process.join(timeout=5)
            updates.close()
