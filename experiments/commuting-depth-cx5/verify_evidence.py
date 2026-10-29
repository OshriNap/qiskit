"""Check the published evidence and code bytes; does not prove circuit correctness."""
import csv,gzip,hashlib,json,statistics
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
def read(name):return json.loads((HERE/name).read_text())
def main():
    hashes=read('SHA256SUMS.json')
    for name,expected in hashes.items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==expected,name
    provenance=read('provenance.json')
    for name,expected in provenance['published_source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    summary=read('confirm-summary.json')
    decision=json.loads(gzip.decompress((HERE/'confirm-decision.json.gz').read_bytes()))
    assert summary['status']==decision['status']=='CONSTRAINT_VIOLATION'
    assert summary['objective']==decision['objective']==[]
    with gzip.open(HERE/'confirm-paired-metrics.csv.gz','rt',newline='') as f:rows=list(csv.DictReader(f))
    assert len(rows)==15600==summary['paired_keys']
    assert len({(r['case_id'],r['seed']) for r in rows})==len(rows)
    for r in rows:
        for key in ('D2_before','D2_after','N2_before','N2_after'):r[key]=int(r[key])
    for m in ('D2','N2'):
        a,b=m+'_before',m+'_after'
        counts={'wins':sum(r[b]<r[a] for r in rows),'ties':sum(r[b]==r[a] for r in rows),'losses':sum(r[b]>r[a] for r in rows)}
        assert counts==summary['counts'][m],(m,counts)
    groups={g['case_id']:g for g in read('confirm-case-means.json')}
    for cid in {r['case_id'] for r in rows}:
        sample=[r for r in rows if r['case_id']==cid]
        assert len(sample)==groups[cid]['pairs']
        for key,expected in groups[cid]['means'].items():assert abs(statistics.mean(r[key] for r in sample)-expected)<1e-10
    for stage in ('correctness','cost'):assert summary['stages'][stage]['status']=='skipped'
    print(json.dumps({'evidence_files_verified':len(hashes),'code_files_verified':len(provenance['source_sha256']),
        'metric_pairs':len(rows),'counts':summary['counts'],'official_verdict':decision['status'],
        'limitation':'Consistency check only; no circuits or benchmarks were rerun.'},indent=2))

if __name__=='__main__':main()
