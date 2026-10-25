"""Reproduce a training-only diagnostic for an audited Best-60 run.

Run from the repository with PYTHONPATH=. and the matching frozen source code.
This diagnostic does not read locked test outcomes or establish empirical power.
"""
import argparse
import hashlib,json,subprocess
from pathlib import Path
import numpy as np
from scipy import sparse
from src.ingestion.best60_research.corpus import extract_corpus
from src.ingestion.best60_research.context import build_context
from src.ingestion.best60_research.experiment import _design,_training_slots,_view_weights,_event_block
from src.ingestion.best60_research.power import interaction_information

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,help='Audited run directory')
    parser.add_argument('--database',required=True,help='Matching read-only SQLite source')
    args=parser.parse_args()
    run=Path(args.run);m=json.loads((run/'manifest.json').read_text())
    for relative, expected in m['code']['files'].items():
        if hashlib.sha256(Path(relative).read_bytes()).hexdigest() != expected:
            raise ValueError('Source code bytes differ from the audited run: '+relative)
    ids=[e for name in ('development','validation') for e in m['splits'][name]['event_ids']]
    corpus=extract_corpus(args.database,include_event_ids=ids)
    if corpus['audit']['source_sha256'] != m['source_sha256'] or corpus['audit']['snapshot_fingerprint'] != m['snapshot_fingerprint']:
     raise ValueError('Source snapshot differs from the audited run')
    context=build_context(corpus,cutoff=min(m['splits']['test']['dates'].values()),embargo_days=7,min_slot_lists=30)
    rows=[r for r in context['eligible_views'] if r['archetype_id']=='kangaskhan-bouffalant']
    focal_lists=[r for r in context['training_lists'] if r['archetype_id']=='kangaskhan-bouffalant']
    local=dict(context);local['slot_names']=_training_slots({'training_lists':focal_lists},set(context['training_event_ids']),30)
    X,y,nuisance,groups,schema=_design(rows,local,training=True)
    events=sorted({r['event'] for r in rows});lookup={e:i for i,e in enumerate(events)}
    event_columns=sparse.csr_matrix((np.ones(len(rows)),(np.arange(len(rows)),[lookup[r['event']] for r in rows])),shape=(len(rows),len(events)))[:,1:]
    z=sparse.hstack([nuisance,event_columns,X],format='csr')
    slots=local['slot_names'];weights=_view_weights(rows,focal=True);probabilities=np.full(len(rows),.5);clusters=np.array([_event_block(context,r) for r in rows])
    base={'scope':'local_optimistic_conditional_information','source_sha256':m['source_sha256'],'code_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'test_outcomes_read':False,'cutoff':context['cutoff'],'embargo_days':7,'observed_views':len(rows),'unique_pairings':len({tuple(r['pairing_key']) for r in rows}),'weight_sum':float(weights.sum()),'supported_copy_slots':len(slots),'event_count':len(events),'utc_date_block_count':int(len(set(clusters))),'probability_assumption':.5,'controls':'all supported focal main effects, frozen metadata/skill controls, event indicators; no prior precision','cluster_caution':'cluster count is descriptive; information does not estimate dependence inflation or empirical detection power','results':{}}
    lumiose=slots.index(('Lumiose City',1))
    columns=[j for j in range(z.shape[1]) if j!=nuisance.shape[1]+event_columns.shape[1]+lumiose]
    base['results']['Lumiose City main copy 1']=interaction_information(z[:,columns],X[:,lumiose],probabilities=probabilities,sample_weight=weights,cluster_ids=clusters)
    candidates=['Ultra Ball','Nest Ball','Mist Energy',"Boss's Orders",'Iono',"Professor's Research",'Switch','Night Stretcher','Energy Retrieval']
    base['unsupported_candidate_names']=[card for card in candidates if not any(name==card for name,copy in slots)]
    base['unsupported_candidate_reason']='No copy threshold meets the frozen 30-list presence and absence screen'
    for j,(card,copy) in enumerate(slots):
     if card not in candidates:continue
     exposure=X[:,lumiose].multiply(X[:,j])
     base['results'][f'Lumiose City copy 1 x {card} copy {copy}']=interaction_information(z,exposure,probabilities=probabilities,sample_weight=weights,cluster_ids=clusters)
    base['analysis_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    path=run/'lumiose-training-information.json';path.write_text(json.dumps(base,indent=2)+'\n');print(json.dumps({'output':str(path),'terms':len(base['results'])}))

if __name__ == "__main__":
    main()
