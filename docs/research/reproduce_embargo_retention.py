"""Reproduce a training-only diagnostic for an audited Best-60 run.

Run from the repository with PYTHONPATH=. and the matching frozen source code.
This diagnostic does not read locked test outcomes or establish empirical power.
"""
import argparse
import hashlib
import json
from pathlib import Path
from src.ingestion.best60_research.corpus import extract_corpus
from src.ingestion.best60_research.context import build_context

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,help='Audited run directory')
    parser.add_argument('--database',required=True,help='Matching read-only SQLite source')
    args=parser.parse_args()
    run=Path(args.run)
    m=json.loads((run/'manifest.json').read_text())
    for relative, expected in m['code']['files'].items():
        if hashlib.sha256(Path(relative).read_bytes()).hexdigest() != expected:
            raise ValueError('Source code bytes differ from the audited run: '+relative)
    ids=[event for name in ('development','validation') for event in m['splits'][name]['event_ids']]
    corpus=extract_corpus(args.database,include_event_ids=ids)
    if corpus['audit']['source_sha256'] != m['source_sha256'] or corpus['audit']['snapshot_fingerprint'] != m['snapshot_fingerprint']:
     raise ValueError('Source snapshot differs from the audited run')
    cutoff=min(m['splits']['test']['dates'].values())
    result={'scope':'training_design_sensitivity_only','test_outcomes_read':False,'predictive_sensitivity':'not_measured_by_this_audit','source_sha256':m['source_sha256'],'cutoff':cutoff,'code_revision':m['code']['revision'],'analysis_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'embargoes':{}}
    for days in (7,14):
     context=build_context(corpus,cutoff=cutoff,embargo_days=days,min_slot_lists=30)
     rows=context['eligible_views'];focal=[r for r in rows if r['archetype_id']=='kangaskhan-bouffalant']
     lists=[r for r in context['training_lists'] if r['archetype_id']=='kangaskhan-bouffalant']
     result['embargoes'][str(days)]={'training_event_count':len(context['training_event_ids']),'pooled_views':len(rows),'pooled_unique_pairings':len({tuple(r['pairing_key']) for r in rows}),'global_copy_slots':len(context['slot_names']),'focal_views':len(focal),'focal_unique_pairings':len({tuple(r['pairing_key']) for r in focal}),'focal_list_count':len(lists),'catalogue_lists':len(context['catalogue'])}
    path=run/'embargo-training-design-sensitivity.json';path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__ == "__main__":
    main()
