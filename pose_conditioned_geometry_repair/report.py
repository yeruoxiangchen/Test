"""Write a small human-readable final report after the formal queue completes."""
from . import common as c


def main():
    summary=c.read(c.OUT/'formal_summary.json')
    gate=c.read(c.OUT/'overfit_gate.json')
    lines=['# Round 1: trainable pose-conditioned native T2 repair','',c.now(),'',
        'Synthetic interventions, known ROI, UID-disjoint Dev64. No ICP or per-method GT fit.',
        'Teacher-support and predicted-SS metrics are separated. This is not an automatic-localization result.','',
        'Overfit32 gate: '+str(gate['passed']),'',
        '| Arm / seed | CD (predicted SS) | F@.01 | F@.02 | CD gain vs damaged |',
        '|---|---:|---:|---:|---:|']
    for name,row in summary['summaries'].items():
        m=row['means']['candidate_predicted_ss']
        lines.append(f'| {name} | {m["cd_l1"]:.6f} | {m["f@0.01"]:.6f} | {m["f@0.02"]:.6f} | {row["predicted_ss_mesh_cd_gain_vs_damaged"]:.2%} |')
    lines+=['','All per-object metrics, failure records and OBJ outputs remain in their evaluation directories.','']
    (c.OUT/'REPORT.md').write_text('\n'.join(lines))
    c.write(c.OUT/'report_receipt.json',{'time':c.now(),'complete':True,'summary':str(c.OUT/'formal_summary.json')})


if __name__=='__main__':
    main()
