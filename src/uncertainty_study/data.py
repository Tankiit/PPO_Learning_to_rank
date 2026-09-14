"""Human-only DS adaptation, question folds, and paired information interventions."""
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
import json
import math
import numpy as np
from src.arr.schema import Candidate, RankingGroup
from src.arr.utils import stable_hash


def read_groups(path):
    return [RankingGroup.from_dict(json.loads(line)) for line in Path(path).read_text().splitlines() if line.strip()]


def write_groups(path, groups):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(g.to_dict()) + '\n' for g in groups))


def audit_ds(paths):
    """Audit all supplied files before selecting human-scored observations.

    Merge identical generation records and unique annotation payloads, retaining
    all source locations. Same text from distinct generators remains distinct.
    Reject inconsistent records; never silently pick a conflicting version.
    """
    unique, ids, qtexts, text_qids = {}, {}, {}, defaultdict(set)
    counts = Counter()
    for path in sorted(map(Path, paths)):
        for line, raw in enumerate(path.read_text().splitlines(), 1):
            if not raw.strip():
                continue
            r = json.loads(raw)
            counts['source_rows'] += 1
            qid = str(r['qid'])
            question = ' '.join(r['question'].split())
            if qid in qtexts and qtexts[qid] != question:
                raise ValueError(f'Conflicting question text for {qid}')
            qtexts[qid] = question
            text_qids[question].add(qid)
            generator = r['student_model'].lower().replace('llama2', 'llama-2')
            key = (qid, generator, r['student_prompt'], ' '.join(r['student_explanation'].split()))
            rid = str(r['id'])
            if rid in ids and ids[rid] != key:
                raise ValueError(f'Conflicting generation with source id {rid}')
            ids[rid] = key
            loc = {'file': str(path), 'line': line, 'source_id': rid}
            if key not in unique:
                unique[key] = {'row': r, 'sources': [], 'annotations': {}}
            else:
                counts['duplicate_generation_rows'] += 1
                if unique[key]['row']['dataset'] != r['dataset']:
                    raise ValueError(f'Conflicting domain for {rid}')
            item = unique[key]
            item['sources'].append(loc)
            for ann in r.get('explanation_annotations', []):
                score = float(ann['explanation_score'])
                if not 0 <= score <= 5:
                    raise ValueError(f'Invalid human score for {rid}')
                ann_key = json.dumps(ann, sort_keys=True)
                item['annotations'][ann_key] = ann
    # Canonicalize exact question aliases so they cannot cross folds.
    canonical = {qid: min(qids) for qids in text_qids.values() for qid in qids}
    byq = defaultdict(list)
    for key, item in unique.items():
        if not item['annotations']:
            counts['unscored_generations_excluded'] += 1
            continue
        r = item['row']
        anns = list(item['annotations'].values())
        byq[canonical[key[0]]].append((r, Candidate(
            stable_hash(*key), r['student_explanation'].strip(),
            float(np.mean([a['explanation_score'] for a in anns]) / 5),
            'DS_Critique_Bank.explanation_annotations.human_mean',
            {'generator': key[1], 'prompt': key[2], 'annotations': anns,
             'sources': item['sources'], 'source_qid': key[0],
             'length_words': len(r['student_explanation'].split())})))
    groups = []
    for qid, pairs in sorted(byq.items()):
        if len(pairs) < 2:
            counts['singleton_questions_excluded'] += 1
            continue
        candidates = tuple(sorted((c for _, c in pairs), key=lambda c: c.candidate_id))
        groups.append(RankingGroup(stable_hash('ds-adaptation-v1', qid), 'unassigned',
            pairs[0][0]['dataset'], pairs[0][0]['question'], candidates,
            stable_hash([c.to_dict() for c in candidates]), {'qid': qid, 'human_only': True}))
    support = Counter()
    for g in groups:
        for generator in set(c.metadata['generator'] for c in g.candidates):
            support[generator] += 1
    counts.update(ranking_questions=len(groups), human_candidates=sum(len(g.candidates) for g in groups))
    return groups, {'counts': dict(counts), 'question_support': dict(support),
        'question_aliases': [sorted(v) for v in text_qids.values() if len(v) > 1],
        'input_hashes': {str(p): stable_hash(Path(p).read_text()) for p in paths}}


def folds(groups, seed=2026, validation_fraction=.15):
    if len(groups) < 9 or not 0 < validation_fraction < 1:
        raise ValueError('Need at least nine questions and a nonzero validation fraction')
    if len({g.group_id for g in groups}) != len(groups):
        raise ValueError('Duplicate question groups')
    rng = np.random.default_rng(seed)
    # Stratified round-robin assignment; fold sizes remain balanced globally.
    buckets = [[], [], []]
    bydomain = defaultdict(list)
    for g in sorted(groups, key=lambda g: g.group_id):
        bydomain[g.domain].append(g)
    offset = 0
    for domain in sorted(bydomain):
        pool = bydomain[domain]
        for j in rng.permutation(len(pool)):
            buckets[offset % 3].append(pool[j]); offset += 1
    out = []
    for f in range(3):
        trainpool = [g for k, b in enumerate(buckets) if k != f for g in b]
        order = rng.permutation(len(trainpool))
        nv = max(1, round(validation_fraction * len(trainpool)))
        out.append({s: [replace(g, split=s) for g in gs] for s, gs in {
            'test': buckets[f], 'validation': [trainpool[i] for i in order[:nv]],
            'train': [trainpool[i] for i in order[nv:]]}.items()})
    return out


def nested_dose(groups, fraction, seed):
    if not 0 < fraction <= 1 or not groups:
        raise ValueError('Invalid dose or empty training pool')
    ordered = sorted(groups, key=lambda g: stable_hash(seed, g.group_id))
    return ordered[:max(1, math.ceil(len(ordered) * fraction))]


def exposure(groups, target, level, seed=2026):
    """Paired replacements: one target slot per supported prompt, two controls.

    All levels use the same eligible questions and three candidates per prompt.
    A target replaces one control; the other two controls stay fixed. Partial
    replaces a nested, seeded half of slots. Domain/question/prompt are exact
    matches. Length and quality balance are audited, not falsely guaranteed.
    """
    if level not in ('absent', 'partial', 'full'):
        raise ValueError('Unknown exposure level')
    planned, slots = [], []
    for g in groups:
        byprompt = defaultdict(list)
        for c in g.candidates:
            byprompt[c.metadata['prompt']].append(c)
        chosen = []
        for prompt, cs in sorted(byprompt.items()):
            target_cs = sorted([c for c in cs if c.metadata['generator'] == target], key=lambda c: stable_hash(seed, c.candidate_id))
            other = sorted([c for c in cs if c.metadata['generator'] != target], key=lambda c: stable_hash(seed, c.candidate_id))
            if not target_cs or len(other) < 3:
                continue
            slot = (g.group_id, prompt)
            slots.append(slot)
            chosen.append((slot, target_cs[0], other[:3]))
        if chosen:
            planned.append((g, chosen))
    active = set(sorted(slots, key=lambda x: stable_hash(seed, x))[:len(slots)//2])
    result = []
    for g, chosen in planned:
        cs = []
        for slot, target_c, other in chosen:
            use = level == 'full' or (level == 'partial' and slot in active)
            cs.extend([target_c if use else other[0], *other[1:]])
        result.append(replace(g, candidates=tuple(cs), data_fingerprint=stable_hash([c.to_dict() for c in cs])))
    return result
