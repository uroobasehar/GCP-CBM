from pathlib import Path
import json
import os
import re
import warnings
from .data import atomic_json

SYSTEM = """You are a bounded classifier-evidence verbalizer. Rewrite the supplied evidence statements
into a short explanation; preserve BOTH descriptor names and their support/opposition roles.
You may copy the statements. They contain all justifiable facts. Do not add physical traits,
class-average comparisons, clinical interpretation, numerical claims or mechanisms.
Class names are outcomes, never concepts. Return ONLY JSON with exactly two STRING keys:
explanation and limitations. For limitations preserve the supplied limitations statement,
including 'unverified' when present. Never use arrays. Keep the total below 150 words.
"""


def template_report(packet):
    prediction = packet['prediction']
    support = sorted([r for r in packet['concepts'] if r['margin_contribution'] > 0], key=lambda r: -r['margin_contribution'])[:3]
    oppose = sorted([r for r in packet['concepts'] if r['margin_contribution'] < 0], key=lambda r: r['margin_contribution'])[:3]
    explanation = f"The model predicted {prediction['class']} with an uncalibrated probability of {prediction['probability']:.3f}; the runner-up was {prediction['runner_up']}. "
    explanation += 'Supporting model patterns: ' + ('; '.join(f"{r['name']} (margin contribution {r['margin_contribution']:+.4f})" for r in support) or 'no positive concept terms; inspect the classifier bias.')
    explanation += '. Opposing model patterns: ' + ('; '.join(f"{r['name']} ({r['margin_contribution']:+.4f})" for r in oppose) or 'none among the selected concepts.')
    measured = [f"{r['name']}: {r['image_descriptor_evidence']['relative_to_validation']} descriptor response relative to validation" for r in support if r.get('image_descriptor_evidence')]
    if measured: explanation += '. Image measurements: ' + '; '.join(measured)
    explanation += f". The classifier bias contributes {packet['exact_decomposition']['margin_bias']:+.4f} to this logit margin."
    if any(r['weak_grounding'] for r in support+oppose):
        explanation += ' Some cited patterns have unverified grounding; their intended guidance must not be reported as established anatomy or a learned clinical finding.'
    return {'predicted_class': prediction['class'], 'confidence': prediction['probability'], 'supporting_concept_indices': [r['concept_index'] for r in support], 'opposing_concept_indices': [r['concept_index'] for r in oppose], 'explanation': explanation, 'limitations': packet['scope']}


def validate_report(value, packet, prose_only=False):
    if not isinstance(value, dict) or any(k not in value for k in ['predicted_class', 'confidence', 'supporting_concept_indices', 'opposing_concept_indices', 'explanation', 'limitations']):
        raise ValueError('LLM JSON is missing required fields.')
    if value['predicted_class'] != packet['prediction']['class'] or abs(float(value['confidence']) - packet['prediction']['probability']) > 1e-6:
        raise ValueError('LLM changed the prediction or numerical confidence.')
    concepts = {r['concept_index']: r for r in packet['concepts']}
    for field, sign in [('supporting_concept_indices', 1), ('opposing_concept_indices', -1)]:
        if not isinstance(value[field], list): raise ValueError('Invalid concept index list.')
        for index in value[field]:
            if not isinstance(index, int) or index not in concepts or concepts[index]['margin_contribution'] * sign <= 0:
                raise ValueError('LLM used an unsupported concept or sign.')
    if not isinstance(value['explanation'], str) or not isinstance(value['limitations'], str):
        raise ValueError('Invalid prose fields.')
    prose=(value['explanation']+' '+value['limitations']).lower()
    if prose_only and re.search(r'\b(higher|lower|sharper|brighter|darker|more distinct)\b.{0,180}\b(compared to|than)\b',value['explanation'].lower()):
        raise ValueError('LLM invented a descriptor comparison between classes or images.')
    if prose_only and re.search(r'\b\d+\.\d+\b|\d\s*%',prose):
        raise ValueError('LLM added numerical claims to qualitative prose.')
    if prose_only and re.search(r'\b(higher|lower|increases?|decreases?|sharper|brighter|darker)\b',value['explanation'].lower()):
        raise ValueError('LLM inferred physical descriptor levels/changes from model contribution signs.')
    if str(packet['prediction']['class']).lower()+' concept' in prose:
        raise ValueError('LLM conflated a predicted class with a learned descriptor concept.')
    # Required semantic stems, rather than opaque numeric concept identifiers, must be named.
    for field in ['supporting_concept_indices','opposing_concept_indices']:
        ranked=[r for r in packet['concepts'] if r['concept_index'] in value[field]]
        if ranked:
            strongest=max(ranked,key=lambda r:abs(r['margin_contribution']))
            stem=strongest['name'].split(' —')[0].split(' ',1)[-1].lower()
            if stem not in value['explanation'].lower():
                raise ValueError('LLM omitted or renamed the strongest supporting/opposing descriptor.')
    clinical_prose=value['explanation'].lower()
    for class_name in [packet['prediction']['class'],packet['prediction']['runner_up']]:
        clinical_prose=clinical_prose.replace(str(class_name).lower(),'').replace(str(class_name).replace('_',' ').lower(),'')
    if prose_only and re.search(r'\b(diagnosis|malignan\w*|benign\w*|tumou?rs?|lesions?|disease|fur|plumage|anatomy)\b',clinical_prose):
        raise ValueError('LLM converted a model class prediction into a clinical diagnosis.')
    if re.search(r'(tumou?r|lesion)\s+(boundary|margin|shape|compactness)|necrosis|contrast enhancement|asymmetric lesion',prose):
        raise ValueError('LLM inferred a clinical/lesion property not established by image-descriptor evidence.')
    selected=set(value['supporting_concept_indices']+value['opposing_concept_indices'])
    weak=any(r['weak_grounding'] for r in packet['concepts'] if r['concept_index'] in selected)
    if prose_only and weak and re.search(r'\b(strong guidance|well-established|clinically proven)\b',value['explanation'].lower()):
        raise ValueError('LLM overstated unverified grounding as established guidance.')
    if weak and not any(w in prose for w in ['weak','unverified','not established']):
        raise ValueError('LLM omitted the required weak-grounding qualification.')
    # This validates structure, numbers, indices and signs, not all semantic claims.
    return value


def write_report_markdown(folder, case, packet, record):
    """Render numerical evidence from the classifier beside the separately generated prose."""
    result=record['result'];prediction=packet['prediction'];parts=packet['exact_decomposition']
    indices=result['supporting_concept_indices']+result['opposing_concept_indices']
    rows=[r for i in indices for r in packet['concepts'] if r['concept_index']==i]
    table='| Descriptor | Margin contribution | Grounding status |\n|---|---:|---|\n'
    table+=''.join(f"| {r['name']} | {r['margin_contribution']:+.4f} | {r.get('guidance_status','unrecorded')} |\n" for r in rows)
    text=(f"# Model evidence explanation\n\nImage: {packet.get('image_path','not recorded')}. Reference class: {packet.get('true_class','not recorded')}.\n\n"
          f"Classifier prediction: **{prediction['class']}**; runner-up: **{prediction['runner_up']}**. "
          f"Uncalibrated probability: {prediction['probability']:.4f}. Logit margin: {parts['margin']:+.4f}; separate bias: {parts['margin_bias']:+.4f}.\n\n"
          +table+f"\n{result['explanation']}\n\n{result['limitations']}\n\nBackend: {record['backend_used']}. "
          'Numerical fields and table come directly from the classifier. Generated prose requires review.\n')
    (Path(folder)/f'case_{case:03d}.md').write_text(text,encoding='utf-8')


def generate_reports(run, assets, backend='template', max_cases=3, llm_device='cpu'):
    run, assets = Path(run), Path(assets)
    # Visual panels and reports must refer to the same image, not unrelated first test batches.
    source = run / 'visual_example_evidence.json'
    if not source.is_file(): source = run / 'example_evidence.json'
    packets = json.loads(source.read_text())[:max_cases]
    folder = run / 'reports'; folder.mkdir(exist_ok=True)
    tokenizer, llm = None, None
    client = None
    if backend == 'local_qwen':
        # This verbalizer is PyTorch-only. Avoid importing optional TF/Flax and reserving
        # another framework's CUDA memory merely because requested libraries are installed.
        os.environ.setdefault('USE_TF', '0')
        os.environ.setdefault('USE_FLAX', '0')
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        if llm_device == 'auto': llm_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        path = assets / 'qwen2.5-1.5b-instruct'
        if not path.is_dir(): raise FileNotFoundError('Offline Qwen assets are missing. Run prepare_assets.py online.')
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        llm = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, torch_dtype=torch.float16 if llm_device == 'cuda' else torch.float32).to(llm_device).eval()
    elif backend == 'gemini':
        from google import genai
        key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
        if not key: raise RuntimeError('Set the API key securely in notebook secrets/environment settings; never put it in the notebook.')
        client = genai.Client(api_key=key)
    elif backend != 'template':
        raise ValueError('Use template, local_qwen, or gemini.')
    records = []
    for case, packet in enumerate(packets):
        raw = None; fallback = None
        if backend != 'template':
            try:
                fixed = template_report(packet)
                allowed = set(fixed['supporting_concept_indices'] + fixed['opposing_concept_indices'])
                # Arithmetic is already displayed by the renderer. Giving a small LLM many
                # numbers/indices invited spurious comparisons and incorrect copied evidence.
                def descriptor(indices, role):
                    if not indices: return None
                    r=next(r for r in packet['concepts'] if r['concept_index']==indices[0])
                    return {'name':r['name'],'role':role,'grounding_status':r.get('guidance_status')}
                supporting=descriptor(fixed['supporting_concept_indices'],'supports prediction over runner-up')
                opposing=descriptor(fixed['opposing_concept_indices'],'opposes prediction relative to runner-up')
                statements=[f"The model favours {packet['prediction']['class']} over {packet['prediction']['runner_up']}."]
                if supporting:statements.append(f"The {supporting['name']} pattern provides supporting model evidence for this margin.")
                else:statements.append('There are no positive concept margin terms; inspect the separate classifier bias.')
                if opposing:statements.append(f"The {opposing['name']} pattern provides opposing model evidence for this margin.")
                weak=any(r['weak_grounding'] for r in packet['concepts'] if r['concept_index'] in allowed)
                limitations=('Some cited descriptor associations are unverified. ' if weak else '')+'These are model-derived descriptor associations, not established anatomical or clinical findings. Probability is uncalibrated.'
                prompt = json.dumps({'evidence_statements':statements,'limitations_statement':limitations,
                                     'required_format':{'explanation':'Join/paraphrase the evidence statements; preserve both descriptor names and roles.',
                                                        'limitations':'Preserve limitations_statement as a string.'}})
                (folder / f'case_{case:03d}_llm_input.json').write_text(prompt,encoding='utf-8')
                if backend == 'local_qwen':
                    import torch
                    text = tokenizer.apply_chat_template([{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True)
                    inputs = tokenizer(text, return_tensors='pt').to(llm_device)
                    with torch.inference_mode():
                        output = llm.generate(**inputs, max_new_tokens=360, do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=tokenizer.eos_token_id)
                    raw = tokenizer.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                else:
                    response = client.models.generate_content(model=os.getenv('GEMINI_MODEL', 'gemini-2.5-flash'), contents=SYSTEM + '\n' + prompt, config={'response_mime_type': 'application/json', 'temperature': 0})
                    raw = response.text
                clean = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw.strip())
                prose = json.loads(clean)
                if not isinstance(prose,dict) or set(prose)!= {'explanation','limitations'}:
                    raise ValueError('LLM must return only explanatory prose; structured prediction fields come from the classifier.')
                result = validate_report({**fixed, **prose}, packet, prose_only=True)
            except Exception as exc:
                fallback = type(exc).__name__ + ': ' + str(exc)[:200]
                warnings.warn('LLM response failed validation; using the deterministic evidence template.')
                result = template_report(packet)
        else:
            result = template_report(packet)
        record = {'backend_requested': backend, 'backend_used': 'template' if fallback else backend, 'fallback_reason': fallback, 'structured_checks': 'passed', 'semantic_validation': 'human review required; JSON checks do not guarantee hallucination-free prose', 'result': result}
        record.update(structured_fields_source='deterministic classifier evidence; LLM supplies prose only',image_path=packet.get('image_path'),true_class=packet.get('true_class'),
                      evidence_source=source.name,figure_prefix=packet.get('figure_prefix'))
        atomic_json(folder / f'case_{case:03d}.json', record)
        write_report_markdown(folder,case,packet,record)
        if raw is not None:
            (folder / f'case_{case:03d}_raw_llm.txt').write_text(raw, encoding='utf-8')
        records.append(record)
    return records
