from pathlib import Path
import json
import os
import re
import warnings
from .data import atomic_json

SYSTEM = """You verbalize model evidence, not the image. Return JSON with keys predicted_class,
confidence, supporting_concept_indices, opposing_concept_indices, explanation, limitations.
Copy the predicted class and confidence exactly. Select support only where margin_contribution>0,
and opposition only where margin_contribution<0. Describe concepts using their supplied names.
Concept indices are ZERO-BASED and must be copied from the supplied allowed index lists, not
inferred from the one-based C01/C02 display names. Use only the supplied selected_concepts.
Never infer medical diagnoses, bird parts, species attributes, pet anatomy, lesions, or causal
image mechanisms from low-level correlations. State weak grounding when present. The output
does not change the classifier. Give a concise explanation and acknowledge uncalibrated probability.
"""


def template_report(packet):
    prediction = packet['prediction']
    support = sorted([r for r in packet['concepts'] if r['margin_contribution'] > 0], key=lambda r: -r['margin_contribution'])[:3]
    oppose = sorted([r for r in packet['concepts'] if r['margin_contribution'] < 0], key=lambda r: r['margin_contribution'])[:3]
    explanation = f"The model predicted {prediction['class']} with an uncalibrated probability of {prediction['probability']:.3f}; the runner-up was {prediction['runner_up']}. "
    explanation += 'Supporting model patterns: ' + ('; '.join(f"{r['name']} (margin contribution {r['margin_contribution']:+.4f})" for r in support) or 'no positive concept terms; inspect the classifier bias.')
    return {'predicted_class': prediction['class'], 'confidence': prediction['probability'], 'supporting_concept_indices': [r['concept_index'] for r in support], 'opposing_concept_indices': [r['concept_index'] for r in oppose], 'explanation': explanation, 'limitations': packet['scope']}


def validate_report(value, packet):
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
    # This validates structure, numbers, indices and signs, not all semantic claims.
    return value


def generate_reports(run, assets, backend='template', max_cases=3, llm_device='cpu'):
    run, assets = Path(run), Path(assets)
    packets = json.loads((run / 'example_evidence.json').read_text())[:max_cases]
    folder = run / 'reports'; folder.mkdir(exist_ok=True)
    tokenizer, llm = None, None
    client = None
    if backend == 'local_qwen':
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
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
                prompt = json.dumps({'prediction': packet['prediction'], 'exact_decomposition': packet['exact_decomposition'],
                                     'allowed_supporting_concept_indices': fixed['supporting_concept_indices'],
                                     'allowed_opposing_concept_indices': fixed['opposing_concept_indices'],
                                     'selected_concepts': [r for r in packet['concepts'] if r['concept_index'] in allowed],
                                     'scope': packet['scope']})
                if backend == 'local_qwen':
                    import torch
                    text = tokenizer.apply_chat_template([{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True)
                    inputs = tokenizer(text, return_tensors='pt').to(llm_device)
                    with torch.inference_mode():
                        output = llm.generate(**inputs, max_new_tokens=600, do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=tokenizer.eos_token_id)
                    raw = tokenizer.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                else:
                    response = client.models.generate_content(model=os.getenv('GEMINI_MODEL', 'gemini-2.5-flash'), contents=SYSTEM + '\n' + prompt, config={'response_mime_type': 'application/json', 'temperature': 0})
                    raw = response.text
                clean = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw.strip())
                result = validate_report(json.loads(clean), packet)
            except Exception as exc:
                fallback = type(exc).__name__ + ': ' + str(exc)[:200]
                warnings.warn('LLM response failed validation; using the deterministic evidence template.')
                result = template_report(packet)
        else:
            result = template_report(packet)
        record = {'backend_requested': backend, 'backend_used': 'template' if fallback else backend, 'fallback_reason': fallback, 'structured_checks': 'passed', 'semantic_validation': 'human review required; JSON checks do not guarantee hallucination-free prose', 'result': result}
        atomic_json(folder / f'case_{case:03d}.json', record)
        (folder / f'case_{case:03d}.md').write_text(f"# Model evidence explanation\n\n{result['explanation']}\n\n{result['limitations']}\n\nBackend: {record['backend_used']}. Prose requires review.\n", encoding='utf-8')
        if raw is not None:
            (folder / f'case_{case:03d}_raw_llm.txt').write_text(raw, encoding='utf-8')
        records.append(record)
    return records
