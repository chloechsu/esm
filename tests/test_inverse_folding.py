def test_esm_if1():
    import esm
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.eval()
    _test(model, alphabet)


def test_esm_if_gvpgnnlarge():
    import esm
    model, alphabet = esm.pretrained.esm_if_gvpgnnlarge_gvp16_21M_UR50()
    model = model.eval()
    _test(model, alphabet)


def _test(model, alphabet):

    import json
    import numpy as np
    from pathlib import Path
    from scipy.stats import special_ortho_group
    from tqdm import tqdm
    import torch
    import esm.inverse_folding
    import esm

    batch_converter = esm.inverse_folding.util.CoordBatchConverter(alphabet)

    example_file = Path(__file__).absolute().parent / "inverse_folding_test_example.json"
    with open(example_file) as f:
        examples = json.load(f)

    with torch.no_grad():
        print('Testing batch inference on 3 examples...')
        # Test batch with multiple examples
        batch = [(e["coords"], None, e["seq"]) for e in examples[:3]]
        coords, confidence, strs, tokens, padding_mask = (
            batch_converter(batch)
        )
        prev_output_tokens = tokens
        logits, _ = model.forward(coords, padding_mask, confidence,
                prev_output_tokens)
        # Remove bos and eos
        target = tokens[:, 1:-1]
        # Shift and remove eos
        logits = logits[:, :, :-2]
        loss = torch.nn.functional.cross_entropy(logits, target, reduction='none')
        coord_mask = torch.all(torch.all(torch.isfinite(coords), dim=-1), dim=-1)
        # Remove bos and eos
        coord_mask = coord_mask[:, 1:-1]
        avgloss_per_example = torch.sum(loss * coord_mask, dim=1) / torch.sum(
            coord_mask, dim=1)
        ppl_per_example = torch.exp(avgloss_per_example).cpu().numpy()
        expected_ppl = [4.5, 4.5, 4.5]
        np.testing.assert_allclose(
            expected_ppl,
            ppl_per_example,
            atol=1.,
        )

        print('Testing on 10 examples from validation set...')
        # Test batch with single example
        for example in tqdm(examples):
            batch = [(example["coords"], None, example["seq"])]
            coords, confidence, strs, tokens, padding_mask = (
                batch_converter(batch)
            )
            prev_output_tokens = tokens
            target = tokens[:, 1:-1]
            logits, _ = model.forward(coords, padding_mask, confidence,
                    prev_output_tokens)
            assert torch.any(torch.isnan(logits)) == False

            # Test equivariance
            R = special_ortho_group.rvs(3)
            R = torch.tensor(R, dtype=torch.float32)
            coords = torch.matmul(coords, R)
            logits_rotated, _ = model.forward(coords, padding_mask,
                    confidence, prev_output_tokens)
            np.testing.assert_allclose(
                    logits.detach().numpy(), 
                    logits_rotated.detach().numpy(), 
                    atol=1e-01
            )
