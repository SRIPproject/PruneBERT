from __future__ import annotations

import json
import sys
from typing import List

import click

from config import PipelineConfig, EmbeddingConfig, ThresholdConfig, RelevanceConfig, ThresholdStrategy, RelevanceStrategy
from preprocess import preprocess_text
from embeddings import get_model, encode_sentences, compute_central_embedding
from relevance import compute_similarity_scores, identify_irrelevant_sentences
from threshold import calculate_threshold


@click.group()
def cli() -> None:
    pass


def _load_lines(input_file: str | None) -> List[str]:
    if input_file:
        with open(input_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    return [line.strip() for line in sys.stdin if line.strip()]


@cli.command("detect")
@click.option("--model-name", default="paraphrase-MiniLM-L6-v2", show_default=True)
@click.option("--batch-size", default=16, show_default=True, type=int)
@click.option("--threshold-strategy", type=click.Choice([s.name for s in ThresholdStrategy]), default="STAT_PERCENTILE_BLEND")
@click.option("--percentile", default=10.0, show_default=True, type=float)
@click.option("--z-k", default=1.5, show_default=True, type=float)
@click.option("--relevance-strategy", type=click.Choice([s.name for s in RelevanceStrategy]), default="COSINE_TO_CENTROID")
@click.option("--json", "as_json", is_flag=True, help="Output JSON instead of text")
@click.option("--min-sentences", default=3, show_default=True, type=int)
@click.argument("input_file", required=False, type=click.Path(exists=True, dir_okay=False, path_type=str))
def detect(
    model_name: str,
    batch_size: int,
    threshold_strategy: str,
    percentile: float,
    z_k: float,
    relevance_strategy: str,
    as_json: bool,
    min_sentences: int,
    input_file: str | None,
) -> None:
    lines = _load_lines(input_file)
    if len(lines) < min_sentences:
        raise click.UsageError(f"Need at least {min_sentences} sentences; got {len(lines)}.")

    pcfg = PipelineConfig(
        embedding=EmbeddingConfig(model_name=model_name, batch_size=batch_size),
        threshold=ThresholdConfig(strategy=ThresholdStrategy[threshold_strategy], percentile=percentile, z_k=z_k),
        relevance=RelevanceConfig(strategy=RelevanceStrategy[relevance_strategy]),
    )

    cleaned = [preprocess_text(s, pcfg.preprocess) for s in lines]
    model = get_model(pcfg.embedding)
    embs = encode_sentences(cleaned, pcfg.embedding, model=model)
    centroid = compute_central_embedding(embs)
    scores = compute_similarity_scores(embs, centroid, pcfg.relevance)

    text_length = sum(len(s.split()) for s in lines)
    thr = calculate_threshold(scores, text_length, pcfg.threshold)
    irrelevant = identify_irrelevant_sentences(lines, scores, thr)

    if as_json:
        out = {
            "scores": scores,
            "threshold": thr,
            "irrelevant": irrelevant,
        }
        click.echo(json.dumps(out, ensure_ascii=False))
    else:
        click.echo(f"threshold={thr:.4f}")
        click.echo("Irrelevant sentences:")
        for s in irrelevant:
            click.echo(s)


@cli.command("scores")
@click.option("--model-name", default="paraphrase-MiniLM-L6-v2", show_default=True)
@click.option("--batch-size", default=16, show_default=True, type=int)
@click.option("--relevance-strategy", type=click.Choice([s.name for s in RelevanceStrategy]), default="COSINE_TO_CENTROID")
@click.argument("input_file", required=False, type=click.Path(exists=True, dir_okay=False, path_type=str))
def scores_cmd(model_name: str, batch_size: int, relevance_strategy: str, input_file: str | None) -> None:
    lines = _load_lines(input_file)
    pcfg = PipelineConfig(
        embedding=EmbeddingConfig(model_name=model_name, batch_size=batch_size),
        relevance=RelevanceConfig(strategy=RelevanceStrategy[relevance_strategy]),
    )
    cleaned = [preprocess_text(s, pcfg.preprocess) for s in lines]
    model = get_model(pcfg.embedding)
    embs = encode_sentences(cleaned, pcfg.embedding, model=model)
    centroid = compute_central_embedding(embs)
    scores = compute_similarity_scores(embs, centroid, pcfg.relevance)
    for s, sc in zip(lines, scores):
        click.echo(f"{sc:.4f}\t{s}")


if __name__ == "__main__":
    cli()
