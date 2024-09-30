import logging
import time

from typing import AsyncGenerator
import argparse
from wtpsplit import WtP
from exorde.item import get_item
from exorde.models import Processed, LiveConfiguration, StaticConfiguration
from exorde.process import process, TooBigError
from exorde_data import Item, Content, Url, Author, Domain, CreatedAt
from typing import AsyncGenerator
import tiktoken
from ftlangdetect import detect as lang_detect
from exorde.counter import AsyncItemCounter
from exorde.evaluate_token_count import evaluate_token_count
from typing import Callable
import datetime
import asyncio

wtp = WtP("wtp-canine-s-1l")


def split_in_sentences(string: str):
    sentences = []
    string_no_lb = string.replace("\n", " ")
    detected_language = lang_detect(string_no_lb, low_memory=False)
    try:
        try:
            sents = wtp.split(string, lang_code=detected_language["lang"])
        except:
            logging.info(
                f"WTP: could not split with lang: {detected_language}, trying with English..."
            )
            sents = wtp.split(string, lang_code="en")

        for doc in sents:
            sentences.append(doc)
    except Exception as e:
        logging.info(f"[Sentence splitter] error: {e}")
        sentences = []

    sentences = [x for x in sentences if x and len(x) > 5]
    return sentences


def aggregate_sents_into_paragraphs(
    sentences: list[str], chunk_size: int = 500
):
    paragraphs = []
    current_paragraph = []
    token_count = 0

    try:
        for sent in sentences:
            sent_ = str(sent).replace("\n", "")
            sent_tokens_count = int(evaluate_token_count(str(sent_)))
            # Check if adding the current sentence exceeds the maximum token count
            if token_count + sent_tokens_count > chunk_size:
                current_paragraph_str = " ".join(current_paragraph)
                paragraphs.append(current_paragraph_str)
                current_paragraph = []
                token_count = 0

            current_paragraph.append(sent_)
            token_count += sent_tokens_count

        # Add the last remaining paragraph
        if len(current_paragraph) > 0:
            current_paragraph_str = " ".join(current_paragraph)
            paragraphs.append(current_paragraph_str)

        logging.info(
            f"[Paragraph aggregator] Made {len(paragraphs)} paragraphs ({chunk_size} tokens long)"
        )
    except Exception as e:
        logging.info(f"[Paragraph aggregator] error: {e}")
        paragraphs = []

    paragraphs = [x for x in paragraphs if x and len(x) > 5]
    return paragraphs


def split_string_into_chunks(string: str, chunk_size: int):
    ## 1) Split main text in sentences
    sentences = split_in_sentences(string)
    ## 2) a) Recompose paragraphs from sentences
    ##    b) while keeping each paragram token count under "max_token_count"
    paragraphs = aggregate_sents_into_paragraphs(sentences, chunk_size)
    return paragraphs


def split_item(item: Item, max_token_count: int) -> list[Item]:
    if not item.content or len(str(item.content)) <= max_token_count:
        return [item]
    else:
        return [
            Item(
                content=Content(str(chunk)),
                author=item.author,
                created_at=CreatedAt(item.created_at),
                domain=Domain(item.domain),
                url=Url(item.url),
            )
            for chunk in split_string_into_chunks(
                str(item.content), max_token_count
            )
        ]


async def process_item(item, lab_configuration, max_depth_classification, websocket_send, spotting_identifier, live_configuration, batch, item_id, selected_batch_size, times, gather_time):
    diff = time.time() - gather_time
    gather_time = time.time()
    times.append(gather_time)
    item_id += 1
    try:
        start_time: float = time.perf_counter()
        splitted_mode = False

        try:
            # Process the item concurrently
            processed_item: Processed = await process(item, lab_configuration, max_depth_classification)
            batch.append((item_id, processed_item))

            # Send websocket message concurrently
            await websocket_send({
                "jobs": {
                    spotting_identifier: {
                        "items": {
                            str(item_id): {
                                "collection_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "domain": str(item.domain),
                                "url": str(item.url),
                            }
                        }
                    }
                }
            })

        except TooBigError:
            logging.info("\n_________ Paragraph maker __________________")
            splitted: list[Item] = split_item(item, live_configuration["max_token_count"])
            splitted_mode = True

            # Log all splitted items
            for i, item in enumerate(splitted):
                logging.info(f"\t\t[Paragraph] Sub-split item {i} = {item}")

            # Create async tasks for processing each chunk concurrently
            tasks = []
            for chunk in splitted:
                tasks.append(asyncio.create_task(process(chunk, lab_configuration, max_depth_classification)))

            processed_chunks = await asyncio.gather(*tasks)

            # Create async tasks for websocket sends
            tasks = []
            for processed_chunk in processed_chunks:
                batch.append((item_id, processed_chunk))
                tasks.append(asyncio.create_task(websocket_send({
                    "jobs": {
                        spotting_identifier: {
                            "items": {
                                str(item_id): {
                                    "collection_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "domain": str(item.domain),
                                    "url": str(item.url),
                                }
                            }
                        }
                    }
                })))

            await asyncio.gather(*tasks)

            # Log execution time for each chunk
            for chunk in splitted:
                item_token_count_ = evaluate_token_count(str(chunk.content))
                end_time: float = time.perf_counter()
                exec_time_s: float = end_time - start_time
                logging.info(f"[PARAGRAPH MODE] + A new sub-item has been processed {len(batch)}/{selected_batch_size} - ({exec_time_s} s) - Source = {str(item['domain'])} -  token count = {item_token_count_}")

        if not splitted_mode:
            end_time: float = time.perf_counter()
            item_token_count = evaluate_token_count(str(item.content))
            exec_time_s: float = end_time - start_time
            logging.info(f" + A new item has been processed {len(batch)}/{selected_batch_size} - ({exec_time_s} s) - Source = {str(item['domain'])} -  token count = {item_token_count}")

        if diff > 90 and len(batch) >= 5:
            logging.info("Early-Stop current batch to prevent data-aging")
            return batch

        # Evaluate the maximum allowed cumulated token count in batch
        max_batch_total_tokens_ = int(live_configuration.get("batch_size", 30000)) * int(live_configuration.get("max_token_count", 150))

        # Evaluate the cumulated number of tokens in the batch
        cumulative_token_size = sum(
            [evaluate_token_count(str(item.item.content)) for (__id__, item) in batch]
        )

        if cumulative_token_size > max_batch_total_tokens_ or len(batch) >= selected_batch_size:
            return batch

    except Exception as e:
        logging.exception("An error occurred while processing an item")
        return []


async def prepare_batch(
    static_configuration: StaticConfiguration,
    live_configuration: LiveConfiguration,
    command_line_arguments: argparse.Namespace,
    counter: AsyncItemCounter,
    websocket_send: Callable,
    spotting_identifier: str,
) -> list[tuple[int, Processed]]:
    max_depth_classification: int = live_configuration["max_depth"]
    batch: list[tuple[int, Processed]] = []  # id, item
    generator: AsyncGenerator[Item, None] = get_item(
        command_line_arguments, counter, websocket_send
    )
    lab_configuration: dict = static_configuration["lab_configuration"]
    item_id = -1
    selected_batch_size = (
        command_line_arguments.custom_batch_size
        if command_line_arguments.custom_batch_size
        else live_configuration["batch_size"]
    )

    gather_time = time.time()
    times = [time.time()]  # [prepare_batch_start_time, ... item.recolt_time]
    async for item in generator:
        tasks.append(asyncio.create_task(
            process_item(item, lab_configuration, max_depth_classification, websocket_send, spotting_identifier, live_configuration, batch, item_id, selected_batch_size, times, gather_time)
        ))

    return await asyncio.gather(*tasks)
    
