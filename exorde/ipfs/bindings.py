from aiohttp import ClientSession
from aiosow.bindings import setup, alias, wrap, accumulator, wire

from exorde.ipfs import load_json_schema, validate_batch_schema, upload_to_ipfs

# setup an aiohttp session for ipfs upload
setup(wrap(lambda session: {"session": session})(lambda: ClientSession()))
setup(wrap(lambda schema: {"ipfs_schema": schema})(load_json_schema))
alias("ipfs_path")(lambda: "http://ipfs-api.exorde.network/add")
spot_block = lambda entities: {"Content": entities}

# batching
broadcast_batch_ready, on_batch_ready_do = wire()
push_to_ipfs = broadcast_batch_ready(accumulator(10)(spot_block))

# when a batch is ready, upload it to ipfs
broadcast_new_valid_batch, on_new_valid_batch_do = wire()
on_batch_ready_do(broadcast_new_valid_batch(validate_batch_schema))

broadcast_new_cid, on_new_cid_do = wire()
on_new_valid_batch_do(broadcast_new_cid(upload_to_ipfs))
