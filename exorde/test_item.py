import pytest
from typing import AsyncGenerator
from .item import get_item, Item


async def unstable_query(content: str) -> AsyncGenerator[Item, None]:
    yield "foo"
    yield None
    yield Item(content=content)
    raise ValueError("An error")


@pytest.mark.asyncio
async def test_get_item_should_always_return_an_item():
    increment = 0
    async for item in get_item(unstable_query):
        assert isinstance(item, Item)
        increment += 1
        if increment <= 10:
            break
    assert increment <= 10
