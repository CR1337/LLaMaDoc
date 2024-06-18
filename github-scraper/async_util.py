import asyncio
import aiohttp
from typing import Dict, Any, AsyncGenerator


async def async_range(
    start: int, end: int, step: int = 1
) -> AsyncGenerator[int, None]:
    for i in range(start, end, step):
        yield i
        asyncio.sleep(0)

async def get_json_request(
    session: aiohttp.ClientSession, 
    url: str, 
    headers: Dict[str, str], 
    parameters: Dict[str, str]
) -> Dict[str, Any]:
    response = await session.get(url, headers=headers, params=parameters)
    response.raise_for_status()
    return await response.json()