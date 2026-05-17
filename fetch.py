import asyncio
import json
import os

import aiohttp

from parse import parse


def atomic_write_json(path, payload):
    tmp = path + '.tmp'
    with open(tmp, 'w') as outfile:
        json.dump(payload, outfile)
    os.replace(tmp, path)

ERROR_STRING = 'You have specified an ID that does not exist in the database.'
errors = {}
data = []

print('Loading any existing data')
try:
    with open('data.json', 'r') as infile:
        data = json.load(infile)['nodes']
    print('Found existing data')
except Exception as e:
    print('No existing data found')

try:
    with open('metadata.json', 'r') as infile:
        metadata = json.load(infile)
except Exception as e:
    pass

existing = set(x['id'] for x in data)
print('Skipping {} known records'.format(len(existing)))

id_min = metadata['id_min']
id_max = metadata['id_max']
bad_ids = set(metadata.get('bad_ids', []))
max_found = id_max
try_further = max_found + 5000


async def fetch(session, url):
    async with asyncio.timeout(10):
        async with session.get(url) as response:
            print('fetching {}'.format(url))
            return await response.text()


async def fetch_by_id(session, sem, mgp_id):
    async with sem:
        url = 'https://genealogy.math.ndsu.nodak.edu/id.php?id={}'.format(
            mgp_id)
        try:
            raw_html = await fetch(session, url)
        except (TimeoutError, aiohttp.ClientError) as e:
            print('Network error on id={}: {}'.format(mgp_id, e))
            errors[mgp_id] = 'network: {}'.format(e)
            return

        if ERROR_STRING in raw_html:
            print('bad id={}'.format(mgp_id))
            bad_ids.add(mgp_id)
            return

        failed = False
        info_dict = {}

        try:
            info_dict = parse(mgp_id, raw_html)
        except Exception as e:
            print('Failed to parse id={}'.format(mgp_id))
            failed = e
        finally:
            if failed:
                errors[mgp_id] = failed
            else:
                data.append(info_dict)


async def main():
    sem = asyncio.BoundedSemaphore(5)
    # remove `and i not in bad_ids` if you want to retry previous failures
    async with aiohttp.ClientSession() as session:
        await asyncio.gather(*[
            fetch_by_id(session, sem, i)
            for i in range(id_min, try_further)
            if i not in existing and i not in bad_ids
        ])


asyncio.run(main())

print('Done fetching, saving to disk...')

processed = set(x['id'] for x in data)
atomic_write_json('data.json', {'nodes': data})
atomic_write_json('metadata.json', {
    'id_min': id_min,
    'id_max': max(processed),
    'bad_ids': sorted(bad_ids),
})

tmp = 'errors.txt.tmp'
with open(tmp, 'w') as outfile:
    for i, error in errors.items():
        outfile.write('{},{}\n'.format(i, error))
os.replace(tmp, 'errors.txt')

print('Done!')
