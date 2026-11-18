"""Rename inpainted topics in merged mcap files to match the live lamb naming."""

import sys
from pathlib import Path

from mcap.reader import make_reader
from mcap.writer import Writer

RENAMES = {
    '/cameras/third_eye/color/image_raw_inpainted/compressed': '/cameras/third_eye/color/image_raw_inpainted_lamb/compressed',
    '/cameras/third_eye/color/image_raw_inpainted_mask/compressed': '/cameras/third_eye/color/image_raw_inpainted_lamb_mask/compressed',
}


def rename_topics(src: Path):
    tmp = src.with_suffix('.tmp')
    with open(src, 'rb') as f_in, open(tmp, 'wb') as f_out:
        reader = make_reader(f_in)
        writer = Writer(f_out)
        writer.start()
        schema_map = {}
        channel_map = {}
        for schema_id, schema in reader.get_summary().schemas.items():
            new_id = writer.register_schema(name=schema.name, encoding=schema.encoding, data=schema.data)
            schema_map[schema_id] = new_id
        for channel_id, channel in reader.get_summary().channels.items():
            topic = RENAMES.get(channel.topic, channel.topic)
            new_id = writer.register_channel(
                topic=topic,
                message_encoding=channel.message_encoding,
                schema_id=schema_map[channel.schema_id],
                metadata=channel.metadata,
            )
            channel_map[channel_id] = new_id
        for schema, channel, message in reader.iter_messages():
            writer.add_message(
                channel_id=channel_map[channel.id],
                log_time=message.log_time,
                data=message.data,
                publish_time=message.publish_time,
                sequence=message.sequence,
            )
        writer.finish()
    tmp.rename(src)
    print(f'  renamed {src.name}')


session = Path(sys.argv[1])
for ep in sorted(session.glob('episode_*')):
    for mcap in sorted(ep.glob('*_inpainted_*.mcap')):
        rename_topics(mcap)
print('done')
