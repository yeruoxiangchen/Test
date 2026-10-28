"""Isolated browser-test server. Never imported by the production application."""
import json
import uuid

from PIL import Image, ImageDraw

from backend.app import create_app
from backend.domain import ROOT


def factory():
    root = ROOT / 'checks' / ('browser-' + uuid.uuid4().hex)
    root.mkdir(parents=True)
    (root / 'settings.json').write_text(json.dumps({'workers': []}))
    app = create_app(root, start_workers=False)

    @app.post('/api/test-fixture/{ident}')
    def finish_fixture(ident: str):
        store = app.state.store
        path = root / 'assets' / (uuid.uuid4().hex + '.png')
        image = Image.new('RGB', (512, 512), '#d7ed93')
        ImageDraw.Draw(image).text((100, 240), 'BROWSER TEST FIXTURE', fill='#25342a')
        image.save(path)
        asset = store.add_asset(path, 'image/png', 'image', ident)
        store.update(ident, status='succeeded', outputs=[asset['id']], message='Browser test fixture')
        return store.get(ident)

    return app
