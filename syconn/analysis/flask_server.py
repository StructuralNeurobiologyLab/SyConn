from flask import Flask, abort, make_response
from flask_cors import CORS, cross_origin
from utils import get_encoded_mesh
from syconn.analysis.backend import SyConnBackend
from knossos_utils import KnossosDataset
import json

ATTRIBUTES = ('sv', 'mi', 'sj', 'vc')

def createDownloadUrl(host, port, backend: SyConnBackend, logger, seg_path, debug):
    """
    Provide info, metadata and encoded mesh to Neuroglancer frontend
    http://<HOST>:<PORT>/<obj_type>/info => info
    http://<HOST>:<PORT>/<ssv_id>:<lod> => json metadata
    http://<HOST>:<PORT>/<ssv_id>:<lod>:<ssv_id> => encoded mesh
    """
    app = Flask(__name__)
    cors = CORS(app, resources={r"/*": {"origins": "*"}})
    app.config['CORS_HEADERS'] = 'Content-Type'
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    dataset = KnossosDataset(seg_path)

    @app.route("/<string:obj_type>/info", methods=['GET'])
    def get_info(obj_type):
        """Download info."""
        try:
            response = make_response(json.dumps({"@type": "neuroglancer_legacy_mesh"}))
            response.cache_control.max_age = 0
            response.content_type = 'application/json'
            return response
        except FileNotFoundError:
            abort(404)

    @app.route("/<string:obj_type>/<int:ssv_id>:<int:lod>", methods=['GET'])
    def get_metadata(obj_type, ssv_id, lod):
        """Download metadata"""
        try:
            fragments = []
            fragments.append('{}:{}:{}_mesh'.format(ssv_id, lod, ssv_id))
            response = make_response(json.dumps({"fragments": fragments}))
            response.cache_control.max_age = 0
            response.content_type = 'application/json'
            return response
        except FileNotFoundError:
            print('Error retrieving json metadata of ssv_id {}'.format(ssv_id))
            abort(404)

    @app.route("/<string:obj_type>/<int:ssv_id_1>:<int:lod>:<int:ssv_id_2>_mesh", methods=['GET'])
    def get_seg(obj_type, ssv_id_1, lod, ssv_id_2):
        """Download encoded mesh"""
        print('Found seg: ', ssv_id_1)
        try:
            if ssv_id_1 != ssv_id_2:
                print('over here amigo')
                abort(404)
            encoded_mesh = get_encoded_mesh(backend, ssv_id_1, obj_type)
            response = make_response(encoded_mesh)
            response.cache_control.max_age = 0
            response.content_type = 'application/octet-stream'
            response.content_encoding = 'precomputed'
            return response
        except FileNotFoundError:
            print('Error retrieving encoded mesh of ssv_id {}'.format(ssv_id_1))
            abort(404)

    app.run(debug=debug, host=host, port=port, use_reloader=False)
