from flask import Flask, request, abort, jsonify, send_from_directory, send_file, make_response
from flask_cors import CORS, cross_origin
from syconn import global_params
import os, json
import zipfile
import numpy as np
from syconn.analysis.utils import _to_filename

def createDownloadUrl():
    # MESH_DIRECTORY = os.path.expanduser('~/mnt/wholebrain/scratch/hashirah/SyConn/syconn/meshes')
    app = Flask(__name__)
    cors = CORS(app, resources={r"/foo": {"origins": "*"}})

    app.config['CORS_HEADERS'] = 'Content-Type'
    app.config['MESH_DIRECTORY'] = os.path.expanduser('~/mnt/wholebrain/scratch/hashirah/SyConn/syconn/meshes/mi')

    @app.route("/")
    def get():
        """Welcome"""
        return 'Welcome to Flask!'

    @app.route("/<path:path>/info", methods=['GET'])
    @cross_origin(origin='*',headers=['Content-Type','Authorization', 'Access-Control-Allow-Origin'])
    def get_info(path):
        """Download info."""
        try:
            print('in info')
            response = make_response(send_from_directory(app.config['MESH_DIRECTORY'], filename='info', as_attachment=True))
            response.cache_control.max_age=0
            # response.headers['Content-Type'] = 'application/json'
            response.content_type = 'application/json'

            return response
            # return send_from_directory(app.config['MESH_DIRECTORY'], filename='info', as_attachment=True)
        except FileNotFoundError:
            abort(404)

    @app.route("/<path:path>/<seg>:0", methods=['GET'])
    @cross_origin(origin='*',headers=['Content-Type','Authorization','Access-Control-Allow-Origin'])
    def get_metadata(path, seg):
        filename = f"{seg}:0"
        """Download meta data file"""
        try:
            print('in seg:0')
            print(filename)
            response = make_response(send_from_directory(app.config['MESH_DIRECTORY'], filename=filename, as_attachment=True))
            response.cache_control.max_age=0
            response.content_type = 'application/json'

            return response
            # return send_from_directory(app.config['MESH_DIRECTORY'], filename=filename, as_attachment=True)
        except FileNotFoundError:
            print(':0 error')
            abort(404)

    @app.route("/<path:path>/<seg1>:0:<seg2>", methods=['GET'])
    @cross_origin(origin='*',headers=['Content-Type','Authorization','Access-Control-Allow-Origin'])
    def get_seg(path, seg1, seg2):
        print('Found seg: ', seg1)
        prefix = seg1
        """Download fragments"""
        try:
            print('in seg')
            print('Prefix: ', prefix)
            # bounds = _to_filename(np.array([600 * 20, 400 * 10, 400 * 10])*1e-9)
            filename = prefix + ':0:' + prefix + '_mesh'
            filename = seg1 + ':0:' + seg2
            print('Filename: ', filename)
            response = make_response(send_from_directory(app.config['MESH_DIRECTORY'], filename=filename, as_attachment=True))
            response.cache_control.max_age=0
            response.content_type = 'application/octet-stream'
            response.content_encoding = 'precomputed'

            return response
            # return send_from_directory(app.config['MESH_DIRECTORY'], filename=filename+':0:0-12000_0-4000_0-4000', as_attachment=True)
        except FileNotFoundError:
            abort(404)

    app.run(debug=True, host='127.0.0.1', port=8000)


if __name__ == "__main__":
    createDownloadUrl()
