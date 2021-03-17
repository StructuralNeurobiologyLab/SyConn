from flask import Flask, request, abort, jsonify, send_from_directory, send_file
from flask_cors import CORS, cross_origin
from syconn import global_params
import os, json
import zipfile

def createDownloadUrl():
    # MESH_DIRECTORY = os.path.expanduser('~/mnt/wholebrain/scratch/hashirah/SyConn/syconn/meshes')
    app = Flask(__name__)
    cors = CORS(app, resources={r"/foo": {"origins": "*"}})

    app.config['CORS_HEADERS'] = 'Content-Type'
    app.config['MESH_DIRECTORY'] = '/wholebrain/u/amancu/SyConn/example_cube2/meshes'

    @app.route("/")
    def get():
        """Welcome"""
        return 'Welcome to Flask!'

    @app.route("/<path:path>/info")
    @cross_origin(origin='*',headers=['Content- Type','Authorization'])
    def get_info(path):
        """Download info."""
        try:
            print('in info')
            return send_from_directory(app.config['MESH_DIRECTORY'], filename='info', as_attachment=True)
        except FileNotFoundError:
            abort(404)

    @app.route("/<path:path>/<seg>:0")
    @cross_origin(origin='*',headers=['Content- Type','Authorization','Access-Control-Allow-Origin'])
    def get_metadata(path, seg):
        filename = f"{seg}:0"
        """Download meta data file"""
        try:
            print('in seg:0')
            return send_from_directory(app.config['MESH_DIRECTORY'], filename=seg+':0', as_attachment=True)
        except FileNotFoundError:
            print(':0 error')
            abort(404)

    @app.route("/<path:path>/<seg>")
    @cross_origin(origin='*',headers=['Content- Type','Authorization','Access-Control-Allow-Origin'])
    def get_seg(path, seg):
        # filename = f"{seg}:0"
        """Download fragments"""
        try:
            print('in seg')
            return send_from_directory(app.config['MESH_DIRECTORY'], filename=seg+':0:0-12000_0-11000_0-11000.gz', as_attachment=True)
        except FileNotFoundError:
            abort(404)

    app.run(debug=True, host='127.0.0.1', port=8001)


if __name__ == "__main__":
    createDownloadUrl()
