import json
import os.path
import re
import gzip

from tqdm import tqdm


class MeshStorage():
    """
    CLass to manage and store precomputed mesh files for a specified path
    """

    def __init__(self, path, progress):
        self.path = path
        self.progress = progress

    def get_path(self):
        return self.path

    def put_file(self, file_path, content,
                 content_type, compress,
                 cache_control=None):

        if compress is None:
            with open(file_path, 'w') as f:
                f.write(content)
            return

        # keep default as gzip
        if compress == "br":
            file_path += ".br"
        elif compress:
            file_path += '.gz'

        if content and content_type and re.search('json|te?xt', content_type) and type(content) is str:
            content = content.encode('utf-8')

        try:
            with open(file_path, 'wb') as f:
                f.write(content)
        except IOError as err:
            with open(file_path, 'wb') as f:  # retry
                f.write(content)

    def put_files(self, files, content_type=None, compress=None, compress_level=None, cache_control=None, block=True):
        """
        Put lots of files at once and get a nice progress bar. It'll also wait
        for the upload to complete.
        Required:
          files: [ (filepath, content), .... ]
        """
        desc = 'Uploading'
        for path, content in tqdm(files, disable=(not self.progress), desc=desc):
            content = gzip.compress(content, compresslevel=compress_level)
            self.put_file(path, content, content_type, compress, cache_control=cache_control)
        return self
