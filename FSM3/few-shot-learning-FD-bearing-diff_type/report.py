#  Copyright (c) 2016, NVIDIA Corporation
#  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of NVIDIA Corporation nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import glob, os, shutil, sys, time, string, warnings, datetime
from collections import OrderedDict
import numpy as np

#----------------------------------------------------------------------------

class Tap:
    def __init__(self, stream):
        self.stream = stream
        self.buffer = ''
        self.file = None
        pass

    def write(self, s):
        self.stream.write(s)
        self.stream.flush()
        if self.file is not None:
            self.file.write(s)
            self.file.flush()
        else:
            self.buffer = self.buffer + s

    def set_file(self, f):
        assert(self.file is None)
        self.file = f
        self.file.write(self.buffer)
        self.file.flush()
        self.buffer = ''

    def flush(self):
        self.stream.flush()
        if self.file is not None:
            self.file.flush()

    def close(self):
        self.stream.close()
        if self.file is not None:
            self.file.close()
            self.file = None

#----------------------------------------------------------------------------

#----------------------------------------------------------------------------

def create_result_subdir(result_dir, run_desc):
    ordinal = 0
    for fname in glob.glob(os.path.join(result_dir, '*')):
        try:
            fbase = os.path.basename(fname)
            ford = int(fbase[fbase.find('__')+2:])
            ordinal = max(ordinal, ford + 1)
        except ValueError:
            pass

    result_subdir = os.path.join(result_dir, '%s__%03d' % (run_desc, ordinal))
    if os.path.isdir(result_subdir):
        return create_result_subdir(result_dir, run_desc) # Retry.
    if not os.path.isdir(result_subdir):
        os.makedirs(result_subdir)
    return result_subdir

#----------------------------------------------------------------------------

def export_sources(target_dir):
    os.makedirs(target_dir)
    for ext in ('py', 'pyproj', 'sln'):
        for fn in glob.glob('*.' + ext):
            shutil.copy2(fn, target_dir)
        if os.path.isdir('src'):
            for fn in glob.glob(os.path.join('src', '*.' + ext)):
                shutil.copy2(fn, target_dir)

#----------------------------------------------------------------------------
