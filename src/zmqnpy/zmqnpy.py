import zmq
import cPickle as pickle
import numpy as np
import copy

class plug(object):
    def __init__(self, url, socket_type, bind=True, blockRead=True, hwm=20):
        self.url = url
        self.hwm = hwm
        self.bind = bind
        self.type = socket_type
        self.flags = 0
        if not blockRead:
            self.flags = zmq.NOBLOCK

    def start(self):
        # import ipdb; ipdb.set_trace()
        self.ctx = zmq.Context()
        self.s = self.ctx.socket(self.type)
        self.s.set_hwm(self.hwm)
        if self.bind:
            self.s.bind(self.url)
        else:
            self.s.connect(self.url)
        if self.type == zmq.SUB:
            self.s.setsockopt(zmq.SUBSCRIBE, '')

    def getNewest(self, data=None):
        flags = self.flags
        if self.flags != zmq.NOBLOCK:
            data = self.get()
        self.flags = zmq.NOBLOCK
        while True:
            newData = self.get()
            if newData is not None:
                data = newData
            else:
                break
        self.flags = flags
        return data

    def get(self):
        try:
            msg = self.s.recv_multipart(copy=False, flags=self.flags)
            if len(msg) == 1:
                return pickle.loads(str(msg[0]))
            elif len(msg) == 2:
                header = pickle.loads(str(msg[0]))
                data = np.frombuffer(msg[1], dtype=header[0])
                return data.reshape(header[1])
            else:
                header = pickle.loads(str(msg[1]))
                data = np.frombuffer(msg[2], dtype=header[0])
                return data.reshape(header[1])

        except zmq.Again:
            return None

    def put(self, data):
        # import ipdb; ipdb.set_trace()
        if data.size > 15000 and data.flags.c_contiguous:
            header = (data.dtype, data.shape)
            message = [pickle.dumps(header), np.getbuffer(data)]
            self.s.send_multipart(message, copy=False)
        else:
            self.s.send_pyobj(data)

    def end(self):
        self.s.close()
        self.ctx.term()



