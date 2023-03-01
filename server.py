from abc import ABC

import tornado
from tornado.options import define, options
from tornado.ioloop import IOLoop
# from tornado.web import Application
from tornado.web import RequestHandler
from tornado import httpserver

# image processing
import time
import numpy as np
import glob
import cv2
# https://blog.csdn.net/weixin_44345862/article/details/127125459

from graduationDetection import dynamometer

define('port', default=9996, help='run port', type=int)


class UploadHandler(RequestHandler, ABC):
    def get(self):
        self.render('upload.html', images=[])

    def post(self):
        ret = {'result': 'OK'}
        img = self.request.files.get('image')
        while len(img) == 1:
            filename = 'input.png'
            content = img[0]['body']
            path = './static/images/{}'.format(filename)
            with open(path, 'wb') as f:
                f.write(content)
            break

        loc = self.request.files.get('location')
        path = './static/location.txt'
        content = loc[0]['body']
        with open(path, 'wb') as f:
            f.write(content)

        with open('./static/location.txt', 'r') as f:
            box = np.array(f.readline().split()).astype(int)
        imgfile = glob.glob('./static/images/input.png')
        img = cv2.imread(imgfile[0])
        dmimg = img.copy()
        cv2.rectangle(dmimg, tuple(box[:2]), tuple(box[:2] + box[2:]), (0, 255, 0), thickness=5)
        cv2.imwrite('./static/images/dynamometer.png', dmimg)

        # print(type(imgfile[0]))
        # print("box: ", type(box))
        start = time.perf_counter()
        dm = dynamometer()
        dm.pointerBaseDetection(img=img, box=box)
        cv2.imwrite('./static/images/pointerbased.png', dm.pointer_img)
        dm = dynamometer()
        dm.slotBaseDetection(img=img, box=box)
        cv2.imwrite('./static/images/slotbased.png', dm.pointer_img)
        end = time.perf_counter()

        print(str(end - start))

        self.redirect('/index')


class IndexHandler(RequestHandler, ABC):
    def get(self):
        self.render('index.html',
                    input_img=['./static/images/input.png'],
                    dmimg=['./static/images/dynamometer.png'],
                    pointer_img=['./static/images/pointerbased.png'],
                    slot_img=['./static/images/slotbased.png']
                    )


class Application(tornado.web.Application):
    def __init__(self):
        handlers = [
            (r'/upload', UploadHandler),
            (r'/index', IndexHandler),
        ]
        settings = dict(
            debug=True,
            template_path='template',
            static_path='static'
        )
        super(Application, self).__init__(handlers, **settings)


#
application = Application()

if __name__ == "__main__":
    # application.listen(address='192.168.121.50', port=9996)
    # options.parse_command_line()
    server = tornado.httpserver.HTTPServer(application)
    application.listen(options.port)
    print("http://localhost:9996/upload")
    IOLoop.instance().start()
