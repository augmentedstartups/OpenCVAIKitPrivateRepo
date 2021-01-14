#Created by Luxonis
#Modified by Augmented Startups - 18/12/2020
#Watch the tutorial Series here : http://bit.ly/OAKSeriesPlaylist
from __future__ import division
import json
import socketserver
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from pathlib import Path
from socketserver import ThreadingMixIn
from time import sleep
import depthai
import cv2
import numpy as np
from PIL import Image


import time

# Import the PCA9685 module.
import Adafruit_PCA9685


pwm = Adafruit_PCA9685.PCA9685()
posX = 300
posY = 150
speedX = 1
speedY = 2
ThresholdX = 20
ThresholdY = 10

class TCPServerRequest(socketserver.BaseRequestHandler):
    def handle(self):
        # Handle is called each time a client is connected
        # When OpenDataCam connects, do not return - instead keep the connection open and keep streaming data
        # First send HTTP header
        header = 'HTTP/1.0 200 OK\r\nServer: Mozarella/2.2\r\nAccept-Range: bytes\r\nConnection: close\r\nMax-Age: 0\r\nExpires: 0\r\nCache-Control: no-cache, private\r\nPragma: no-cache\r\nContent-Type: application/json\r\n\r\n'
        self.request.send(header.encode())
        while True:
            sleep(0.1)
            if hasattr(self.server, 'datatosend'):
                self.request.send(self.server.datatosend.encode() + "\r\n".encode())


# HTTPServer MJPEG
class VideoStreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
        self.end_headers()
        while True:
            sleep(0.1)
            if hasattr(self.server, 'frametosend'):
                image = Image.fromarray(cv2.cvtColor(self.server.frametosend, cv2.COLOR_BGR2RGB))
                stream_file = BytesIO()
                image.save(stream_file, 'JPEG')
                self.wfile.write("--jpgboundary".encode())

                self.send_header('Content-type', 'image/jpeg')
                self.send_header('Content-length', str(stream_file.getbuffer().nbytes))
                self.end_headers()
                image.save(self.wfile, 'JPEG')


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""
    pass

# Configure min and max servo pulse lengths
servo_min = 300  # Min pulse length out of 4096
servo_max = 600 # Max pulse length out of 4096

# Helper function to make setting a servo pulse width simpler.
def set_servo_pulse(channel, pulse):
    pulse_length = 1000000    # 1,000,000 us per second
    pulse_length //= 60       # 60 Hz
    print('{0}us per period'.format(pulse_length))
    pulse_length //= 4096     # 12 bits of resolution
    print('{0}us per bit'.format(pulse_length))
    pulse *= 1000
    pulse //= pulse_length
    pwm.set_pwm(channel, 0, pulse)

# Set frequency to 60hz, good for servos.
pwm.set_pwm_freq(60)

# start TCP data server
server_TCP = socketserver.TCPServer(("192.168.0.117", 8070), TCPServerRequest)
th = threading.Thread(target=server_TCP.serve_forever)
th.daemon = True
th.start()


# start MJPEG HTTP Server
server_HTTP = ThreadedHTTPServer(("192.168.0.117", 8090), VideoStreamHandler)
th2 = threading.Thread(target=server_HTTP.serve_forever)
th2.daemon = True
th2.start()

device = depthai.Device('', False)

pipeline = device.create_pipeline(config={
    "streams": ["metaout", "previewout"],
    "ai": {
        "blob_file": str(Path('./mobilenet-ssd/model.blob').resolve().absolute()),
        "blob_file_config": str(Path('./mobilenet-ssd/config.json').resolve().absolute())
    }
})

if pipeline is None:
    raise RuntimeError("Error initializing pipelne")

detections = []

while True:
    nnet_packets, data_packets = pipeline.get_available_nnet_and_data_packets()
    


    for nnet_packet in nnet_packets:
        detections = list(nnet_packet.getDetectedObjects())

    for packet in data_packets:
        if packet.stream_name == 'previewout':
            data = packet.getData()
            data0 = data[0, :, :]
            data1 = data[1, :, :]
            data2 = data[2, :, :]
            frame = cv2.merge([data0, data1, data2])

            img_h = frame.shape[0]
            img_w = frame.shape[1]

            for detection in detections:
                if detection.label ==15:# If person is detected
                    left, top = int(detection.x_min * img_w), int(detection.y_min * img_h)
                    right, bottom = int(detection.x_max * img_w), int(detection.y_max * img_h)
                    #print(detection.label)
                    cv2.putText(frame, str(detection.label), (left, top + 10), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 0))
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    
                    ##### Finding BBOX center ####
                    cx = int(left + (right-left)/2)
                    cy = int(top + (bottom-top)/2)
                    # Draw Line
                    ImageCenterX, ImageCenterY = int(img_w/2), int(img_h/2) #Center coordinates
                    line_thickness = 2
                    cv2.line(frame, (ImageCenterX,ImageCenterY ), (cx, cy), (0, 255, 0), thickness=line_thickness)
                    
                    ####Servo Motor Control ####
                   
                        ### Vertical
                    if cy!=0:
                        #print(int(ImageCenterY-cy))
                        #print(posY)
                        if cy>ImageCenterY +ThresholdY:
                            posY = int(np.clip(posY+speedY,320,580))
                        elif cy<ImageCenterY-ThresholdY:
                            posY =  int(np.clip(posY-speedY,320,580))
                        threading.Thread(target=pwm.set_pwm(0, 0, posY)).start()
                        
                        ### Horizontal
                    if cx!=0:
                        #print(int(ImageCenterX-cx))
                        #print(posX)
                        if cx>ImageCenterX +ThresholdX:
                            posX = int(np.clip(posX-speedX,150,650))
                        elif cx<ImageCenterX-ThresholdX:
                            posX =  int(np.clip(posX+speedX,150,650))
                        threading.Thread(target=pwm.set_pwm(1, 0, posX)).start()              
        
                    #threading.Thread(target=pwm.set_pwm(0, 0, servo_max)).start()


            server_TCP.datatosend = json.dumps([detection.get_dict() for detection in detections])
            server_HTTP.frametosend = frame
            cv2.imshow('previewout', frame)

    if cv2.waitKey(1) == ord('q'):
        break

del pipeline
del device