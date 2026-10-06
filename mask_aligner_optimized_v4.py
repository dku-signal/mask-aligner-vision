# Standalone Mask Aligner: Arduino Mega, MG996R continuous servo signal D30
# pip install opencv-python numpy pyserial
import argparse
import json
import math
import threading
import time
import webbrowser
from collections import OrderedDict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse
import cv2
"""Four-square mask geometry and occlusion-masked wafer contrast matching."""
from pathlib import Path
from itertools import combinations
import math
import cv2
import numpy as np

class CornerKeys:
    def __init__(self, directory=None, center_tolerance=8.0, cache_mb=64):
        self.center_tolerance=float(center_tolerance)
        self.arm = 1/3
        path=Path(directory or Path(__file__).resolve().parent/'templates')/'wafer_key.png'
        if path.is_file():
            raw=cv2.imdecode(np.fromfile(str(path),np.uint8),cv2.IMREAD_GRAYSCALE)
            if raw is not None:
                _,b=cv2.threshold(raw,0,1,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                h,w=b.shape
                # Use geometry only; brightness of a photographed background is irrelevant.
                widths=[b[int(h*.15),:].mean(),b[int(h*.85),:].mean(),
                        b[:,int(w*.15)].mean(),b[:,int(w*.85)].mean()]
                value=float(np.median(widths))
                if .20<=value<=.55:self.arm=value
        self.bank=OrderedDict();self.previous_mask=None;self.previous_wafer=None
        self.bank_limit=max(0,int(cache_mb*1024*1024))
        self.bank_bytes=0
        self.token=None;self.shape=None

    @staticmethod
    def angle_delta(a,b):return (a-b+45)%90-45

    def squares(self,gray):
        result=[]
        # Several thresholds separate printed black from wafer ink and changing substrate.
        levels=np.unique(np.clip(np.r_[np.percentile(gray,[.5,1,2,4,6,9,13,18,25]),
                                      np.arange(30,151,15)],8,190).astype(int))
        max_side=min(gray.shape)*.24
        for level in levels:
            binary=(gray<level).astype(np.uint8)*255
            contours,_=cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area=cv2.contourArea(c)
                if area<64 or area>max_side**2:continue
                peri=cv2.arcLength(c,True)
                p=cv2.approxPolyDP(c,.035*peri,True)
                if len(p)!=4 or not cv2.isContourConvex(p):continue
                pts=p[:,0,:].astype(np.float32)
                edges=np.roll(pts,-1,axis=0)-pts
                lengths=np.linalg.norm(edges,axis=1)
                side=float(np.mean(lengths))
                if lengths.min()<7 or lengths.max()/lengths.min()>1.45:continue
                if np.max(abs(np.sum(edges*np.roll(edges,-1,axis=0),axis=1)/
                              (lengths*np.roll(lengths,-1))))>.40:continue
                if area/side**2<.73:continue
                x,y,w,h=cv2.boundingRect(p)
                if x<2 or y<2 or x+w>=gray.shape[1]-2 or y+h>=gray.shape[0]-2:continue
                # Compare each square with its immediate surround, not a fixed black value.
                pad=max(3,int(side*.13));x0=max(0,x-pad);y0=max(0,y-pad)
                roi=gray[y0:min(gray.shape[0],y+h+pad),x0:min(gray.shape[1],x+w+pad)]
                mask=np.zeros(roi.shape,np.uint8)
                cv2.fillConvexPoly(mask,(pts-[x0,y0]).astype(np.int32),1)
                inner=cv2.erode(mask,np.ones((3,3),np.uint8))>0
                ring=(cv2.dilate(mask,np.ones((2*pad+1,2*pad+1),np.uint8))>0)&(mask==0)
                if not inner.any() or not ring.any():continue
                ink=float(np.median(roi[inner]));outside=float(np.percentile(roi[ring],65))
                contrast=outside-ink
                if contrast<12:continue
                center=pts.mean(axis=0)
                item=dict(center=center,side=side,quad=pts,ink=ink,contrast=contrast)
                near=next((i for i,t in enumerate(result) if
                           np.linalg.norm(t['center']-center)<.3*side),None)
                if near is None:result.append(item)
                elif side>result[near]['side'] and side<1.3*result[near]['side']:result[near]=item
        return sorted(result,key=lambda t:t['contrast'],reverse=True)[:48]

    def mask_geometry(self,gray):
        squares=self.squares(gray)
        groups={}
        for i,a in enumerate(squares):
            neighbors=[]
            for j,b in enumerate(squares):
                if i==j or not .72<a['side']/b['side']<1.39:continue
                v=b['center']-a['center'];d=float(np.linalg.norm(v))
                if 1.45<d/a['side']<3.2:neighbors.append((j,v,d))
            for (j,u,du),(k,v,dv) in combinations(neighbors,2):
                if not .76<du/dv<1.32 or abs(float(u@v)/(du*dv))>.20:continue
                target=squares[j]['center']+squares[k]['center']-a['center']
                choices=[(np.linalg.norm(t['center']-target),l) for l,t in enumerate(squares)
                         if l not in (i,j,k) and .72<t['side']/a['side']<1.39]
                error,l=min(choices,default=(1e9,-1))
                observed=4 if error<=.32*a['side'] else 3
                ids=tuple(sorted((i,j,k,l) if observed==4 else (i,j,k)))
                q=[squares[z] for z in ids]
                # A fourfold angle average treats horizontal and vertical edges identically.
                edges=np.concatenate([np.roll(t['quad'],-1,axis=0)-t['quad'] for t in q])
                z=np.exp(4j*np.arctan2(edges[:,1],edges[:,0])).mean()
                edge_angle=float(np.angle(z)/4*180/np.pi)%90
                z=np.exp(4j*np.array([math.atan2(u[1],u[0]),math.atan2(v[1],v[0])])).mean()
                angle=float(np.angle(z)/4*180/np.pi)%90
                # Square edges must agree with the grid of their centers.
                if abs(self.angle_delta(edge_angle,angle))>9:continue
                center=np.mean([t['center'] for t in q],axis=0) if observed==4 else a['center']+(u+v)/2
                score=1-(error/a['side'] if observed==4 else .3)-abs(du-dv)/max(du,dv)
                score+=min(t['contrast'] for t in q)/255
                if self.previous_mask is not None:
                    score-=min(.3,float(np.linalg.norm(center-self.previous_mask['center']))/1000)
                quads=[t['quad'] for t in q]
                if observed==3:quads.append(a['quad']+u+v)
                groups[ids]=dict(center=center,angle=angle,size=(du+dv)/2+np.mean([t['side'] for t in q]),
                                 quads=quads,observed=observed,score=score,ink=np.median([t['ink'] for t in q]))
        if not groups:return None
        return max(groups.values(),key=lambda g:(g['observed'],g['score']))

    def pattern(self,size,angle):
        size=int(round(size));angle=round(angle%90,1);key=(size,angle)
        if key not in self.bank:
            yy,xx=np.mgrid[:size,:size].astype(np.float32)
            c=(size-1)/2
            p=((abs(xx-c)<size*self.arm/2)|(abs(yy-c)<size*self.arm/2)).astype(np.float32)
            # Observe the arm endpoints against a narrow exterior collar as well.
            # They still constrain the center when all four white corners are covered.
            pad=max(2,int(round(size*.04)))
            p=np.pad(p,pad)
            extent=size+2*pad;c=(extent-1)/2
            rad=math.radians(angle);n=int(math.ceil(extent*(abs(math.cos(rad))+abs(math.sin(rad)))))
            mat=cv2.getRotationMatrix2D((c,c),-angle,1);mat[:,2]+=(n-extent)/2
            p=cv2.warpAffine(p,mat,(n,n));support=cv2.warpAffine(np.ones((extent,extent),np.float32),mat,(n,n))
            required=p.nbytes+support.nbytes
            # Eviction changes storage only; regenerated float32 arrays are identical.
            while self.bank and self.bank_bytes+required>self.bank_limit:
                _,old=self.bank.popitem(last=False)
                self.bank_bytes-=old[0].nbytes+old[1].nbytes
            if required>self.bank_limit:return p,support
            self.bank[key]=(p,support)
            self.bank_bytes+=required
        self.bank.move_to_end(key)
        return self.bank[key]

    def masked_scores(self,image,valid,p,support):
        # Weighted Pearson correlation. Mask squares never contribute pixels to this fit.
        cc=lambda a,b:cv2.matchTemplate(a,b,cv2.TM_CCORR)
        n=cc(valid,support);n=np.maximum(n,1)
        st=cc(valid,p);st2=cc(valid,p*p)
        si=cc(image*valid,support);si2=cc(image*image*valid,support)
        sit=cc(image*valid,p)
        vi=np.maximum(si2-si*si/n,0);vt=np.maximum(st2-st*st/n,0)
        score=(sit-si*st/n)/np.sqrt(np.maximum(vi*vt,1e-6))
        visible=(n>=.43*support.sum()) & (st>=.28*p.sum()) & ((n-st)>=.08*(support.sum()-p.sum()))
        visible &= (vi/n>=.0016)&(vt/n>.055)
        score[~visible]=-1
        return score

    def wafer(self,gray,mask):
        valid=np.ones(gray.shape,np.float32)
        for q in mask['quads']:cv2.fillConvexPoly(valid,q.astype(np.int32),0)
        valid=1-cv2.dilate(1-valid,np.ones((5,5),np.uint8))
        # Contrast, not absolute brightness, defines the wafer's dark cross/light corners.
        image=1-gray.astype(np.float32)/255
        cx,cy=mask['center'];r=int(mask['size']*2.5)
        x0,y0=max(0,int(cx)-r),max(0,int(cy)-r)
        x1,y1=min(gray.shape[1],int(cx)+r),min(gray.shape[0],int(cy)+r)
        prior=self.previous_wafer
        if prior is not None:
            px,py=prior['center'];pr=int(prior['size']*1.6)
            x0,y0=max(x0,int(px)-pr),max(y0,int(py)-pr)
            x1,y1=min(x1,int(px)+pr),min(y1,int(py)+pr)
        if x1-x0<24 or y1-y0<24:return None
        scene=image[y0:y1,x0:x1];weights=valid[y0:y1,x0:x1]
        scale=.35
        small=cv2.resize(scene,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
        sv=cv2.resize(weights,small.shape[::-1],interpolation=cv2.INTER_AREA)
        minimum=max(30,int(mask['size']*.46));maximum=min(220,int(mask['size']*1.2))
        sizes=range(minimum,maximum+1,10)
        angles=range(0,90,10)
        if prior is not None:
            sizes=range(max(minimum,int(prior['size'])-8),min(maximum,int(prior['size'])+8)+1,4)
            angles=[prior['angle']+a for a in range(-15,16,5)]
        best=None
        for size in sizes:
            for a in angles:
                p,s=self.pattern(size*scale,a)
                if p.shape[0]>=min(small.shape):continue
                score=self.masked_scores(small,sv,p,s)
                _,value,_,loc=cv2.minMaxLoc(score)
                if best is None or value>best['score']:
                    best=dict(center=(x0+(loc[0]+(p.shape[0]-1)/2)/scale,y0+(loc[1]+(p.shape[0]-1)/2)/scale),size=size,angle=a,score=value)
        if best is None or best['score']<.55:return None
        bx,by=best['center'];r=int(best['size']*.71+12)
        x0,y0=max(0,int(bx)-r),max(0,int(by)-r)
        x1,y1=min(gray.shape[1],int(bx)+r),min(gray.shape[0],int(by)+r)
        scene=image[y0:y1,x0:x1];weights=valid[y0:y1,x0:x1]
        candidates=[]
        for size in range(max(minimum,best['size']-4),min(maximum,best['size']+4)+1,4):
            for a in np.arange(best['angle']-6,best['angle']+6.1,1):
                p,s=self.pattern(size,a)
                if p.shape[0]>=min(scene.shape):continue
                score=self.masked_scores(scene,weights,p,s)
                _,value,_,loc=cv2.minMaxLoc(score)
                candidates.append(dict(center=(x0+loc[0]+(p.shape[0]-1)/2,y0+loc[1]+(p.shape[0]-1)/2),
                                       size=size,angle=float(a)%90,score=value))
        if not candidates:return None
        best=max(candidates,key=lambda t:t['score'])
        if best['score']<.70:return None
        other=[t['score'] for t in candidates if abs(self.angle_delta(t['angle'],best['angle']))>=4]
        best['angle_valid']=best['score']>.76 and (not other or best['score']-max(other)>.010)
        return best

    @staticmethod
    def detection(t):
        if t is None:return None
        cx,cy=t['center'];size=t['size'];r=math.radians(t['angle'])
        n=int(math.ceil(size*(abs(math.sin(r))+abs(math.cos(r)))))
        return dict(center=(float(cx),float(cy)),rect=(int(cx-n/2),int(cy-n/2),n,n),
                    area=float(size*size),score=float(t['score']),quads=t.get('quads',[]),observed=t.get('observed',0))

    def process(self,frame,phase,token=None,stationary=True):
        if token!=self.token or frame.shape!=self.shape:
            self.previous_mask=self.previous_wafer=None
            self.token=token;self.shape=frame.shape
        gray=cv2.GaussianBlur(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),(3,3),0)
        mask=self.mask_geometry(gray)
        if mask is None:return None,None,None,None,False
        wafer=self.wafer(gray,mask)
        if wafer is None and self.previous_wafer is not None:
            # Reacquire after an unusually large motor step without using stale coordinates.
            self.previous_wafer=None
            wafer=self.wafer(gray,mask)
        ma=(mask['angle'],mask['score'])
        wa=(wafer['angle'],wafer['score']) if wafer and wafer['angle_valid'] else None
        if stationary:
            self.previous_mask=mask
            if wafer:self.previous_wafer=wafer
        # Fresh, independently measured centers replace a background-dependent solid fill.
        finish=bool(phase in ('XY','DONE') and wafer is not None and
                    max(abs(np.asarray(mask['center'])-wafer['center']))<self.center_tolerance and
                    wafer['score']>=.78)
        return self.detection(mask),self.detection(wafer),ma,wa,finish

import numpy as np
try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None
    list_ports = None
APP_VERSION = '2026.09.16-v4-faster-settle'
HOST = '127.0.0.1'
HTTP_PORT = 8765
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
JPEG_QUALITY = 72
PREVIEW_FPS = 20
PATTERN_CACHE_MB = 64
OVERLAY_MAX_AGE_SEC = 0.5
OVERLAY_SMOOTHING_MS = 120
ARDUINO_PORT = 'AUTO'
ARDUINO_BAUD = 9600
# X/Y below mean camera-image axes: X = horizontal, Y = vertical.
# True assumes Arduino X moves image Y and Arduino Y moves image X.
# Set False if Arduino X already moves image X.
SWAP_XY_COMMANDS = True
# Calibration is per IMAGE axis after command mapping, not Arduino label.
# Existing values retained as initial estimates; verify signs on the machine.
# If correcting X increases abs(dx), reverse MOTOR_SIGN_X only (likewise Y).
MOTOR_SIGN_X = 1
MOTOR_SIGN_Y = -1
STEPS_PER_PIXEL_X = 1.65
STEPS_PER_PIXEL_Y = 1.95
MOVE_GAIN_X = 0.85
MOVE_GAIN_Y = 0.85
MAX_STEPS_PER_AXIS = 650
# Manual button step count; 0 keeps the page's own value. calibration.json sets it.
MANUAL_STEPS = 0
DEAD_BAND_PX = 6.0
AXIS_SETTLE_SEC = 0.35
STABLE_HISTORY = 4
STABLE_MAX_JITTER_PX = 8.0
CENTERED_DONE_FRAMES = 12
WAFER_LOST_DONE_FRAMES = 8
THETA_TOL_DEG = 4.0
THETA_CONFIRM_FRAMES = 2
THETA_JITTER_DEG = 1.5
THETA_SETTLE_SEC = 0.35
POST_MOVE_GUARD_SEC = 0.10
THETA_MIN_MS = 20
THETA_MAX_MS = 250  # must not exceed firmware THETA_MAX_MS (ERR T range)
THETA_MOVE_GAIN = 0.60
THETA_FINE_GAIN = 0.40
THETA_FINE_RANGE_DEG = 8.0
THETA_PROBE_MS = (100, 150, 200, 250)
THETA_MAX_PULSES = 100
THETA_TIMEOUT_SEC = 180.0
VISION_TIMEOUT_SEC = 5.0

def wrap90(a):
    return (a + 45.0) % 90.0 - 45.0

def clamp_steps(value):
    value = int(round(value))
    return max(-MAX_STEPS_PER_AXIS, min(MAX_STEPS_PER_AXIS, value))

def put_text(img, text, pos, color=(0, 255, 0), scale=0.85, thickness=2):
    x, y = pos
    cv2.putText(img, text, (x + 2, y + 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def draw_detection(frame, det, label, color):
    if det is None:
        return
    cx, cy = det['center']
    x, y, w, h = det['rect']
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.circle(frame, (int(cx), int(cy)), 3, color, -1)
    for i,quad in enumerate(det.get('quads', [])):
        measured = i < det.get('observed', 4)
        cv2.polylines(frame, [quad.astype(np.int32)], True, color if measured else (0,165,255), 2 if measured else 1)
    detail = f"{det['observed']}/4" if label == 'MASK' else f"score={det['score']:.2f}"
    put_text(frame, f"{label} {detail}", (int(cx) + 12, int(cy) - 12), color, 0.65, 2)

def is_stable(history):
    if len(history) < STABLE_HISTORY:
        return False
    mask_points = np.array([h[0] for h in history], dtype=np.float32)
    wafer_points = np.array([h[1] for h in history], dtype=np.float32)
    mask_avg = np.mean(mask_points, axis=0)
    wafer_avg = np.mean(wafer_points, axis=0)
    mask_jitter = np.max(np.linalg.norm(mask_points - mask_avg, axis=1))
    wafer_jitter = np.max(np.linalg.norm(wafer_points - wafer_avg, axis=1))
    return mask_jitter <= STABLE_MAX_JITTER_PX and wafer_jitter <= STABLE_MAX_JITTER_PX

def get_stable_centers(history):
    mask_points = np.array([h[0] for h in history], dtype=np.float32)
    wafer_points = np.array([h[1] for h in history], dtype=np.float32)
    mask_center = np.median(mask_points, axis=0)
    wafer_center = np.median(wafer_points, axis=0)
    return ((float(mask_center[0]), float(mask_center[1])), (float(wafer_center[0]), float(wafer_center[1])))

def get_quadrant(dx, dy):
    if dx >= 0 and dy >= 0:
        return 1
    if dx < 0 and dy >= 0:
        return 2
    if dx < 0 and dy < 0:
        return 3
    return 4

def compute_steps(mask_center, wafer_center):
    mx, my = mask_center
    wx, wy = wafer_center
    dx = wx - mx
    dy = wy - my
    quadrant = get_quadrant(dx, dy)
    if abs(dx) < DEAD_BAND_PX:
        steps_x = 0
    else:
        steps_x = clamp_steps(dx * STEPS_PER_PIXEL_X * MOTOR_SIGN_X * MOVE_GAIN_X)
    if abs(dy) < DEAD_BAND_PX:
        steps_y = 0
    else:
        steps_y = clamp_steps(dy * STEPS_PER_PIXEL_Y * MOTOR_SIGN_Y * MOVE_GAIN_Y)
    # Each image axis is independent. The other error must never flip its sign.
    return (dx, dy, steps_x, steps_y, quadrant, False)

def arduino_axis(image_axis):
    if image_axis not in ('X', 'Y'):
        raise ValueError('image_axis must be X or Y')
    return {'X': 'Y', 'Y': 'X'}[image_axis] if SWAP_XY_COMMANDS else image_axis

class ArduinoLink:

    def __init__(self, port='AUTO', baud=9600):
        self.ser = None
        self.port = None
        self.lock = threading.Lock()
        self.rx_buffer = bytearray()
        self.last_rx = ''
        self.last_error = '' 
        if serial is None:
            print('pyserial is not installed. Run: python -m pip install pyserial')
            return
        self.port = self.find_port() if port == 'AUTO' else port
        if self.port is None:
            print('Arduino port not found. Motor commands disabled.')
            return
        try:
            self.ser = serial.Serial(self.port, baud, timeout=0.05, write_timeout=2.0)
            time.sleep(2.0)
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            print(f'Arduino connected: {self.port}')
        except Exception as e:
            print(f'Arduino connection failed: {e}')
            self.ser = None

    def find_port(self):
        ports = list(list_ports.comports())
        for p in ports:
            text = f'{p.device} {p.description}'.lower()
            if ('arduino' in text or 'ch340' in text or 'usb serial' in text or 'usb-serial' in text or
                    'usbmodem' in text or 'usbserial' in text):
                return p.device
        # macOS always lists Bluetooth/debug ports; never fall back to those.
        ports = [p for p in ports if 'bluetooth' not in p.device.lower() and 'debug' not in p.device.lower()]
        if ports:
            return ports[0].device
        return None

    def connected(self):
        return self.ser is not None

    def send(self, cmd):
        print(f'-> {cmd}')
        if not self.ser:
            print('Arduino not connected.')
            return False
        try:
            with self.lock:
                self.ser.reset_input_buffer()
                self.rx_buffer.clear()
                if cmd.strip().lower() not in ('s', 'stop'):
                    self.last_error = ''
                self.ser.write((cmd.strip() + '\n').encode('utf-8'))
            return True
        except Exception as e:
            self.last_error = f'SERIAL WRITE FAILED: {e}'
            print(self.last_error, flush=True)
            return False

    def read_line(self):
        if not self.ser:
            return ''
        try:
            with self.lock:
                self.rx_buffer.extend(self.ser.readline())
                if b'\n' not in self.rx_buffer:
                    if len(self.rx_buffer) > 1024:
                        self.rx_buffer.clear()
                        self.last_error = 'SERIAL RX line too long'
                    return ''
                raw, _, tail = self.rx_buffer.partition(b'\n')
                self.rx_buffer = bytearray(tail)
                line = raw.decode(errors='replace').strip()
                if line:
                    self.last_rx = line
                return line
        except Exception as e:
            self.last_error = f'SERIAL READ FAILED: {e}'
            return ''

    def wait_done(self, stop_event, timeout=25.0, expected='DONE'):
        deadline = time.monotonic() + timeout
        self.last_error = ''
        while self.connected() and time.monotonic() < deadline:
            if stop_event.is_set():
                self.stop()
                self.last_error = 'Cancelled by STOP'
                return False
            line = self.read_line()
            if self.last_error:
                error = self.last_error
                self.stop()
                self.last_error = error
                print(error, flush=True)
                return False
            if line:
                print(f'<- {line}', flush=True)
            if line == expected:
                return True
            if line == 'STOP' or line.startswith('ERR'):
                self.last_error = f'Arduino replied: {line}'
                return False
        error = f'ACK TIMEOUT: expected {expected}; last RX={self.last_rx!r}'
        self.stop()
        self.last_error = error
        print(error, flush=True)
        return False

    def stop(self):
        self.send('s')

    def close(self):
        if self.ser:
            try:
                self.stop()
                self.ser.close()
            except Exception:
                pass

class PreviewSmoother:
    """Time-based display interpolation only. Motor measurements stay untouched."""
    def __init__(self, smoothing_ms=OVERLAY_SMOOTHING_MS):
        self.tau=max(0.0,smoothing_ms/1000.0)
        self.reset()

    def reset(self):
        self.context=self.source_time=self.render_time=None
        self.targets=[None,None]
        self.states=[None,None]

    @staticmethod
    def geometry(det,angle):
        if det is None:return None
        return dict(center=np.array(det['center'],dtype=float),
                    extent=np.array(det['rect'][2:4],dtype=float),
                    quads=[np.array(q,dtype=float,copy=True) for q in det.get('quads',[])],
                    angle=None if angle is None else float(angle[0]),
                    observed=det.get('observed',0))

    @staticmethod
    def match_quads(previous,current):
        # Discovery order, vertex start and winding can all change between frames.
        remaining=list(previous);aligned=[]
        for target in current:
            index=min(range(len(remaining)),key=lambda i:float(np.sum(
                (remaining[i].mean(axis=0)-target.mean(axis=0))**2)))
            prior=remaining.pop(index)
            choices=[np.roll(q,shift,axis=0) for q in (prior,prior[::-1]) for shift in range(4)]
            aligned.append(min(choices,key=lambda q:float(np.sum((q-target)**2))).copy())
        return aligned

    def render(self,md,wd,ma,wa,source_time,context,now,completed_at=None,max_age=None):
        # Source age and time since completion serve different purposes.
        completed_at=source_time if completed_at is None else completed_at
        max_age=OVERLAY_MAX_AGE_SEC if max_age is None else max_age
        if (source_time is None or now-source_time>VISION_TIMEOUT_SEC or
                now-completed_at>max_age):
            self.reset()
            return None,None,None,None
        if self.tau==0:return md,wd,ma,wa
        if context!=self.context or (self.render_time is not None and
                                    now-self.render_time>OVERLAY_MAX_AGE_SEC):
            self.reset();self.context=context
        if source_time!=self.source_time:
            for i,(det,angle) in enumerate(((md,ma),(wd,wa))):
                target=self.geometry(det,angle);state=self.states[i]
                if target is None:self.states[i]=None
                else:
                    snap=(state is None or np.linalg.norm(target['center']-state['center'])>8.0
                          or target['observed']!=state['observed']
                          or len(target['quads'])!=len(state['quads']))
                    if not snap and target['angle'] is not None and state['angle'] is not None:
                        snap=abs(wrap90(target['angle']-state['angle']))>8.0
                    if snap:self.states[i]=self.geometry(det,angle)
                    elif target['quads']:
                        state['quads']=self.match_quads(state['quads'],target['quads'])
                self.targets[i]=target
            self.source_time=source_time
        elapsed=0.0 if self.render_time is None else max(0.0,now-self.render_time)
        alpha=1.0-math.exp(-elapsed/self.tau)
        self.render_time=now
        detections=[];angles=[]
        for i,(det,angle) in enumerate(((md,ma),(wd,wa))):
            state,target=self.states[i],self.targets[i]
            if det is None or state is None or target is None:
                detections.append(None);angles.append(None);continue
            state['center']+=alpha*(target['center']-state['center'])
            state['extent']+=alpha*(target['extent']-state['extent'])
            for q,destination in zip(state['quads'],target['quads']):q+=alpha*(destination-q)
            if target['angle'] is None or state['angle'] is None:state['angle']=target['angle']
            else:state['angle']=(state['angle']+alpha*wrap90(target['angle']-state['angle']))%90
            shown=dict(det)
            shown['center']=tuple(float(x) for x in state['center'])
            width,height=state['extent'];cx,cy=state['center']
            shown['rect']=(int(round(cx-width/2)),int(round(cy-height/2)),
                           int(round(width)),int(round(height)))
            shown['quads']=[q.copy() for q in state['quads']]
            detections.append(shown)
            angles.append(None if angle is None else (state['angle'],angle[1]))
        return detections[0],detections[1],angles[0],angles[1]


class MaskAlignerApp:

    def __init__(self, camera_index=CAMERA_INDEX, cache_mb=PATTERN_CACHE_MB, preview_fps=PREVIEW_FPS, overlay_smoothing_ms=OVERLAY_SMOOTHING_MS, detector='opencv', skip_theta=False):
        self.skip_theta = skip_theta
        self.lock = threading.RLock()
        self.frame_ready = threading.Condition(self.lock)
        self.jpeg_ready = threading.Condition(self.lock)
        self.latest_frame = None
        self.capture_seq = 0
        self.capture_time = 0.0
        self.camera_epoch = 0
        self.motion_epoch = 0
        self.overlay = None
        self.jpeg_seq = 0
        self.preview_fps_limit = max(1, min(30, preview_fps))
        self.overlay_smoothing_ms = overlay_smoothing_ms
        self.capture_fps = self.preview_fps = self.vision_fps = 0.0
        self.vision_ms = 0.0
        self.cache_used_mb = 0.0
        self.cache_limit_mb = cache_mb
        self.camera_shape = None
        self.workers = []
        self.exit_event = threading.Event()
        self.stop_event = threading.Event()
        self.arduino = ArduinoLink(ARDUINO_PORT, ARDUINO_BAUD)
        self.cap = None
        self.camera_index = camera_index
        self.requested_camera_index = camera_index
        self.last_jpeg = None
        self.history = deque(maxlen=STABLE_HISTORY)
        self.auto_running = False
        self.align_finished = False
        self.moving = False
        self.stable = False
        self.centered_count = 0
        self.wafer_lost_count = 0
        self.mask_detected = False
        self.wafer_detected = False
        self.dx = None
        self.dy = None
        self.steps_x = 0
        self.steps_y = 0
        self.quadrant = None
        self.quadrant_fix = False
        self.next_info = None
        self.status = 'STARTING'
        if detector == 'opencv-v4':
            self.key_detector = CornerKeys(center_tolerance=DEAD_BAND_PX, cache_mb=cache_mb)
        elif detector == 'opencv':
            # Default path needs only opencv/numpy; sam2_keys pulls in torch.
            from subpixel import PreciseKeys
            self.key_detector = PreciseKeys(center_tolerance=DEAD_BAND_PX, cache_mb=cache_mb)
        else:
            from sam2_keys import make_detector
            self.key_detector = make_detector(detector, center_tolerance=DEAD_BAND_PX, cache_mb=cache_mb)
        self.samples = deque(maxlen=STABLE_HISTORY)
        self.frame_id = 0
        self.frame_time = 0.0  # Most recent accepted analysis completion (original semantics).
        self.measurement_captured_at = 0.0
        self.measurement_context = None
        self.phase = 'IDLE'
        self.motion_stage = 'IDLE'
        self.theta_error = None
        self.seq = 0
        self.ignore_until = 0.0
        self.previous = None
        self.response = None
        self.pulses = 0
        self.bad_response = 0
        self.pulse_cap = THETA_MAX_MS
        self.last_used = -1
        self.confirm = 0
        self.started = 0.0
        self.last_good = 0.0
        self.generation = 0
        self.retry_ms = 0
        self.retry_direction = 1
        self.mask_center = None
        self.wafer_center = None
        self.theta_done_error = None
        self.theta_recheck = None

    def clear_samples(self):
        self.history.clear()
        self.samples.clear()
        self.next_info = None
        self.stable = False
        self.theta_error = None

    def update_measurements(self, md, wd, ma, wa, now, visual_finish=False):
        # Caller holds self.lock. Only real, successfully captured frames enter here.
        self.frame_id += 1
        self.frame_time = now
        self.mask_detected = md is not None
        self.wafer_detected = wd is not None
        self.mask_center = md['center'] if md is not None else None
        self.wafer_center = wd['center'] if wd is not None else None
        self.steps_x = self.steps_y = 0
        if md is not None and wd is not None:
            self.dx = wd['center'][0] - md['center'][0]
            self.dy = wd['center'][1] - md['center'][1]
        else:
            self.dx = self.dy = None
        if self.moving or now < self.ignore_until:
            self.clear_samples()
            self.wafer_lost_count = 0
            return

        if self.phase == 'XY':
            # Completion needs observed complementary-pattern overlap, never mere loss.
            self.theta_error = None
            self.samples.clear()
            if visual_finish and md is not None and wd is not None:
                self.clear_samples()
                self.last_good = now
                if self.auto_running:
                    self.wafer_lost_count += 1
                    self.status = f'CENTER ALIGN VERIFIED {self.wafer_lost_count}/{WAFER_LOST_DONE_FRAMES}'
                return
            self.wafer_lost_count = 0
            if md is None or wd is None:
                self.clear_samples()
                self.status = 'XY: WAIT square grid / visible wafer arms'
                return
            self.last_good = now
            self.history.append((md['center'], wd['center']))
            self.stable = bool(is_stable(self.history))
        else:
            self.wafer_lost_count = 0
            if ma is None or wa is None:
                self.clear_samples()
                if self.phase == 'THETA':
                    self.status = 'THETA: waiting for a reliable key angle'
                return
            self.history.append((md['center'], wd['center']))
            self.samples.append((ma[0], wa[0]))
            self.last_good = now
            self.stable = bool(is_stable(self.history))
            if len(self.samples) == STABLE_HISTORY:
                arr = np.asarray(self.samples)
                unwrapped = arr[-1] + (arr - arr[-1] + 45) % 90 - 45
                med = np.median(unwrapped, axis=0)
                self.stable &= bool(np.max(abs(unwrapped - med)) <= THETA_JITTER_DEG)
                self.theta_error = wrap90(float(med[1] - med[0]))
            if self.phase == 'THETA':
                if len(self.samples) < STABLE_HISTORY:
                    self.status = f'THETA: stable images {len(self.samples)}/{STABLE_HISTORY}'
                elif not self.stable:
                    self.status = f'THETA: image changing; rechecking {STABLE_HISTORY} frames'
                else:
                    self.status = 'THETA: stable image ready'

        self.next_info = None
        if self.stable:
            mc, wc = get_stable_centers(self.history)
            values = compute_steps(mc, wc)
            self.dx, self.dy, self.steps_x, self.steps_y, self.quadrant, self.quadrant_fix = values
            self.next_info = (mc, wc, *values)

    def frame_context(self):
        # Caller holds self.lock. Epochs catch an entire move during one analysis.
        return (self.generation, self.phase, self.moving, self.motion_epoch,
                self.camera_epoch, self.camera_index)

    def current_packet(self, packet):
        # Caller holds self.lock; timestamps are conservative host read-start times.
        return (packet[3] == self.frame_context() and
                self.requested_camera_index == self.camera_index)

    def invalidate_camera_frame(self):
        with self.frame_ready:
            self.camera_epoch += 1
            self.latest_frame = self.overlay = None
            self.capture_time = 0.0
            self.capture_fps = 0.0
            self.clear_samples()
            self.frame_ready.notify_all()

    def capture_loop(self):
        # This is the only thread that opens, reads or releases VideoCapture.
        previous_capture = 0.0
        last_success = time.monotonic()
        try:
            while not self.exit_event.is_set():
                self.ensure_camera()
                if self.cap is None:
                    self.exit_event.wait(1.0)
                    continue
                with self.lock:
                    context = self.frame_context()
                    read_started = time.monotonic()
                ok, frame = self.cap.read()
                captured = time.monotonic()
                if not ok:
                    if captured - last_success > 1.0:
                        self.cap.release()
                        self.cap = None
                        self.invalidate_camera_frame()
                        previous_capture = 0.0
                        last_success = captured
                    self.exit_event.wait(0.05)
                    continue
                last_success = captured
                with self.frame_ready:
                    if context != self.frame_context() or self.requested_camera_index != self.camera_index:
                        continue
                    self.capture_seq += 1
                    # Replace one reference; never accumulate a frame queue.
                    self.latest_frame = (self.capture_seq, frame, read_started, context)
                    self.capture_time = captured
                    self.camera_shape = (frame.shape[1], frame.shape[0])
                    if previous_capture:
                        rate = 1.0 / max(captured - previous_capture, 1e-6)
                        self.capture_fps = rate if not self.capture_fps else .8*self.capture_fps + .2*rate
                    previous_capture = captured
                    self.frame_ready.notify_all()
        except Exception as exc:
            self.invalidate_camera_frame()
            self.fail(f'Camera error: {exc}')
        finally:
            if self.cap is not None:
                self.cap.release()
                self.cap = None

    def vision_loop(self):
        last_seq = -1
        last_completed = 0.0
        try:
            while not self.exit_event.is_set():
                with self.frame_ready:
                    self.frame_ready.wait_for(
                        lambda: self.exit_event.is_set() or
                        (self.latest_frame is not None and self.latest_frame[0] != last_seq), timeout=.25)
                    if self.exit_event.is_set():
                        break
                    packet = self.latest_frame
                    if packet is None or packet[0] == last_seq:
                        continue
                    last_seq, frame, read_started, context = packet
                    if (not self.current_packet(packet) or self.moving or
                            read_started < self.ignore_until or time.monotonic()-read_started > 1.5):
                        continue
                    source_phase = self.phase
                    token = (self.generation, self.camera_index, self.camera_epoch)
                started = time.monotonic()
                md, wd, ma, wa, visual_finish = self.key_detector.process(
                    frame, source_phase, token, stationary=True)
                now = time.monotonic()
                with self.lock:
                    self.vision_ms = (now - started)*1000
                    period = now-last_completed if last_completed else now-started
                    if last_completed:
                        self.vision_fps = 1.0/max(period, 1e-6)
                    last_completed = now
                    self.cache_used_mb = self.key_detector.bank_bytes/(1024*1024)
                    if (not self.current_packet(packet) or self.moving or
                            read_started < self.ignore_until or now-read_started > VISION_TIMEOUT_SEC):
                        # A discarded analysis must not seed the next local search.
                        self.key_detector.previous_mask = self.key_detector.previous_wafer = None
                        continue
                    # Motion/settling was checked against read-start above. A result
                    # just completed is fresh even when computation needed >0.5/1.5 s.
                    self.measurement_captured_at = read_started
                    self.measurement_context = context
                    self.update_measurements(md, wd, ma, wa, now, visual_finish)
                    ttl = min(VISION_TIMEOUT_SEC, max(1.0, 2.0*max(now-started, period)+.25))
                    self.overlay = dict(captured_at=read_started, completed_at=now,
                                        context=context, md=md, wd=wd, ma=ma, wa=wa, ttl=ttl)
                self.exit_event.wait(.01)
        except Exception as exc:
            self.fail(f'Vision error: {exc}')

    def preview_loop(self):
        smoother = PreviewSmoother(self.overlay_smoothing_ms)
        last_seq = -1
        last_preview = 0.0
        last_blank = 0.0
        while not self.exit_event.is_set():
            started = time.monotonic()
            with self.frame_ready:
                self.frame_ready.wait_for(
                    lambda: self.exit_event.is_set() or
                    (self.latest_frame is not None and self.latest_frame[0] != last_seq), timeout=.25)
                if self.exit_event.is_set():
                    break
                packet = self.latest_frame
                fresh = packet is not None and started-self.capture_time <= 1.5 and self.current_packet(packet)
                if fresh and packet[0] == last_seq:
                    continue
                if fresh:
                    last_seq, frame, read_started, context = packet
                    overlay = self.overlay
                    md = wd = ma = wa = None
                    overlay_source_time = overlay_completed_at = None
                    overlay_ttl = OVERLAY_MAX_AGE_SEC
                    if (overlay is not None and overlay['context'] == context and not self.moving
                            and started-overlay['completed_at'] <= overlay['ttl']
                            and started-overlay['captured_at'] <= VISION_TIMEOUT_SEC):
                        overlay_source_time = overlay['captured_at']
                        overlay_completed_at = overlay['completed_at']
                        overlay_ttl = overlay['ttl']
                        md, wd, ma, wa = (overlay[name] for name in ('md', 'wd', 'ma', 'wa'))
                    phase, status = self.phase, self.status
                    moving, stage = self.moving, self.motion_stage
                    err, locked_error = self.theta_error, self.theta_done_error
                    dx, dy = self.dx, self.dy
            if not fresh:
                smoother.reset()
                if started-last_blank >= .5:
                    self.make_blank_frame()
                    last_blank = started
                self.exit_event.wait(.05)
                continue
            md, wd, ma, wa = smoother.render(md, wd, ma, wa, overlay_source_time, context, time.monotonic(),
                                             completed_at=overlay_completed_at, max_age=overlay_ttl)
            display = frame.copy()
            for det, angle, name, color in ((md, ma, 'MASK', (0,0,255)), (wd, wa, 'WAFER', (0,255,0))):
                draw_detection(display, det, name, color)
                if det is None:
                    continue
                cx, cy = det['center']
                if phase in ('XY', 'DONE'):
                    cv2.drawMarker(display, (int(cx),int(cy)), color, cv2.MARKER_CROSS, 20, 2)
                elif angle is not None:
                    for a in (angle[0], angle[0]+90):
                        r = math.radians(a)
                        vx, vy = 45*math.cos(r), 45*math.sin(r)
                        cv2.line(display, (int(cx-vx),int(cy-vy)), (int(cx+vx),int(cy+vy)),color,2)
            if md is not None and wd is not None:
                cv2.line(display, tuple(int(v) for v in md['center']), tuple(int(v) for v in wd['center']), (255,255,255),1)
            put_text(display, ('ALIGNING / ' if phase in ('THETA','XY') else '')+phase+' / '+status, (12,28),scale=0.48)
            if phase in ('XY','DONE'):
                text = 'THETA ALIGN DONE / ' + ('XY CENTER ALIGN' if phase=='XY' else 'ALIGN FINISH')
            elif moving:
                text = stage+' / angle paused'
            elif err is None:
                text = 'THETA: WAIT square grid / visible wafer arms'
            else:
                text = f'theta={err:+.2f} deg'
            put_text(display,text,(12,55),(0,255,255),scale=0.62)
            if dx is not None:
                put_text(display,f'CENTER dx={dx:+.1f}px dy={dy:+.1f}px',(12,82),scale=0.58)
            if md is None and wd is None:
                put_text(display, 'LIVE / waiting for fresh key result', (12,108), (0,255,255), scale=.50)
            ok,jpg = cv2.imencode('.jpg',display,[int(cv2.IMWRITE_JPEG_QUALITY),JPEG_QUALITY])
            if ok:
                with self.jpeg_ready:
                    if self.current_packet(packet):
                        self.last_jpeg = jpg.tobytes()
                        self.jpeg_seq += 1
                        now = time.monotonic()
                        if last_preview:
                            self.preview_fps = 1.0/max(now-last_preview, 1e-6)
                        last_preview = now
                        self.jpeg_ready.notify_all()
            self.exit_event.wait(max(0, 1.0/self.preview_fps_limit-(time.monotonic()-started)))

    def fail(self, message):
        self.stop_align()
        with self.lock:
            self.phase = 'FAULT'
            self.status = message

    def start_align(self):
        with self.lock:
            if self.exit_event.is_set() or self.moving or self.auto_running:
                return
            if not self.arduino.connected():
                self.status = 'Arduino disconnected'
                return
            with self.lock:
                self.stop_event.clear()
                self.auto_running = True
                self.align_finished = False
                self.centered_count = 0
                self.wafer_lost_count = 0
                self.history.clear()
                self.next_info = None
                self.status = 'ALIGN START'
            self.generation += 1
            self.clear_samples()
            # No theta motor on the current build: go straight to XY centring.
            self.phase = 'XY' if self.skip_theta else 'THETA'
            self.theta_recheck = None
            self.theta_done_error = None
            self.previous = self.response = None
            self.retry_ms = 0
            self.retry_direction = 1
            self.pulses = self.bad_response = self.confirm = 0
            self.pulse_cap = THETA_MAX_MS
            self.last_used = self.frame_id
            self.started = self.last_good = time.monotonic()

    def stop_align(self):
        with self.lock:
            self.stop_event.set()
            self.generation += 1
            self.auto_running = False
            self.align_finished = False
            self.phase = 'STOP'
            self.motion_stage = 'STOP'
            self.status = 'STOP'
            self.clear_samples()
            self.arduino.stop()

    def reset(self):
        self.stop_align()

    def request_next_camera(self):
        self.stop_align()
        with self.lock:
            self.requested_camera_index = (self.requested_camera_index + 1) % 5
            self.status = f'Switching camera to {self.requested_camera_index}'
            self.history.clear()
            self.next_info = None
            self.align_finished = False

    def execute(self, commands, generation=None):
        with self.lock:
            if self.exit_event.is_set() or self.moving or self.stop_event.is_set() or (generation is not None and generation != self.generation):
                return False
            self.moving = True
            self.motion_epoch += 1
            self.overlay = None
            self.clear_samples()
        success = True
        error = ''
        try:
            for command, expected, timeout, settle in commands:
                with self.lock:
                    if self.exit_event.is_set() or self.stop_event.is_set():
                        success = False
                        error = 'Cancelled by STOP'
                        break
                    self.motion_stage = 'WAIT_ACK'
                    self.status = f'WAIT ACK: {command}'
                    sent = self.arduino.send(command)
                if not sent or not self.arduino.wait_done(self.stop_event, timeout, expected):
                    success = False
                    error = self.arduino.last_error or 'Serial command failed'
                    break
                with self.lock:
                    self.motion_stage = 'SETTLING'
                    self.status = f'ACK OK; settling {settle:.2f}s'
                if self.stop_event.wait(settle):
                    success = False
                    error = 'Cancelled by STOP'
                    break
        finally:
            with self.lock:
                self.clear_samples()
                self.ignore_until = time.monotonic() + POST_MOVE_GUARD_SEC
                self.last_good = time.monotonic()
                self.moving = False
                self.motion_epoch += 1
                self.motion_stage = 'REACQUIRE' if success else 'STOP'
                if success and not self.stop_event.is_set():
                    self.status = 'ACK OK; WAIT fresh stable image'
        if not success:
            self.fail(error)
        return success

    def move_xy_sequential(self, steps_x, steps_y, generation=None):
        commands = [(f'{arduino_axis(axis)} {steps}', 'DONE', 25.0, AXIS_SETTLE_SEC) for axis, steps in (('X', steps_x), ('Y', steps_y)) if steps != 0]
        return self.execute(commands, generation)

    def manual_move(self, axis, steps):
        with self.lock:
            if self.exit_event.is_set() or self.moving or self.auto_running or axis not in ('X', 'Y'):
                return
            self.auto_running = True
            self.phase = 'MANUAL'
            self.stop_event.clear()
            self.generation += 1
            generation = self.generation

        def worker():
            if MANUAL_STEPS:
                steps = MANUAL_STEPS if steps > 0 else -MANUAL_STEPS
            ok = self.move_xy_sequential(clamp_steps(steps) if axis == 'X' else 0, clamp_steps(steps) if axis == 'Y' else 0, generation)
            with self.lock:
                if generation == self.generation:
                    self.auto_running = False
                    self.status = 'MANUAL DONE' if ok else 'MANUAL STOP'
        manual_worker = threading.Thread(target=worker, name='manual_move', daemon=True)
        with self.lock:
            self.workers = [thread for thread in self.workers if thread.is_alive()]
            self.workers.append(manual_worker)
            manual_worker.start()

    def begin_theta_recheck(self, before, direction, duration, err, reason):
        with self.lock:
            self.theta_recheck = (before, direction, duration, err)
            self.response = None
            self.retry_ms = 0
            self.confirm = 0
            self.pulse_cap = max(THETA_MIN_MS, min(self.pulse_cap, duration // 2))
            self.clear_samples()
            self.motion_stage = 'THETA_RECHECK'
            self.status = f'THETA RECHECK - {reason}; automatic retry'
        print(f'THETA {reason}: remeasure then continue, cap={self.pulse_cap}ms', flush=True)

    def auto_loop(self):
        while not self.exit_event.wait(0.02):
            with self.lock:
                if not self.auto_running or self.moving or self.phase == 'MANUAL':
                    continue
                now = time.monotonic()
                stale = now - self.last_good > VISION_TIMEOUT_SEC
                expired = now - self.started > THETA_TIMEOUT_SEC and self.phase == 'THETA'
                info, err, fid = (self.next_info, self.theta_error, self.frame_id)
                generation = self.generation
                if now - self.capture_time > 1.5:
                    self.fail('Camera unavailable')
                    continue
                result_current = (now-self.frame_time <= 1.5 and
                                  now-self.measurement_captured_at <= VISION_TIMEOUT_SEC and
                                  self.measurement_context == self.frame_context() and
                                  self.measurement_captured_at >= self.ignore_until)
                if result_current and self.phase == 'XY' and self.wafer_lost_count >= WAFER_LOST_DONE_FRAMES:
                    self.finish_align('ALIGN FINISH - FOUR-CORNER CENTERS VERIFIED')
                    self.phase = 'DONE'
                    self.motion_stage = 'DONE'
                    continue
                if stale or expired:
                    self.fail('Camera/key unavailable or theta timeout')
                    continue
                if now < self.ignore_until or not result_current:
                    continue
                if fid == self.last_used:
                    continue
                if self.phase == 'XY':
                    if info is None:
                        continue
                    self.last_used = fid
                    sx, sy = info[4], info[5]
                    if sx == 0 and sy == 0:
                        self.status = 'XY CENTERED - verifying fresh corner/wafer centers'
                        continue
                else:
                    if info is None or err is None:
                        self.confirm = 0
                        continue
                    self.last_used = fid
                    sx = sy = None
            if sx is not None:
                self.move_xy_sequential(sx, sy, generation)
                continue
            if abs(err) <= THETA_TOL_DEG:
                with self.lock:
                    if generation != self.generation or not self.auto_running or self.phase != 'THETA':
                        continue
                    self.confirm += 1
                    self.status = f'THETA VERIFY {self.confirm}/{THETA_CONFIRM_FRAMES}'
                    if self.confirm >= THETA_CONFIRM_FRAMES:
                        self.phase = 'XY'
                        self.theta_done_error = err
                        self.theta_recheck = None
                        self.previous = None
                        self.retry_ms = 0
                        self.wafer_lost_count = 0
                        self.confirm = 0
                        self.clear_samples()
                        self.status = 'THETA DONE - STARTING XY'
                continue
            if self.theta_recheck is not None:
                before, direction, duration, observed = self.theta_recheck
                if abs(wrap90(err - observed)) > THETA_JITTER_DEG:
                    self.theta_recheck = (before, direction, duration, err)
                    with self.lock:
                        self.clear_samples()
                        self.status = 'THETA RECHECK - waiting for settled angle'
                    continue
                change = wrap90(err - before)
                self.response = change / (direction * duration) if abs(change) >= 0.25 else None
                self.theta_recheck = None
                self.bad_response = 0
                print(f'THETA recheck OK: error={err:+.2f}, response={self.response}, cap={self.pulse_cap}ms; continuing', flush=True)
            if self.previous is not None:
                before, direction, duration = self.previous
                change = wrap90(err - before)
                self.previous = None
                if abs(change) > 12:
                    self.begin_theta_recheck(before, direction, duration, err, 'large movement')
                    continue
                print(f'THETA observed: before={before:+.2f}, after={err:+.2f}, change={change:+.2f} deg, pulse={duration}ms', flush=True)
                if abs(change) < 0.25:
                    self.bad_response += 1
                    if duration >= THETA_MAX_MS:
                        self.fail(f'No measured wafer rotation at {THETA_MAX_MS}ms; inspect loaded shaft/coupling and power')
                        continue
                    self.retry_ms = next(ms for ms in THETA_PROBE_MS if ms > duration)
                    self.retry_direction = direction
                    print(f'THETA no response: next pulse {self.retry_ms}ms, then remeasure', flush=True)
                else:
                    rate = change / (direction * duration)
                    if self.response is not None and rate * self.response < 0:
                        self.begin_theta_recheck(before, direction, duration, err, 'response changed')
                        continue
                    self.response = rate if self.response is None else 0.6 * self.response + 0.4 * rate
                    self.bad_response = 0
                    self.retry_ms = 0
                    if abs(err) > abs(before) + 0.5 or err * before < 0:
                        self.pulse_cap = max(THETA_MIN_MS, self.pulse_cap // 2)
            self.confirm = 0
            if self.pulses >= THETA_MAX_PULSES:
                self.fail('Theta pulse limit: inspect calibration')
                continue
            if self.retry_ms:
                direction, duration = self.retry_direction, self.retry_ms
                self.retry_ms = 0
            elif self.response is None:
                direction, duration = 1, min(THETA_PROBE_MS[0], self.pulse_cap)
            else:
                direction = 1 if -err / self.response > 0 else -1
                gain = THETA_FINE_GAIN if abs(err) <= THETA_FINE_RANGE_DEG else THETA_MOVE_GAIN
                duration = int(np.clip(gain * abs(err / self.response), THETA_MIN_MS, self.pulse_cap))
            self.seq += 1
            with self.lock:
                self.status = f'THETA pulse {direction:+d}, {duration} ms; err={err:+.2f}'
            if self.execute([(f'T {self.seq} {direction} {duration}', f'DONE T {self.seq}', 2.0, THETA_SETTLE_SEC)], generation):
                self.previous = (err, direction, duration)
                self.pulses += 1

    def start(self):
        for target in (self.capture_loop, self.vision_loop, self.preview_loop, self.auto_loop):
            worker = threading.Thread(target=target, name=target.__name__, daemon=True)
            self.workers.append(worker)
            worker.start()

    def open_camera(self, index):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 30)
        # Best effort only: some DirectShow drivers ignore this property.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def ensure_camera(self):
        with self.lock:
            requested = self.requested_camera_index
        if self.cap is not None and requested == self.camera_index:
            return
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.invalidate_camera_frame()
        self.cap = self.open_camera(requested)
        with self.lock:
            self.camera_index = requested
            self.history.clear()
            self.next_info = None
            if self.cap is None:
                self.status = f'Camera {requested} open failed'
            else:
                self.status = f'Camera {requested} opened'

    def make_blank_frame(self):
        frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        with self.lock:
            text = self.status
        put_text(frame, text, (32, 64), (0, 255, 255), 0.9, 2)
        ok, encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if ok:
            with self.jpeg_ready:
                self.last_jpeg = encoded.tobytes()
                self.jpeg_seq += 1
                self.jpeg_ready.notify_all()

    def finish_align(self, message):
        self.arduino.stop()
        with self.lock:
            self.auto_running = False
            self.align_finished = True
            self.moving = False
            self.centered_count = 0
            self.history.clear()
            self.next_info = None
            self.status = message

    def get_status(self):
        with self.lock:
            return {'version': APP_VERSION, 'axis_mapping': f"image X -> Arduino {arduino_axis('X')}; image Y -> Arduino {arduino_axis('Y')}", 'status': self.status, 'camera_index': self.camera_index, 'arduino_connected': self.arduino.connected(), 'arduino_port': self.arduino.port or '-', 'auto_running': self.auto_running, 'align_finished': self.align_finished, 'moving': self.moving, 'stable': self.stable, 'history_len': len(self.history), 'stable_history': STABLE_HISTORY, 'mask_detected': self.mask_detected, 'wafer_detected': self.wafer_detected, 'wafer_lost_count': self.wafer_lost_count, 'dx': self.dx, 'dy': self.dy, 'steps_x': self.steps_x, 'steps_y': self.steps_y, 'quadrant': self.quadrant, 'quadrant_fix': self.quadrant_fix, 'phase': self.phase, 'theta_aligned': self.phase in ('XY', 'DONE'), 'theta_error': self.theta_error, 'theta_done_error': self.theta_done_error, 'mask_center': self.mask_center, 'wafer_center': self.wafer_center, 'motion_stage': self.motion_stage, 'last_rx': self.arduino.last_rx, 'serial_error': self.arduino.last_error,
                    'capture_fps': self.capture_fps if time.monotonic()-self.capture_time < 1.5 else 0,
                    'preview_fps': self.preview_fps if time.monotonic()-self.capture_time < 1.5 else 0,
                    'vision_fps': self.vision_fps if not self.moving and time.monotonic()-self.frame_time < VISION_TIMEOUT_SEC else 0,
                    'vision_ms': self.vision_ms, 'cache_mb': self.cache_used_mb,
                    'cache_limit_mb': self.cache_limit_mb, 'camera_shape': self.camera_shape,
                    'overlay_smoothing_ms': self.overlay_smoothing_ms,
                    'result_age_ms': (time.monotonic()-self.frame_time)*1000 if self.frame_time else None,
                    'measurement_age_ms': (time.monotonic()-self.measurement_captured_at)*1000 if self.measurement_captured_at else None}

    def get_jpeg(self):
        with self.lock:
            return self.last_jpeg

    def shutdown(self):
        self.exit_event.set()
        self.stop_align()
        with self.frame_ready:
            self.frame_ready.notify_all()
            self.jpeg_ready.notify_all()
        for worker in self.workers:
            if worker is not threading.current_thread():
                worker.join(timeout=.75)
        self.arduino.close()

class RequestHandler(BaseHTTPRequestHandler):
    server_version = 'MaskAlignerHTTP/1.0'

    def log_message(self, fmt, *args):
        return

    @property
    def app(self):
        return self.server.app

    def send_json(self, payload, status=200):
        data = json.dumps(payload).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == '/':
            data = HTML_PAGE.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if parsed.path == '/api/status':
            self.send_json(self.app.get_status())
            return
        if parsed.path == '/video.mjpg':
            self.send_response(200)
            self.send_header('Age', '0')
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            self.connection.settimeout(2.0)
            last_seq = -1
            while not self.app.exit_event.is_set():
                with self.app.jpeg_ready:
                    self.app.jpeg_ready.wait_for(
                        lambda: self.app.exit_event.is_set() or
                        (self.app.last_jpeg is not None and self.app.jpeg_seq != last_seq),
                        timeout=1.0)
                    if self.app.exit_event.is_set():
                        break
                    jpg = self.app.last_jpeg
                    if jpg is None or self.app.jpeg_seq == last_seq:
                        continue
                    last_seq = self.app.jpeg_seq
                try:
                    self.wfile.write(b'--frame\r\n')
                    self.wfile.write(b'Content-Type: image/jpeg\r\n')
                    self.wfile.write(f'Content-Length: {len(jpg)}\r\n\r\n'.encode('ascii'))
                    self.wfile.write(jpg)
                    self.wfile.write(b'\r\n')
                    self.wfile.flush()
                except (OSError, TimeoutError):
                    break
            return
        self.send_error(404)

    def do_POST(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if parsed.path == '/api/start':
            self.app.start_align()
            self.send_json({'ok': True})
            return
        if parsed.path == '/api/stop':
            self.app.stop_align()
            self.send_json({'ok': True})
            return
        if parsed.path == '/api/reset':
            self.app.reset()
            self.send_json({'ok': True})
            return
        if parsed.path == '/api/camera_next':
            self.app.request_next_camera()
            self.send_json({'ok': True})
            return
        if parsed.path == '/api/manual':
            axis = query.get('axis', [''])[0]
            steps_text = query.get('steps', ['0'])[0]
            try:
                steps = int(steps_text)
            except ValueError:
                self.send_json({'ok': False, 'error': 'steps must be integer'}, 400)
                return
            self.app.manual_move(axis, steps)
            self.send_json({'ok': True})
            return
        self.send_error(404)

CALIBRATED_KEYS = ('SWAP_XY_COMMANDS', 'MOTOR_SIGN_X', 'MOTOR_SIGN_Y', 'STEPS_PER_PIXEL_X',
                   'STEPS_PER_PIXEL_Y', 'MAX_STEPS_PER_AXIS', 'DEAD_BAND_PX', 'MANUAL_STEPS')


def load_calibration(path):
    """Override the stage constants above with calibrate_xy.py output, if present."""
    from pathlib import Path
    path = Path(path)
    if not path.is_file():
        print(f'No {path.name}: using built-in stage constants (run calibrate_xy.py)')
        return
    data = json.loads(path.read_text())
    for key in CALIBRATED_KEYS:
        if key in data:
            globals()[key] = data[key]
    print(f'Loaded {path.name} ({data.get("_measured_at", "?")}): ' +
          ', '.join(f'{k}={data[k]}' for k in CALIBRATED_KEYS if k in data))


def find_key_camera(max_index=4):
    """Index of the first camera whose frame shows both keys (USB indices shift on replug)."""
    detector = CornerKeys()
    for index in range(max_index):
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        found = False
        for _ in range(10):
            ok, frame = cap.read()
            if ok:
                md, wd, *_ = detector.process(frame, 'IDLE', token=index)
                if md is not None and wd is not None:
                    found = True
                    break
        cap.release()
        if found:
            print(f'Camera auto: index {index} sees both keys')
            return index
    print(f'Camera auto: no camera sees both keys; using {CAMERA_INDEX}')
    return CAMERA_INDEX


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', default=str(CAMERA_INDEX),
                        help="camera index, or 'auto' to pick the camera that sees both keys")
    parser.add_argument('--port', type=int, default=HTTP_PORT)
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--arduino-port', default=None,
                        help='serial port, e.g. COM3 or /dev/cu.usbmodemXXXX (default: auto). --port is the web page port')
    parser.add_argument('--skip-theta', action='store_true',
                        help='start alignment at XY (no theta motor connected)')
    parser.add_argument('--calibration', default=str(Path(__file__).resolve().parent/'calibration.json'),
                        help='calibrate_xy.py output to load (default: calibration.json next to this file)')
    parser.add_argument('--detector', choices=('opencv', 'opencv-v4', 'sam2', 'hybrid', 'yolo-sam2', 'opencv-sam2'), default='opencv',
                        help='opencv: rule-based squares + subpixel edge fit; opencv-v4: original integer-pixel; sam2/hybrid: SAM2.1 hiera-small segmentation; yolo-sam2/opencv-sam2: YOLO or OpenCV boxes -> SAM2')
    parser.add_argument('--cache-mb', type=int, default=PATTERN_CACHE_MB,
                        help='Template array cache limit in MiB (default: 64)')
    parser.add_argument('--preview-fps', type=int, default=PREVIEW_FPS,
                        help='Preview FPS limit only; recognition is not throttled (1-30)')
    parser.add_argument('--opencv-threads', type=int, default=0,
                        help='0: keep OpenCV default; positive: limit internal workers')
    parser.add_argument('--overlay-smoothing-ms', type=float, default=OVERLAY_SMOOTHING_MS,
                        help='Display interpolation time constant in ms; 0 disables it (default: 120)')
    args = parser.parse_args()
    if args.cache_mb < 0 or not 1 <= args.preview_fps <= 30 or args.opencv_threads < 0:
        parser.error('cache-mb >= 0, preview-fps 1..30, opencv-threads >= 0 required')
    if not 0 <= args.overlay_smoothing_ms <= 1000:
        parser.error('overlay-smoothing-ms must be 0..1000')
    if args.opencv_threads:
        cv2.setNumThreads(args.opencv_threads)
    load_calibration(args.calibration)
    if args.arduino_port:
        global ARDUINO_PORT
        ARDUINO_PORT = args.arduino_port
    args.camera = find_key_camera() if args.camera == 'auto' else int(args.camera)
    httpd = ThreadingHTTPServer((HOST, args.port), RequestHandler)
    app = MaskAlignerApp(camera_index=args.camera, cache_mb=args.cache_mb, preview_fps=args.preview_fps, overlay_smoothing_ms=args.overlay_smoothing_ms, detector=args.detector, skip_theta=args.skip_theta)
    httpd.app = app
    app.start()
    url = f'http://{HOST}:{args.port}'
    print('')
    print(f'Mask Aligner Web UI / {APP_VERSION} / detector={args.detector}')
    print(f'Open: {url}')
    print('Buttons: START ALIGN / STOP / RESET / NEXT CAMERA')
    print('Press Ctrl+C in this terminal to quit.')
    print('')
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.shutdown()
        httpd.server_close()

HTML_PAGE = r"""<!doctype html>
<html lang="ko">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mask Aligner v4</title>
<style>
body{font-family:Arial,"Malgun Gothic",sans-serif;background:#eef2f6;color:#17202a;margin:20px}
main{display:flex;gap:20px;flex-wrap:wrap}.video{position:relative;max-width:800px;flex:1;min-width:320px}
img{width:100%;background:#111;border-radius:8px}aside{background:white;padding:18px;width:320px;border-radius:8px}
button{padding:12px;margin:4px;border:0;border-radius:5px;cursor:pointer;background:#dce5ef}
.start{background:#168a50;color:white}.stop{background:#c73737;color:white}
#finish{display:none;position:absolute;inset:0;align-items:center;justify-content:center;background:#0008;color:white;font-size:40px;font-weight:bold}
pre{white-space:pre-wrap;line-height:1.7}small{display:block;margin-top:16px;line-height:1.7}
</style>
<h2>Mask Aligner — Theta → X → Y · v4</h2>
<main><div class="video"><img src="/video.mjpg"><div id="finish">ALIGN FINISH</div></div>
<aside>
<button class="start" onclick="post('/api/start')">START ALIGN</button>
<button class="stop" onclick="post('/api/stop')">STOP</button>
<button onclick="post('/api/reset')">RESET</button>
<button onclick="post('/api/camera_next')">NEXT CAMERA</button>
<div id="thetaDone" role="status" hidden style="padding:14px;margin:12px 0;background:#dcfce7;color:#166534;border:1px solid #86efac;border-radius:6px;font-weight:bold"></div>
<pre id="status">연결 중...</pre>
<small id="performance"></small>
<small>인식 표시는 작은 떨림을 완화해 보여줍니다. 정렬 제어와 위의 측정 수치는 원본 인식 결과를 사용합니다.</small>
<p>화면 기준 Manual X / Y</p>
<small>수동 +/−는 매핑된 Arduino 채널의 스텝 부호입니다. 화면 이동 방향은 장비에 따라 다릅니다.</small>
<button onclick="manual('X',-30)">X -</button><button onclick="manual('X',30)">X +</button>
<button onclick="manual('Y',-30)">Y -</button><button onclick="manual('Y',30)">Y +</button>
<small>START 한 번으로 세타 구동·검증을 반복하고, ±4° 이내 확인 후 X → Y 중심 정렬로 자동 전환합니다.<br>X·Y에서는 각도 검출을 요구하지 않습니다.<br>MASK의 검은 사각형 4개 배치와 WAFER의 남은 십자 팔로 중심을 구합니다. 각 축의 중심 오차가 6px 미만인 상태로 8프레임 연속 확인되면 ALIGN FINISH입니다. MASK 3/4는 3개 실측과 1개 위치 복원을 뜻합니다.</small>
</aside></main>
<script>
async function post(path){await fetch(path,{method:'POST'});await refresh()}
function manual(axis,steps){return post(`/api/manual?axis=${axis}&steps=${steps}`)}
let refreshing=false;
async function refresh(){if(refreshing)return;refreshing=true;try{
const d=await(await fetch('/api/status',{cache:'no-store',signal:AbortSignal.timeout(3000)})).json();
const n=x=>x==null?'-':x.toFixed(2);
const banner=document.getElementById('thetaDone');
banner.hidden=!d.theta_aligned;
banner.textContent=d.phase==='DONE'?'세타 정렬 완료 · X·Y 정렬 완료':'세타 정렬 완료 — X·Y 정렬 시작 / 진행 중';
document.getElementById('status').textContent=`${d.phase}: ${d.status}\nCamera: ${d.camera_index}\nArduino: ${d.arduino_connected?d.arduino_port:'OFF'}\nStage: ${d.motion_stage}\nRX: ${d.last_rx||'-'}\nSerial: ${d.serial_error||'OK'}\nTheta: ${d.theta_aligned?'완료 시 '+n(d.theta_done_error):n(d.theta_error)} deg\nMASK 중심: ${d.mask_center?d.mask_center.map(n).join(', '):'-'}\nWAFER 중심: ${d.wafer_center?d.wafer_center.map(n).join(', '):'-'}\n중심 정렬 확인: ${d.wafer_lost_count}/8\ndx: ${n(d.dx)} px / dy: ${n(d.dy)} px\n축 매핑: ${d.axis_mapping}\n화면 X 보정: ${d.steps_x} / 화면 Y 보정: ${d.steps_y}\nMASK: ${d.mask_detected?'OK':'NO'}\nWAFER: ${d.wafer_detected?'OK':'NO'}\nStable: ${d.stable} (${d.history_len}/${d.stable_history})`;
document.getElementById('finish').style.display=d.align_finished?'flex':'none';
document.getElementById('performance').textContent=`촬영 ${n(d.capture_fps)} / 화면 ${n(d.preview_fps)} / 인식 ${n(d.vision_fps)} FPS · 인식 ${n(d.vision_ms)} ms · 패턴 캐시 ${n(d.cache_mb)} / ${d.cache_limit_mb} MiB · 해상도 ${d.camera_shape?d.camera_shape.join('×'):'-'} · 촬영 후 ${n(d.measurement_age_ms)} ms / 결과 갱신 후 ${n(d.result_age_ms)} ms · 표시 완화 ${d.overlay_smoothing_ms} ms`;
}catch(e){document.getElementById('status').textContent='UI disconnected'}finally{refreshing=false}}
setInterval(refresh,300);refresh();
</script></html>
"""

if __name__ == "__main__":
    main()
