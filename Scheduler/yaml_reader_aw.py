import yaml

import requests
import time
import cv2 as cv
import yaml
from multiprocessing import Process
import imghdr
import subprocess
import json
import tensorflow as tf
from Edge.edge_server import Server
import os


res = {'0': (360, 360), '1': (720, 600), '2': (960, 720), '3':(1350, 900), '4':(1920, 1080)}
fps = {'0': '2', '1': '3', '2': '5', '3': '10', '4': '15'}
par = {'0': '0',
       '1': 'FirstStageFeatureExtractor/resnet_v1_101/resnet_v1_101/pool1/MaxPool',
       '2': 'FirstStageFeatureExtractor/resnet_v1_101/resnet_v1_101/block2/unit_1/bottleneck_v1/Relu',
       '3': 'FirstStageFeatureExtractor/resnet_v1_101/resnet_v1_101/block3/unit_1/bottleneck_v1/Relu',
       '4': 'FirstStageFeatureExtractor/resnet_v1_101/resnet_v1_101/block3/unit_23/bottleneck_v1/Relu',
       '5': 'SecondStageFeatureExtractor/resnet_v1_101/block4/unit_1/bottleneck_v1/Relu'
}
def init(session, addr):
# prepare headers for http request
    start_time = time.time()
    response = session.post(addr+"/init")
    print(response.text)
    end_time = time.time()
    # print(end_time-start_time)

def send_image(session, addr, image_path, r, p):
    start_time = time.time()
    # image = cv2.imread(image_path)
    # with tf.Session() as sess:
    # raw_data = tf.gfile.FastGFile(image_path, 'rb').read()
    # img_data = tf.image.decode_png(raw_data)
    # img_resized = tf.image.resize_images(img_data, [r[0], r[1]], method=0)
    # print(img_resized.get_shape(), type(img_resized))
    # resized_image = cv2.resize(image, (r[0], r[1]), interpolation=cv2.INTER_CUBIC)
    image_to_send = {"media": open('img_%s.png'%(str(r[1])), 'rb')}
    end_time = time.time()
    # print('resize image:', end_time-start_time)
    start_time = time.time()
    add_info = {'point': p}  # 这里是可以继续添加字段
    # send http request with image and receive response
    url = addr + '/image'
    response = session.post(url, data = add_info, files=image_to_send)
    end_time = time.time()
    # print('send image:', end_time-start_time)
    # decode response
    # print(json.loads(response.text))

    # print(end_time-start_time)

def create_model(session, addr):
    url = addr + '/create'
    response = session.post(url)
    print(response.text)


def gpu_monitor(session, addr, index, flag):
    url = addr + '/gpu_monitor'
    add_info = {'index': str(index), 'function': flag}
    response = session.post(url, data=add_info)
    print(response.text)

def kill_app(session, addr):
    url = addr + '/kill'
    response = session.post(url)
    print(response.text)

# send a video
def send_video(session, addr):

    encoded_video_path = 'video/output000.mp4'
    video_to_send = {"media": open(encoded_video_path, "rb")}
    # send http request with image and receive response
    start_time = time.time()
    url = addr + '/video'
    response = session.post(url, files=video_to_send)
    # decode response
    # print(json.loads(response.text))
    print(response.text)
    end_time = time.time()
    print(end_time-start_time)


def processing_video(camera_id, ip, port, r, f, p, thr):
    print(camera_id, 'config', r, f, 'processing_video')
    sst = time.time()
    addr = 'http://%s:%s'%(ip, str(port))
    session = requests.Session()
    init(session, addr)
    fps = int(f)
    skip = int(30/fps)
    cnt = 1
    inference_time = []
    while(cnt <= thr):
        start_time = time.time()
        tmp = str(cnt).zfill(8)
        image_path = '/data/rch/video_set/camera_%s_1080/%s.png'%(str(camera_id+1), tmp)
        send_image(session, addr, image_path, r, p)   # 这里可能需要进行修改  采用processing形式
        end_time = time.time()
        complete_time = end_time-start_time

        inference_time.append(complete_time)
        delta = (1/float(f))-complete_time
        # print(complete_time, delta)
        if delta > 0:
            time.sleep(delta)
        cnt = cnt+skip
    res_dict = {}
    eet = time.time()
    print(camera_id, 'processing_video', eet-sst)
    res_dict['ID'] = camera_id
    res_dict['inference time'] = inference_time
    file_name = 'results/camera_%s_%s.json'%(camera_id, 'cloud_model_in_cloud')
    json.dump(res_dict, open(file_name, 'w'))

def processing_video_edge_model_in_edge(camera_id, r, f, p, thr):
    sst = time.time()
    fps = int(f)
    skip = int(30 / fps)
    cnt = 1
    inference_time = []
    model_path = '/data/rch/iwqos/Scheduler/Edge/frozen_inference_graph_edge.pb'
    gpu_id = int(camera_id)/4
    server = Server(str(gpu_id), model_path, nframes=None)
    image = cv.imread('img_%s.png'%(str(r[1])))
    # image = cv.imread('img_900.png')
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    while (cnt <= thr):
        start_time = time.time()
        tmp = str(cnt).zfill(8)
        image_path = '/data/rch/video_set/camera_1_%s/%s.png' % (str(r[1]), tmp)

        # raw_data = tf.gfile.FastGFile(image_path, 'rb').read()
        # img_data = tf.image.decode_png(raw_data)
        # img_resized = tf.image.resize_images(img_data, [r[0], r[1]], method=0)
        results = server.perform_single_image(image, p, binary_image=False)
        end_time = time.time()
        complete_time =  end_time-start_time
        inference_time.append(complete_time)
        if (1/fps)-complete_time > 0:
            time.sleep((1/fps)-complete_time)
        # print("time", complete_time)
        cnt = cnt+skip
    eet = time.time()
    print(camera_id, 'processing_video_edge_model_in_edge', eet-sst)
    res_dict = {}
    res_dict['ID'] = camera_id
    res_dict['inference time'] = inference_time
    file_name = 'results/camera_%s_%s.json'%(camera_id, 'edge_model_in_edge')
    json.dump(res_dict, open(file_name, 'w'))



def processing_video_cloud_model_in_edge(camera_id, r, f, p, thr):
    print(camera_id, 'config', r, f, 'processing_video_cloud_model_in_edge')
    sst = time.time()
    fps = int(f)
    skip = int(30 / fps)
    cnt = 1
    inference_time = []
    model_path = '/data/rch/iwqos/Scheduler/Edge/frozen_inference_graph.pb'
    gpu_id = int(camera_id)/4
    server = Server(str(gpu_id), model_path, nframes=None)
    image = cv.imread('img_%s.png'%(str(r[1])))
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    while (cnt <= thr):
        start_time = time.time()
        tmp = str(cnt).zfill(8)
        image_path = '/data/rch/video_set/camera_1_1080/%s.png' % (tmp)
        # print(image_path)
        # raw_data = tf.gfile.FastGFile(image_path, 'rb').read()
        # img_data = tf.image.decode_png(raw_data)
        # img_resized = tf.image.resize_images(img_data, [r[0], r[1]], method=0)
        results = server.perform_single_image(image, p, binary_image=False)
        end_time = time.time()
        complete_time =  end_time-start_time
        inference_time.append(complete_time)
        if (1/fps)-complete_time > 0:
            time.sleep((1/fps)-complete_time)
        # print("time", complete_time)
        cnt = cnt+skip
    eet = time.time()
    print(camera_id, 'processing_video_cloud_model_in_edge', eet-sst)
    res_dict = {}
    res_dict['ID'] = camera_id
    res_dict['inference time'] = inference_time
    file_name = 'results/camera_%s_%s.json'%(camera_id, 'cloud_model_in_edge')
    json.dump(res_dict, open(file_name, 'w'))

def  main_optimal(yaml_name, index):
    f = open(yaml_name, 'r')
    content = yaml.load(f)
    print(content)
    print(yaml_name)
    thr = 9000
    dur = thr / 30 / 5
    addr = 'http://203.91.121.212:5000'
    session = requests.Session()
    init(session, addr)
    gpu_monitor(session, addr, index, 'optimal')
    cmd = 'nvidia-smi dmon -i 0 -d 5 -s u -c {} -f results/gpu_{}.log'.format(int(dur + 6), 0)
    print(cmd)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    cmd = 'nvidia-smi dmon -i 1 -d 5 -s u -c {} -f results/gpu_{}.log'.format(int(dur + 6), 1)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    time.sleep(2)

    for key in content:
        if len(str(key)) is 1:
            config = content[key][1]
            print(config)
            if config[0] is '0' and config[3] is '1':
                config = config[4:]
            if config[0] is '1' and config[3] is '0':
                config =  config[1:3]

            if len(config) is 2:
                r = res[config[0]]
                f = fps[config[1]]
                p = '0'
                proc = Process(target=processing_video_edge_model_in_edge, args=(int(key), r, f, p, thr,))
                proc.start()

            if len(config) is 3:
                r = res[config[0]]
                f = fps[config[1]]
                p = par[config[2]]
                cport = 5000 + int(key)
                cip = '203.91.121.212'
                proc1 = Process(target=processing_video, args=(int(key), cip, cport, r, f, p, thr,))
                proc1.start()
                if p != '0':
                    proc2 = Process(target=processing_video_cloud_model_in_edge, args=(int(key), r, f, p, thr,))
                    proc2.start()
            if len(config) is 7:
                er = res[config[1]]
                ef = fps[config[2]]
                cr = res[config[4]]
                cf = fps[config[5]]
                cp = par[config[6]]
                eip = '203.91.121.211'
                p = '0'
                proc1 = Process(target=processing_video_edge_model_in_edge, args=(int(key), er, ef, p, thr,))
                proc1.start()

                cip = '203.91.121.212'
                cport = 5000 + int(key)
                proc2 = Process(target=processing_video, args=(int(key), cip, cport, cr, cf, cp, thr,))
                proc2.start()
                if cp != '0':
                    proc3 = Process(target=processing_video_cloud_model_in_edge, args=(int(key), cr, cf, cp, thr,))
                    proc3.start()

def main(yaml_name, index):
    f = open(yaml_name,  'r')
    content = yaml.load(f)
    print(content)
    print(yaml_name)
    thr = 9000
    dur = thr/30/5
    addr = 'http://203.91.121.212:5000'
    session = requests.Session()
    init(session, addr)
    gpu_monitor(session, addr, index, 'our')
    cmd = 'nvidia-smi dmon -i 0 -d 5 -s u -c {} -f results/gpu_{}.log'.format(int(dur+6), 0)
    print(cmd)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    cmd = 'nvidia-smi dmon -i 1 -d 5 -s u -c {} -f results/gpu_{}.log'.format(int(dur+6), 1)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    time.sleep(2)
    for key in content:
        # print(key)
        if len(key) is 4:
            config = content[key]['config']
            # print(config)
            if len(config) is 2:
                r = res[config[0]]
                f = fps[config[1]]
                p = '0'

                proc = Process(target = processing_video_edge_model_in_edge, args = (int(key), r, f, p, thr,))
                proc.start()

            if len(config) is 3:
                r = res[config[0]]
                f = fps[config[1]]
                p = par[config[2]]
                cport = 5000 + int(key)
                cip = '203.91.121.212'
                proc1 = Process(target = processing_video, args = (int(key), cip, cport, r, f, p, thr,))
                proc1.start()
                if p !='0':
                    proc2 = Process(target = processing_video_cloud_model_in_edge, args = (int(key), r, f, p, thr,))
                    proc2.start()
            if len(config) is 7:
                er = res[config[1]]
                ef = fps[config[2]]
                cr = res[config[4]]
                cf = fps[config[5]]
                cp = par[config[6]]
                eip = '203.91.121.211'
                p = '0'
                proc1 = Process(target = processing_video_edge_model_in_edge, args = (int(key), er, ef, p, thr,))
                proc1.start()

                cip = '203.91.121.212'
                cport = 5000 + int(key)
                proc2 = Process(target = processing_video, args = (int(key), cip, cport, cr, cf, cp, thr,))
                proc2.start()
                if cp != '0':
                    proc3 = Process(target=processing_video_cloud_model_in_edge, args=(int(key), cr, cf, cp, thr,))
                    proc3.start()

def main_ourp(yaml_name, index):
    f = open(yaml_name,  'r')
    content = yaml.load(f)
    print(content)
    print(yaml_name)
    thr = 9000
    dur = thr/30/5
    addr = 'http://203.91.121.212:5000'
    session = requests.Session()
    init(session, addr)
    gpu_monitor(session, addr, index, 'ourp')
    cmd = 'nvidia-smi dmon -i 0 -d 5 -s u -c {} -f results/gpu_{}.log'.format(int(dur+6), 0)
    print(cmd)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    cmd = 'nvidia-smi dmon -i 1 -d 5 -s u -c {} -f results/gpu_{}.log'.format(int(dur+6), 1)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    time.sleep(2)
    for key in content:
        # print(key)
        if len(key) is 4:
            config = content[key]['config']
            # print(config)
            if len(config) is 2:
                r = res[config[0]]
                f = fps[config[1]]
                p = '0'

                proc = Process(target = processing_video_edge_model_in_edge, args = (int(key), r, f, p, thr,))
                proc.start()

            if len(config) is 3:
                r = res[config[0]]
                f = fps[config[1]]
                p = par[config[2]]
                cport = 5000 + int(key)
                cip = '203.91.121.212'
                proc1 = Process(target = processing_video, args = (int(key), cip, cport, r, f, p, thr,))
                proc1.start()
                if p !='0':
                    proc2 = Process(target = processing_video_cloud_model_in_edge, args = (int(key), r, f, p, thr,))
                    proc2.start()
            if len(config) is 7:
                er = res[config[1]]
                ef = fps[config[2]]
                cr = res[config[4]]
                cf = fps[config[5]]
                cp = par[config[6]]
                eip = '203.91.121.211'
                p = '0'
                proc1 = Process(target = processing_video_edge_model_in_edge, args = (int(key), er, ef, p, thr,))
                proc1.start()

                cip = '203.91.121.212'
                cport = 5000 + int(key)
                proc2 = Process(target = processing_video, args = (int(key), cip, cport, cr, cf, cp, thr,))
                proc2.start()
                if cp != '0':
                    proc3 = Process(target=processing_video_cloud_model_in_edge, args=(int(key), cr, cf, cp, thr,))
                    proc3.start()
def  main_videoedge(yaml_name, index):
    f = open(yaml_name, 'r')
    content = yaml.load(f)
    print(content)
    print(yaml_name)
    thr = 9000
    dur = thr / 30 / 5
    addr = 'http://203.91.121.212:5000'
    session = requests.Session()
    init(session, addr)
    gpu_monitor(session, addr, index, 'videoedge')
    cmd = 'nvidia-smi dmon -i 0 -d 5 -s u -c {} -f results/gpu_{}.log'.format(int(dur + 6), 0)
    print(cmd)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    cmd = 'nvidia-smi dmon -i 1 -d 5 -s u -c {} -f results/gpu_{}.log'.format(int(dur + 6), 1)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    time.sleep(2)

    for key in content:
        if key is not 'resource':
            config = content[key][0]
            print(config)
            if len(config) is 2:
                r = res[config[0]]
                f = fps[config[1]]
                p = '0'
                proc = Process(target=processing_video_edge_model_in_edge, args=(int(key), r, f, p, thr,))
                proc.start()

            if len(config) is 3:
                r = res[config[0]]
                f = fps[config[1]]
                p = '0'
                cport = 5000 + int(key)
                cip = '203.91.121.212'
                proc1 = Process(target=processing_video, args=(int(key), cip, cport, r, f, p, thr,))
                proc1.start()
                # proc2 = Process(target=processing_video_cloud_model_in_edge, args=(int(key), r, f, p, thr,))
                # proc2.start()
            if len(config) is 7:
                er = res[config[1]]
                ef = fps[config[2]]
                cr = res[config[4]]
                cf = fps[config[5]]
                cp = '0'
                eip = '203.91.121.211'
                p = '0'
                proc1 = Process(target=processing_video_edge_model_in_edge, args=(int(key), er, ef, p, thr,))
                proc1.start()

                cip = '203.91.121.212'
                cport = 5000 + int(key)
                proc2 = Process(target=processing_video, args=(int(key), cip, cport, cr, cf, cp, thr,))
                proc2.start()
                if cp != '0':
                    proc3 = Process(target=processing_video_cloud_model_in_edge, args=(int(key), cr, cf, cp, thr,))
                    proc3.start()

if __name__ == '__main__':

    optimal_name = ['gb_aw_1', 'gb_aw_2', 'gb_aw_3', 'gb_aw_4', 'gb_aw_5',
                    'gb_aw_6', 'gb_aw_7', 'gb_aw_8', 'gb_aw_9',  'gb_aw_10']
    our_name = ['sol_aw_1', 'sol_aw_2', 'sol_aw_3', 'sol_aw_4', 'sol_aw_5',
                'sol_aw_6', 'sol_aw_7', 'sol_aw_8', 'sol_aw_9', 'sol_aw_10']
    videoedge_name = ['videoedge_aw_1', 'videoedge_aw_2', 'videoedge_aw_3', 'videoedge_aw_4', 'videoedge_aw_5',
                      'videoedge_aw_6', 'videoedge_aw_7', 'videoedge_aw_8', 'videoedge_aw_9', 'videoedge_aw_10']
    ourp_name = ['sol_p_aw_1', 'sol_p_aw_2', 'sol_p_aw_3', 'sol_p_aw_4', 'sol_p_aw_5',
                 'sol_p_aw_6', 'sol_p_aw_7', 'sol_p_aw_8', 'sol_p_aw_9', 'sol_p_aw_10']
    aw = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # for i in range(len(optimal_name)):
    '''
    for i in range(len(our_name)):
        yaml_name = '/data/rch/iwqos/Scheduler/aw/{}.yaml'.format(our_name[i])
        print(yaml_name)
        # yaml_name = '/data/rch/iwqos/Scheduler/num/videoedge_num_32.yaml'
        main(yaml_name, i)
        # main_optimal(yaml_name, i)
        time.sleep(1000)
        name = 'results_{}'.format(aw[i])
        # name = 'results_{}'.format(num[i])
        cmd = 'cp -r  results ../result_aw/{}'.format(name)
        os.system(cmd)
        cmd = 'rm results/*'
        os.system(cmd)
        # main_videoedge(yaml_name)
    '''
    for i in range(len(optimal_name)):
        yaml_name = '/data/rch/iwqos/Scheduler/aw/{}.yaml'.format(optimal_name[i])
        print(yaml_name)
        # yaml_name = '/data/rch/iwqos/Scheduler/num/videoedge_num_32.yaml'
        # main(yaml_name, i)
        main_optimal(yaml_name, i)

        name = 'optimal_results_{}'.format(aw[i])
        # name = 'results_{}'.format(num[i])
        cmd = 'cp -r  results ../result_aw/{}'.format(name)
        os.system(cmd)
        cmd = 'rm results/*'
        os.system(cmd)
        # main_videoedge(yaml_name)

    for i in range(len(videoedge_name)):
        yaml_name = '/data/rch/iwqos/Scheduler/aw/{}.yaml'.format(videoedge_name[i])
        print(yaml_name)
        # yaml_name = '/data/rch/iwqos/Scheduler/num/videoedge_num_32.yaml'
        main_videoedge(yaml_name, i)

        name = 'videoedge_results_{}'.format(aw[i])
        # name = 'results_{}'.format(num[i])
        cmd = 'cp -r  results ../result_aw/{}'.format(name)
        os.system(cmd)
        cmd = 'rm results/*'
        os.system(cmd)
        # main_videoedge(yaml_name)

    for i in range(len(ourp_name)):
        yaml_name = '/data/rch/iwqos/Scheduler/aw/{}.yaml'.format(ourp_name[i])
        print(yaml_name)
        # yaml_name = '/data/rch/iwqos/Scheduler/num/videoedge_num_32.yaml'
        main_ourp(yaml_name, i)

        name = 'ourp_results_{}'.format(aw[i])
        # name = 'results_{}'.format(num[i])
        cmd = 'cp -r  results ../result_aw/{}'.format(name)
        os.system(cmd)
        cmd = 'rm results/*'
        os.system(cmd)

