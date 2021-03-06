from flask import Flask, request
from werkzeug.utils import secure_filename
import os
from datetime import datetime
from process import Process
from time import time
import datetime
from Crypto.Cipher import AES
import base64
import models
import fitz.fitz as fitz
app = Flask(__name__)


@app.route('/info', methods=['GET'])
def get():
    global y, m, d
    return 'Đây là phiên bản thương mại của ứng dụng trích xuất thông tin\
    văn bản hành chính\n Xin hãy liên hệ với chúng tôi nếu muốn tiếp tục \
    sử dụng thông qua số điện thoại: 0392200524 hoặc email: 20130704@student.hust.edu.vn\
    Ứng dụng có thời hạn đến ngày {} tháng {} năm {}'.format(d, m, y)


@app.route('/standard', methods=['GET'])
def standard():
    for i in os.listdir('/home/dung/Data/cccd'):
        process('/home/dung/Data/cccd/{}'.format(i))


@app.route('/extract', methods=['POST'])
def extract():
    if request.method == 'POST':
        global count
        count += 1
        if count % 100 == 0 and time() > time2float(datetime.datetime(y, m, d)):
            count = 1
            global pause
            pause = True
            return('Đây là phiên bản thương mại của ứng dụng trích xuất thông tin\
                văn bản hành chính\n Xin hãy liên hệ với chúng tôi nếu muốn tiếp tục \
                sử dụng thông qua số điện thoại: 0392200524 hoặc email: 20130704@student.hust.edu.vn')
        if pause == False:
            try:
                card = request.files['card']
                card_name = 'card.{}.PNG'.format(time())
                card.save(secure_filename(card_name))
                return process.extract(card_name)
            except:
                return 'Error'
            finally:
                if os.path.exists(card_name):
                    os.remove(card_name)
        else:
            return('Đây là phiên bản thương mại của ứng dụng trích xuất thông tin\
                văn bản hành chính\n Xin hãy liên hệ với chúng tôi nếu muốn tiếp tục \
                sử dụng thông qua số điện thoại: 0392200524 hoặc email: 20130704@student.hust.edu.vn')


@app.route('/auth', methods=['POST'])
def auth():
    if request.method == 'POST':
        global count
        count += 1
        if count % 100 == 0 and time() > time2float(datetime.datetime(y, m, d)):
            count = 1
            global pause
            pause = True
            return('Đây là phiên bản thương mại của ứng dụng trích xuất thông tin\
                văn bản hành chính\n Xin hãy liên hệ với chúng tôi nếu muốn tiếp tục \
                sử dụng thông qua số điện thoại: 0392200524 hoặc email: 20130704@student.hust.edu.vn')
        if pause == False:
            try:
                card = request.files['card']
                face = request.files['face']
                t = time()
                card_name = 'card.{}.PNG'.format(t)
                face_name = 'face.{}.PNG'.format(t)
                card.save(secure_filename(card_name))
                face.save(secure_filename(face_name))
                return process.authenticate(card_name, face_name)
            except:
                return 'Error'
            finally:
                if os.path.exists(card_name):
                    os.remove(card_name)
                if os.path.exists(face_name):
                    os.remove(face_name)
        else:
            return('Đây là phiên bản thương mại của ứng dụng trích xuất thông tin\
                văn bản hành chính\n Xin hãy liên hệ với chúng tôi nếu muốn tiếp tục \
                sử dụng thông qua số điện thoại: 0392200524 hoặc email: 20130704@student.hust.edu.vn')


@app.route('/active', methods=['POST'])
def active():
    if request.method == 'POST':
        key = request.get_json()['key']
        secret_key = '123456789012345a'
        cipher = AES.new(secret_key, AES.MODE_ECB)
        decoded = cipher.decrypt(base64.b64decode(key))
        decoded = decoded.decode('utf-8').split(' ')
        file = open('4.txt', 'r', encoding="utf8")
        ls = cipher.decrypt(base64.b64decode(file.read()))
        ls = ls.decode('utf-8').split(' ')
        if ls[0] != decoded[0]:
            return 'Key này không có hiệu lực xin hãy liên hệ với chúng tôi nếu muốn tiếp tục \
                sử dụng thông qua số điện thoại: 0392200524 hoặc email: 20130704@student.hust.edu.vn'
        else:
            file.close()
            os.remove('4.txt')
            file = open('4.txt', 'a', encoding="utf8")
            file.write(key)
            file.close()
            global pause, y, m, d
            pause, y, m, d = False, int(decoded[1]), int(
                decoded[2]), int(decoded[3])
        return 'Licence được gia hạn thành công đến ngày {} tháng {} năm {}'.format(d, m, y)


def time2float(t):
    epoch = datetime.datetime.utcfromtimestamp(0)
    total_seconds = (t - epoch).total_seconds()
    return total_seconds


def initiation():
    file = open('6.txt', 'r', encoding="utf8")
    secret_key = '123456789012345a'
    cipher = AES.new(secret_key, AES.MODE_ECB)
    ls = cipher.decrypt(base64.b64decode(file.read()))
    ls = ls.decode('utf-8').split(' ')
    y, m, d = int(ls[1]), int(ls[2]), int(ls[3])
    if time() < time2float(datetime.datetime(y, m, d)):
        return False, y, m, d, 1
    print('Đây là phiên bản thương mại của ứng dụng trích xuất thông tin\
    văn bản hành chính\n Xin hãy liên hệ với chúng tôi nếu muốn tiếp tục \
        sử dụng thông qua số điện thoại: 0392200524 hoặc email: 20130704@student.hust.edu.vn')
    return True, y, m, d, 1


if __name__ == '__main__':
    pause, y, m, d, count = initiation()
    print('Bạn có muốn cài đặt ứng dụng với cuda ndivia không?(y/n)\
    Hãy chắc chắn máy tính của bạn có cuda nếu muốn sử dụng !')
    while True:
        cuda = input()
        if cuda.lower() == 'y':
            process = Process('cuda')
            print('Ứng dụng được khởi động với cuda')
            break
        elif cuda.lower() == 'n':
            print('Ứng dụng được khởi động với cpu')
            process = Process('cpu')
            break
        else:
            print('Hãy chọn y hoặc n !')
    print('Hãy chọn cổng bạn muốn cài đặt ứng dụng và chắc chắn rằng nó chưa được sử dụng !')
    while True:
        port = input()
        try:
            port = int(port)
            print('Ứng dụng được khởi động trên cổng ', port)
            break
        except Exception:
            print('Cổng cài đặt ứng dụng chưa đúng, xin chọn lại !')
    app.run(host='0.0.0.0', port=port)

