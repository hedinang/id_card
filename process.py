from detect import Detect
from ocr import Ocr
from unet import Rotation
from face import Face
from Levenshtein import distance
import cv2


class Process:
    def __init__(self, device):
        super(Process, self).__init__()
        self.detect = Detect(device)
        self.ocr = Ocr(device)
        self.rotation = Rotation(device)
        self.face = Face(device)
        self.born_character = ['0', '1', '2',
                               '3', '4', '5', '6', '7', '8', '9']
        self.gender = ['nam', 'nữ']
        self.religion = ['phật giáo', 'công giáo', 'tín lành',
                         'cao đài', 'hòa hảo', 'ấn độ giáo', 'hồi giáo', 'không']
        self.ethnic = ['kinh', 'tày', 'thái', 'mường', 'khmer', 'hoa', 'nùng', "h'mông", 'dao', 'gia rai', 'ê đê',
                       'ba na', 'sán chay', 'chăm', 'cơ ho', 'xơ đăng', 'sán dìu', 'hrê', 'ra glai', 'mnông', 'thổ',
                       'stiêng', 'khơ mú', 'bru - vân kiều', 'cơ tu', 'giáy', 'tà ôi', 'mạ', 'giẻ-triêng', 'co',
                       'chơ ro', 'xinh mun', 'hà nhì', 'chu ru', 'lào', 'la chí', 'kháng', 'phù lá', 'la hủ', 'la ha',
                       'pà thẻn', 'lự', 'ngái', 'chứt', 'lô lô', 'mảng', 'cơ lao', 'bố y', 'cống', 'si la', 'pu péo',
                       'rơ măm', 'brâu', 'ơ đu']
        self.public_place = ['TP Cần Thơ', 'TP Đà Nẵng', 'TP Hải Phòng', 'TP Hà Nội', 'TP HCM',
                             'Tỉnh An Giang', 'Tỉnh Bà Rịa - Vũng Tàu', 'Tỉnh Bắc Giang', 'Tỉnh Bắc Kạn', 'Tỉnh Bạc Liêu',
                             'Tỉnh Bắc Ninh', 'Tỉnh Bến Tre', 'Tỉnh Bình Định', 'Tỉnh Bình Dương', 'Tỉnh Bình Phước', 'Tỉnh Bình Thuận',
                             'Tỉnh Cà Mau', 'Tỉnh Cao Bằng', 'Tỉnh Đắk Lắk', 'Tỉnh Đắk Nông', 'Tỉnh Điện Biên', 'Tỉnh Đồng Nai',
                             'Tỉnh Đồng Tháp', 'Tỉnh Gia Lai', 'Tỉnh Hà Giang'	, 'Tỉnh Hà Nam', 'Tỉnh Hà Tĩnh', 'Tỉnh Hải Dương',
                             'Tỉnh Hậu Giang', 'Tỉnh Hòa Bình', 'Tỉnh Hưng Yên', 'Tỉnh Khánh Hòa', 'Tỉnh Kiên Giang', 'Tỉnh Kon Tum',
                             'Tỉnh Lai Châu', 'Tỉnh Lâm Đồng', 'Tỉnh Lạng Sơn', 'Tỉnh Lào Cai', 'Tỉnh Long An', 'Tỉnh Nam Định',
                             'Tỉnh Nghệ An', 'Tỉnh Ninh Bình', 'Tỉnh Ninh Thuận', 'Tỉnh Phú Thọ', 'Tỉnh Quảng Bình', 'Tỉnh Quảng Nam',
                             'Tỉnh Quảng Ngãi', 'Tỉnh Quảng Ninh', 'Tỉnh Quảng Trị', 'Tỉnh Sóc Trăng', 'Tỉnh Sơn La', 'Tỉnh Tây Ninh',
                             'Tỉnh Thái Bình', 'Tỉnh Thái Nguyên', 'Tỉnh Thanh Hóa', 'Tỉnh Thừa Thiên Huế', 'Tỉnh Tiền Giang',
                             'Tỉnh Trà Vinh', 'Tỉnh Tuyên Quang', 'Tỉnh Vĩnh Long', 'Tỉnh Vĩnh Phúc', 'Tỉnh Yên Bái', 'Tỉnh Phú Yên']

        self.province = open(
            '/home/dung/Data/Address/province.txt').read().split('\n')

    def search(self, s, array):
        min = 100
        result = array[0]
        for e in array:
            dis = distance(s, e)
            if dis < min:
                min = dis
                result = e
        return result

    def extract_address(self, address):
        max_distance = 100
        txt_province = ''
        for p in self.province:
            compute = distance(address, p)
            if compute < max_distance:
                max_distance = compute
                txt_province = p
        txt_commune = ''
        if txt_province == '':
            return [None, None, None]
        if max_distance < 25:
            commune = open(
                '/home/dung/Data/Address/commune/{}.txt'.format(txt_province)).read().split('\n')
            max_distance = 100
            for c in commune:
                compute = distance(address, c)
                if compute < max_distance:
                    max_distance = compute
                    txt_commune = c
            if max_distance < 10:
                return txt_commune.split(', ')
            else:
                txt_province = txt_province.split(', ')
                txt_province = [None]+txt_province
                return txt_province
        else:
            return [None, None, None]

    def extract(self, img_path):
        result_card = {
            'result': []
        }
        cards = self.detect(0, img_path)
        for card in cards:
            card = self.rotation(card, card)
            queue = self.detect(1, None, card)[0]
            results = {
                'name': ['', ''],  # ten
                'number': '',  # so cmt
                'born': '',  # ngay sinh
                'gender': '',
                'nation': '',  # quoc tich
                'home': ['', ''],  # que
                'address': ['', ''],  # cho o hien tai
                'expiry': '',  # ngay het han
                'characteristic': ['', ''],  # dac diem nhan dang
                'date': '',
                'ethnic': '',  # dan toc
                'religion': '',  # ton giao
                'address_province': '',
                'address_district': '',
                'address_commune': '',
                'home_province': '',
                'home_district': '',
                'home_commune': '',
                'public_place': ''
            }
            for e in queue:
                if e != None:
                    sub_ocr = self.ocr(e[4], e[5])
                    if e[5] == 'gender':
                        results[e[5]] = self.search(
                            sub_ocr.lower(), self.gender)
                    elif e[5] == 'ethnic':
                        results[e[5]] = self.search(
                            sub_ocr.lower(), self.ethnic)
                    elif e[5] == 'religion':
                        results[e[5]] = self.search(
                            sub_ocr.lower(), self.religion)
                    elif e[5] == 'nation':
                        results['nation'] = 'Việt Nam'
                    elif e[5] == 'public_place':
                        results['public_place'] = self.search(
                            sub_ocr, self.public_place)
                    elif e[5] == 'number':
                        for c in sub_ocr:
                            if c.isnumeric() is False:
                                sub_ocr = sub_ocr.replace(c, '')
                        results['number'] = sub_ocr
                    elif e[5] in ['born', 'date']:
                        date_result = ['a']
                        i = 1
                        prev_sub_ocr = ''
                        for c in sub_ocr:
                            if c in self.born_character:
                                i += 1
                                date_result.append(c)
                            else:
                                if date_result[i-1].isnumeric():
                                    if prev_sub_ocr.isnumeric() is False:
                                        continue
                                    date_result.append('/')
                                    i += 1
                            prev_sub_ocr = c
                        s = ''
                        s = s.join(date_result[1:])
                        results[e[5]] = s
                    elif e[5] == 'name1':
                        results['name'][0] = sub_ocr
                    elif e[5] == 'name2':
                        results['name'][1] = sub_ocr
                    elif e[5] == 'home1':
                        results['home'][0] = sub_ocr
                    elif e[5] == 'home2':
                        results['home'][1] = sub_ocr
                    elif e[5] == 'address1':
                        results['address'][0] = sub_ocr
                    elif e[5] == 'address2':
                        results['address'][1] = sub_ocr
                    elif e[5] == 'characteristic1':
                        results['characteristic'][0] = sub_ocr
                    elif e[5] == 'characteristic2':
                        results['characteristic'][1] = sub_ocr
            results['name'] = '{} {}'.format(
                results['name'][0], results['name'][1]).strip()
            results['home'] = '{} {}'.format(
                results['home'][0], results['home'][1]).strip()
            results['address'] = '{} {}'.format(
                results['address'][0], results['address'][1]).strip()
            results['characteristic'] = '{} {}'.format(
                results['characteristic'][0], results['characteristic'][1]).strip()
            results['home_commune'], results['home_district'], results['home_province'] = self.extract_address(
                results['home'])
            results['address_commune'], results['address_district'], results['address_province'] = self.extract_address(
                results['address'])
            result_card['result'].append(results)

        return result_card

    def authenticate(self, card_name, face_name):
        result_card = {
            'result': []
        }
        cards = self.detect(0, card_name)
        for card in cards:
            card = self.rotation(card, card)
            queue, face_in_card = self.detect(1, None, card)
            face = cv2.imread(face_name)
            face = self.detect(1, None, face)[1]
            results = {
                'face': False,
                'name': ['', ''],  # ten
                'number': '',  # so cmt
                'born': '',  # ngay sinh
                'gender': '',
                'nation': '',  # quoc tich
                'home': ['', ''],  # que
                'address': ['', ''],  # cho o hien tai
                'expiry': '',  # ngay het han
                'characteristic': ['', ''],  # dac diem nhan dang
                'date': '',
                'ethnic': '',  # dan toc
                'religion': '',  # ton giao
                'address_province': '',
                'address_district': '',
                'address_commune': '',
                'home_province': '',
                'home_district': '',
                'home_commune': ''
            }
            # check xem co ton tai face ko
            if face_in_card is None or face is None:
                result_card['result'].append(results)
                return result_card
            # check xem mat co khop khong
            if self.face(face_in_card, face) is False:
                result_card['result'].append(results)
                return result_card
            for e in queue:
                if e != None:
                    sub_ocr = self.ocr(e[4], e[5])
                    if e[5] == 'gender':
                        results[e[5]] = self.search(
                            sub_ocr.lower(), self.gender)
                    elif e[5] == 'ethnic':
                        results[e[5]] = self.search(
                            sub_ocr.lower(), self.ethnic)
                    elif e[5] == 'religion':
                        results[e[5]] = self.search(
                            sub_ocr.lower(), self.religion)
                    elif e[5] == 'nation':
                        results['nation'] = 'Việt Nam'
                    elif e[5] == 'number':
                        for c in sub_ocr:
                            if c.isnumeric() is False:
                                sub_ocr = sub_ocr.replace(c, '')
                        results['number'] = sub_ocr
                    elif e[5] in ['born', 'date']:
                        date_result = ['a']
                        i = 1
                        prev_sub_ocr = ''
                        for c in sub_ocr:
                            if c in self.born_character:
                                i += 1
                                date_result.append(c)
                            else:
                                if date_result[i-1].isnumeric():
                                    if prev_sub_ocr.isnumeric() is False:
                                        continue
                                    date_result.append('/')
                            prev_sub_ocr = c
                        s = ''
                        s = s.join(date_result[1:])
                        results[e[5]] = s
                    elif e[5] == 'name1':
                        results['name'][0] = sub_ocr
                    elif e[5] == 'name2':
                        results['name'][1] = sub_ocr
                    elif e[5] == 'home1':
                        results['home'][0] = sub_ocr
                    elif e[5] == 'home2':
                        results['home'][1] = sub_ocr
                    elif e[5] == 'address1':
                        results['address'][0] = sub_ocr
                    elif e[5] == 'address2':
                        results['address'][1] = sub_ocr
                    elif e[5] == 'characteristic1':
                        results['characteristic'][0] = sub_ocr
                    elif e[5] == 'characteristic2':
                        results['characteristic'][1] = sub_ocr
            results['face'] = True
            results['name'] = '{} {}'.format(
                results['name'][0], results['name'][1]).strip()
            results['home'] = '{} {}'.format(
                results['home'][0], results['home'][1]).strip()
            results['address'] = '{} {}'.format(
                results['address'][0], results['address'][1]).strip()
            results['characteristic'] = '{} {}'.format(
                results['characteristic'][0], results['characteristic'][1]).strip()
            results['home_commune'], results['home_district'], results['home_province'] = self.extract_address(
                results['home'])
            results['address_commune'], results['address_district'], results['address_province'] = self.extract_address(
                results['address'])
            result_card['result'].append(results)

        return result_card

