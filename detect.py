import cv2
import torch
import numpy as np
import torchvision
from models.common import Conv


class Detect:
    def __init__(self, device):
        super(Detect, self).__init__()
        self.device = torch.device(device)
        self.model = torch.load('1.pth')
        self.model.to(self.device)
        self.half = False
        if device == 'cuda':
            self.model.half()
            self.half = True
        self.stride = int(self.model.stride.max())
        self.conf_thres, self.iou_thres = 0.25, 0.35

    def scale_coords(self, img1_shape, coords, img0_shape, ratio_pad=None):
        gain = min(img1_shape[0] / img0_shape[0],
                   img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / \
            2, (img1_shape[0] - img0_shape[0] * gain) / 2
        coords[:, [0, 2]] -= pad[0]
        coords[:, [1, 3]] -= pad[1]
        coords[:, :4] /= gain
        coords[:, 0].clamp_(0, img0_shape[1])
        coords[:, 1].clamp_(0, img0_shape[0])
        coords[:, 2].clamp_(0, img0_shape[1])
        coords[:, 3].clamp_(0, img0_shape[0])
        return coords

    def xywh2xyxy(self, x):
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y

    def box_iou(self, box1, box2):
        def box_area(box):
            return (box[2] - box[0]) * (box[3] - box[1])
        area1 = box_area(box1.T)
        area2 = box_area(box2.T)
        inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) -
                 torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
        return inter / (area1[:, None] + area2 - inter)

    def non_max_suppression(self, prediction, conf_thres=0.35, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                            labels=()):
        nc = prediction.shape[2] - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates
        max_wh = 4096  # (pixels) minimum and maximum box width and height
        max_det = 300  # maximum number of detections per image
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        redundant = True  # require redundant detections
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS
        output = [torch.zeros((0, 6), device=prediction.device)
                  ] * prediction.shape[0]
        for xi, x in enumerate(prediction):  # image index, image inference
            x = x[xc[xi]]  # confidence
            if labels and len(labels[xi]):
                l = labels[xi]
                v = torch.zeros((len(l), nc + 5), device=x.device)
                v[:, :4] = l[:, 1:5]  # box
                v[:, 4] = 1.0  # conf
                v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
                x = torch.cat((x, v), 0)
            if not x.shape[0]:
                continue
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
            box = self.xywh2xyxy(x[:, :4])
            if multi_label:
                i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat(
                    (box[i], x[i, j + 5, None], j[:, None].float()), 1)
            else:  # best class only
                conf, j = x[:, 5:].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float()), 1)[
                    conf.view(-1) > conf_thres]
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            elif n > max_nms:  # excess boxes
                # sort by confidence
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            # boxes (offset by class), scores
            boxes, scores = x[:, :4] + c, x[:, 4]
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]
            if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float(
                ) / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy
            output[xi] = x[i]
        return output

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), stride=32):
        shape = img.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
            new_unpad[1]
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
        dw /= 2
        dh /= 2
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img

    def preprocess(self, origin):
        img = self.letterbox(origin)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        return img, origin
    #aaaaaaaaaaaaaaaaaa
    def draw(self, area):
        x1, y1, x2, y2 = int(area[0]), int(
            area[1]), int(area[2]), int(area[3])
        return [x1, y1, x2, y2+1]

    def __call__(self, phase, img_path=None, image=None):
        if phase == 0:
            origin = cv2.imread(img_path)
        else:
            origin = image
        img, origin = self.preprocess(origin)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()
        img /= 255.0
        img = img.unsqueeze(0)
        pred = self.model(img)
        det = self.non_max_suppression(
            pred, self.conf_thres, self.iou_thres)[0]
        if phase == 0:
            cards = []
            if len(det):
                det[:, :4] = self.scale_coords(
                    img.shape[2:], det[:, :4], origin.shape).round()
                for x1, y1, x2, y2, conf, cls in det:
                    if cls in [15, 16, 18, 19, 21, 22]:
                        x1, x2, y1, y2 = int(x1.item()), int(
                            x2.item()), int(y1.item()), int(y2.item())
                        card = origin[y1:y2, x1:x2, :]
                        cards.append(card)
            return cards
        else:
            name1 = None  # ten 0
            number = None  # so cmt 1
            born = None  # ngay sinh 2
            gender = None  # 3
            nation = None  # quoc tich 4
            home1 = None  # que 5
            home2 = None  # que 6
            address1 = None  # cho o hien tai 7
            address2 = None  # cho o hien tai 8
            expiry = None  # ngay het han 9
            characteristic1 = None  # dac diem nhan dang 10
            characteristic2 = None  # dac diem nhan dang 11
            date = None  # ngay cap 12
            ethnic = None  # dan toc 13
            religion = None  # ton giao 14
            before_cmt = None  # chung minh thu truoc 15
            after_cmt = None  # 16
            name2 = None  # 17
            before_cccd = None  # 18
            after_cccd = None  # 19
            encode = None  # 20
            before_chip = None  # 21
            after_chip = None  # 22
            public_place = None  # 23 noi cap
            face = None  # 24 mat

            det[:, :4] = self.scale_coords(
                img.shape[2:], det[:, :4], origin.shape).round()
            for x1, y1, x2, y2, conf, cls in det:
                if cls == 1:
                    if name1 == None or name1[4] < conf:
                        name1 = [x1, y1, x2, y2, conf, 'name1']
                elif cls == 0:
                    if number == None or number[4] < conf:
                        number = [x1, y1, x2, y2, conf, 'number']
                elif cls == 2:
                    if born == None or born[4] < conf:
                        born = [x1, y1, x2, y2, conf, 'born']
                elif cls == 3:
                    if gender == None or gender[4] < conf:
                        gender = [x1, y1, x2, y2, conf, 'gender']
                elif cls == 4:
                    if nation == None or nation[4] < conf:
                        nation = [x1, y1, x2, y2, conf, 'nation']
                elif cls == 5:
                    if home1 == None or home1[4] < conf:
                        home1 = [x1, y1, x2, y2, conf, 'home1']
                elif cls == 6:
                    if home2 == None or home2[4] < conf:
                        home2 = [x1, y1, x2, y2, conf, 'home2']
                elif cls == 7:
                    if address1 == None or address1[4] < conf:
                        address1 = [x1, y1, x2, y2, conf, 'address1']
                elif cls == 8:
                    if address2 == None or address2[4] < conf:
                        address2 = [x1, y1, x2, y2, conf, 'address2']
                elif cls == 9:
                    if expiry == None or expiry[4] < conf:
                        expiry = [x1, y1, x2, y2, conf, 'expiry']
                elif cls == 10:
                    if characteristic1 == None or characteristic1[4] < conf:
                        characteristic1 = [x1, y1, x2,
                                           y2, conf, 'characteristic1']
                elif cls == 11:
                    if characteristic2 == None or characteristic2[4] < conf:
                        characteristic2 = [x1, y1, x2,
                                           y2, conf, 'characteristic2']
                elif cls == 12:
                    if date == None or date[4] < conf:
                        date = [x1, y1, x2, y2, conf, 'date']
                elif cls == 13:
                    if ethnic == None or ethnic[4] < conf:
                        ethnic = [x1, y1, x2, y2, conf, 'ethnic']
                elif cls == 14:
                    if religion == None or religion[4] < conf:
                        religion = [x1, y1, x2, y2, conf, 'religion']
                elif cls == 15:
                    if before_cmt == None or before_cmt[4] < conf:
                        before_cmt = [x1, y1, x2, y2, conf, 'before_cmt']
                elif cls == 16:
                    if after_cmt == None or after_cmt[4] < conf:
                        after_cmt = [x1, y1, x2, y2, conf, 'after_cmt']
                elif cls == 17:
                    if name2 == None or name2[4] < conf:
                        name2 = [x1, y1, x2, y2, conf, 'name2']
                elif cls == 18:
                    if before_cccd == None or religion[4] < conf:
                        before_cccd = [x1, y1, x2, y2, conf, 'before_cccd']
                elif cls == 19:
                    if after_cccd == None or after_cccd[4] < conf:
                        after_cccd = [x1, y1, x2, y2, conf, 'after_cccd']
                elif cls == 20:
                    if encode == None or encode[4] < conf:
                        encode = [x1, y1, x2, y2, conf, 'encode']
                elif cls == 21:
                    if before_chip == None or before_chip[4] < conf:
                        before_chip = [x1, y1, x2, y2, conf, 'before_chip']
                elif cls == 22:
                    if after_chip == None or after_chip[4] < conf:
                        after_chip = [x1, y1, x2, y2, conf, 'after_chip']
                elif cls == 23:
                    if public_place == None or public_place[4] < conf:
                        public_place = [x1, y1, x2, y2, conf, 'public_place']
                elif cls == 24:
                    if face == None or face[4] < conf:
                        face = [x1, y1, x2, y2, conf, 'face']
            if name1 != None:
                name1 = self.draw(name1)
                name1.append(origin[name1[1]:name1[3], name1[0]:name1[2], :])
                name1.append('name1')
            if number != None:
                number = self.draw(number)
                number.append(
                    origin[number[1]:number[3], number[0]:number[2], :])
                number.append('number')
            if born != None:
                born = self.draw(born)
                born.append(
                    origin[born[1]:born[3], born[0]:born[2], :])
                born.append('born')
            if gender != None:
                gender = self.draw(gender)
                gender.append(
                    origin[gender[1]:gender[3], gender[0]:gender[2], :])
                gender.append('gender')
            if nation != None:
                nation = self.draw(nation)
                nation.append(
                    origin[nation[1]:nation[3], nation[0]:nation[2], :])
                nation.append('nation')
            if home1 != None:
                home1 = self.draw(home1)
                home1.append(
                    origin[home1[1]:home1[3], home1[0]:home1[2], :])
                home1.append('home1')
            if home2 != None:
                home2 = self.draw(home2)
                home2.append(
                    origin[home2[1]:home2[3], home2[0]:home2[2], :])
                home2.append('home2')
            if address1 != None:
                address1 = self.draw(address1)
                address1.append(
                    origin[address1[1]:address1[3], address1[0]:address1[2], :])
                address1.append('address1')
            if address2 != None:
                address2 = self.draw(address2)
                address2.append(
                    origin[address2[1]:address2[3], address2[0]:address2[2], :])
                address2.append('address2')
            if expiry != None:
                expiry = self.draw(expiry)
                expiry.append(
                    origin[expiry[1]:expiry[3], expiry[0]:expiry[2], :])
                expiry.append('expiry')
            if characteristic1 != None:
                characteristic1 = self.draw(characteristic1)
                characteristic1.append(
                    origin[characteristic1[1]:characteristic1[3], characteristic1[0]:characteristic1[2], :])
                characteristic1.append('characteristic1')
            if characteristic2 != None:
                characteristic2 = self.draw(characteristic2)
                characteristic2.append(
                    origin[characteristic2[1]:characteristic2[3], characteristic2[0]:characteristic2[2], :])
                characteristic2.append('characteristic2')
            if date != None:
                date = self.draw(date)
                date.append(
                    origin[date[1]:date[3], date[0]:date[2], :])
                date.append('date')
            if ethnic != None:
                ethnic = self.draw(ethnic)
                ethnic.append(
                    origin[ethnic[1]:ethnic[3], ethnic[0]:ethnic[2], :])
                ethnic.append('ethnic')
            if religion != None:
                religion = self.draw(religion)
                religion.append(
                    origin[religion[1]:religion[3], religion[0]:religion[2], :])
                religion.append('religion')
            if before_cmt != None:
                before_cmt = self.draw(before_cmt)
                before_cmt.append(
                    origin[before_cmt[1]:before_cmt[3], before_cmt[0]:before_cmt[2], :])
                before_cmt.append('before_cmt')
            if after_cmt != None:
                after_cmt = self.draw(after_cmt)
                after_cmt.append(
                    origin[after_cmt[1]:after_cmt[3], after_cmt[0]:after_cmt[2], :])
                after_cmt.append('after_cmt')
            if name2 != None:
                name2 = self.draw(name2)
                name2.append(
                    origin[name2[1]:name2[3], name2[0]:name2[2], :])
                name2.append('name2')
            if before_cccd != None:
                before_cccd = self.draw(before_cccd)
                before_cccd.append(
                    origin[before_cccd[1]:before_cccd[3], before_cccd[0]:before_cccd[2], :])
                before_cccd.append('before_cccd')
            if after_cccd != None:
                after_cccd = self.draw(after_cccd)
                after_cccd.append(
                    origin[after_cccd[1]:after_cccd[3], after_cccd[0]:after_cccd[2], :])
                after_cccd.append('after_cccd')
            if encode != None:
                encode = self.draw(encode)
                encode.append(
                    origin[encode[1]:encode[3], encode[0]:encode[2], :])
                encode.append('encode')
            if before_chip != None:
                before_chip = self.draw(before_chip)
                before_chip.append(
                    origin[before_chip[1]:before_chip[3], before_chip[0]:before_chip[2], :])
                before_chip.append('before_chip')
            if after_chip != None:
                after_chip = self.draw(after_chip)
                after_chip.append(
                    origin[after_chip[1]:after_chip[3], after_chip[0]:after_chip[2], :])
                after_chip.append('after_chip')
            if public_place != None:
                public_place = self.draw(public_place)
                public_place.append(
                    origin[public_place[1]:public_place[3], public_place[0]:public_place[2], :])
                public_place.append('public_place')
            if face != None:
                face = self.draw(face)
                face = cv2.resize(
                    origin[face[1]:face[3], face[0]:face[2], :], (112, 112))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = torch.tensor(face, dtype=torch.float32).permute(
                    2, 0, 1).unsqueeze(0)
                face.div_(255).sub_(0.5).div_(0.5)  # (img/255 - 0.5)/0.5

        return [
            name1,
            number,
            born,
            gender,
            nation,
            home1,
            home2,
            address1,
            address2,
            expiry,
            characteristic1,
            characteristic2,
            date,
            ethnic,
            religion,
            before_cmt,
            after_cmt,
            name2,
            before_cccd,
            after_cccd,
            encode,
            before_chip,
            after_chip,
            public_place
        ], face

