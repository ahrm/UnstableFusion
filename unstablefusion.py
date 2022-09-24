from multiprocessing import dummy
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys
import numpy as np
from PIL import Image
from diffusionserver import StableDiffusionHandler
import cv2
import pathlib
import time
import json
import os
import datetime
import requests
from base64 import decodebytes
import traceback
import random
from dataclasses import dataclass

SIZE_INCREASE_INCREMENT = 20

def cv2_telea(img, mask):
    ret = cv2.inpaint(img, 255 - mask, 5, cv2.INPAINT_TELEA)
    return ret, mask


def cv2_ns(img, mask):
    ret = cv2.inpaint(img, 255 - mask, 5, cv2.INPAINT_NS)
    return ret, mask

def get_quicksave_path():
    parent_path = pathlib.Path(__file__).parents[0]
    folder_path = parent_path / 'quicksaves'
    folder_path.mkdir(exist_ok=True, parents=True)
    return folder_path

def get_modifiers_path():
    parent_path = pathlib.Path(__file__).parents[0]
    return str(parent_path / 'modifiers.txt')

def save_modifiers(mods):
    with open(get_modifiers_path(), 'w') as outfile:
        outfile.write(mods)

def load_modifiers():
    with open(get_modifiers_path(), 'r') as infile:
        return infile.read()

def get_unique_filename():
    return str(int(time.time())) + ".png"

def get_most_recent_saved_file():
    parent_path = pathlib.Path(__file__).parents[0]
    folder_path = parent_path / 'quicksaves'

    most_recent = datetime.datetime.min
    most_recent_path = None

    for file in os.listdir(folder_path):
        path = folder_path / file
        # get date
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(path))
        if mtime > most_recent:
            most_recent = mtime
            most_recent_path = path
    return most_recent_path
        

def quicksave_image(np_image, file_path=None):
    if file_path == None:
        file_path = get_quicksave_path() / (get_unique_filename())

    Image.fromarray(np_image).save(file_path)
    return file_path

def gaussian_noise(img, mask):
    noise = np.random.randn(mask.shape[0], mask.shape[1], 3)
    noise = (noise + 1) / 2 * 255
    noise = noise.astype(np.uint8)
    nmask = mask.copy()
    nmask[mask > 0] = 1
    img = nmask[:, :, np.newaxis] * img + (1 - nmask[:, :, np.newaxis]) * noise
    return img, mask

inpaint_options = ['cv2_ns',
         'cv2_telea',
         'gaussian']
        
inpaint_functions = {
    'cv2_ns': cv2_ns,
    'cv2_telea': cv2_telea,
    'gaussian': gaussian_noise
}

shortcuts_ = {
    'undo': 'Ctrl+Z',
    'redo': 'Ctrl+Shift+Z',
    'open': 'O',
    'toggle_scratchpad': 'S',
    'quicksave': 'F5',
    'quickload': 'F9',
    'select_color': 'C',
    'generate': 'Return',
    'inpaint': 'Space',
    'reimagine': 'R',
    'export': 'Ctrl+S',
    'increase_size': '+',
    'decrease_size': '-',
    'paste_from_scratchpad': 'p',
    'toggle_preview': 'T',
}

def get_shortcut_dict():
    parent_path = pathlib.Path(__file__).parents[0]
    file_path = parent_path / 'keys.json'

    if os.path.exists(file_path):
        with open(file_path) as f:
            json_content = json.load(f)
        for key, value in json_content.items():
            shortcuts_[key] = value
    return shortcuts_

def get_texture():
    SIZE = 512
    Z = np.zeros((SIZE, SIZE), dtype=np.uint8)
    return np.stack([Z, Z, Z, Z], axis=2)


testtexture = get_texture()

def qimage_from_array(arr):
    maximum = arr.max()
    if arr.shape[-1] != 4:
        arr = np.concatenate([arr, np.ones((arr.shape[0], arr.shape[1], 1)) * 255], axis=2)

    if maximum > 0 and maximum <= 1:
        return  QImage((arr.astype('uint8') * 255).data, arr.shape[1], arr.shape[0], QImage.Format_RGBA8888)
    else:
        return  QImage(arr.astype('uint8').data, arr.shape[1], arr.shape[0], QImage.Format_RGBA8888)

testimage = qimage_from_array(testtexture)

class DummyStableDiffusionHandler:

    def __init__(self):
        pass

    def inpaint(self, prompt, image, mask, strength=0.75, steps=50, guidance_scale=7.5, seed=-1):
        inpainted_image = np.zeros_like(image)
        for i in range(inpainted_image.shape[1]):
            for j in range(inpainted_image.shape[0]):
                inpainted_image[i, j, 0] = 255 * i // inpainted_image.shape[1]
                inpainted_image[i, j, 1] = 255 * j // inpainted_image.shape[1]

        new_image = image.copy()[:, :, :3]
        # new_image[mask > 0] = np.array([255, 0, 0])
        new_image[mask > 0] = inpainted_image[mask > 0]
        return Image.fromarray(new_image)

    def generate(self, prompt, width=512, height=512, strength=0.75, steps=50, guidance_scale=7.5, seed=-1):

        np_im = np.zeros((height, width, 3), dtype=np.uint8)
        np_im[:, :, 2] = 255
        for i in range(np_im.shape[0]):
            np_im[i, :, 1] = int((i / np_im.shape[0]) * 255)
        for j in range(np_im.shape[1]):
            np_im[:, j, 2] = int((j / np_im.shape[0]) * 255)
        return Image.fromarray(np_im)

    def reimagine(self, prompt, image, steps=50, guidance_scale=7.5, seed=-1):
        return image


class ServerStableDiffusionHandler:

    def __init__(self, server_address):
        self.addr = server_address
        if self.addr[-1] != '/':
            self.addr += '/'
    
    def inpaint(self, prompt, image, mask, strength=0.75, steps=50, guidance_scale=7.5, seed=-1, callback=None):
        request_data = {
            'prompt': prompt,
            'strength': strength,
            'steps': steps,
            'guidance_scale': guidance_scale,
            'seed': seed,
            'image': image.tolist(),
            'mask': mask.tolist()
        }
        url = self.addr + 'inpaint'

        resp = requests.post(url, json=request_data)

        resp_data = resp.json()
        size = resp_data['image_size']
        mode = resp_data['image_mode']
        image_data = decodebytes(bytes(resp_data['image_data'], encoding='ascii'))

        return Image.frombytes(mode, size, image_data)
    
    def generate(self, prompt, width=512, height=512, strength=0.75, steps=50, guidance_scale=7.5,seed=-1, callback=None):
            request_data = {
                'prompt': prompt,
                'strength': strength,
                'steps': steps,
                'guidance_scale': guidance_scale,
                'seed': seed,
                'width': width,
                'height': height
            }
            url = self.addr + 'generate'

            resp = requests.post(url, json=request_data)

            resp_data = resp.json()
            size = resp_data['image_size']
            mode = resp_data['image_mode']
            image_data = decodebytes(bytes(resp_data['image_data'], encoding='ascii'))

            return Image.frombytes(mode, size, image_data)
    
    def reimagine(self, prompt, image, steps=50, guidance_scale=7.5, seed=-1, strength=7.5, callback=None):
        request_data = {
            'prompt': prompt,
            'steps': steps,
            'guidance_scale': guidance_scale,
            'seed': seed,
            'strength': strength,
            'image': image.tolist(),
        }
        url = self.addr + 'reimagine'

        resp = requests.post(url, json=request_data)

        resp_data = resp.json()
        size = resp_data['image_size']
        mode = resp_data['image_mode']
        image_data = decodebytes(bytes(resp_data['image_data'], encoding='ascii'))

        return Image.frombytes(mode, size, image_data)

class StableDiffusionManager:

    def __init__(self):
        self.cached_local_handler = None
        self.mode_widget = None
        self.huggingface_token_widget = None
        self.server_address_widget = None
    
    def disable_safety(self):
        if self.cached_local_handler != None:
            self.cached_local_handler.img2img.safety_checker = dummy_safety_checker
            self.cached_local_handler.text2img.safety_checker = dummy_safety_checker
            self.cached_local_handler.inpainter.safety_checker = dummy_safety_checker

    def get_local_handler(self, token=True):
        if self.cached_local_handler == None:
            self.cached_local_handler = StableDiffusionHandler(token)

        return self.cached_local_handler
    
    def get_server_handler(self):
        addr = self.server_address_widget.text()
        return ServerStableDiffusionHandler(addr)
    
    def get_huggingface_token(self):
        if len(self.huggingface_token_widget.text()) == 0:
            return True
        else:
            return self.huggingface_token_widget.text()

    def get_handler(self):
        if self.mode_widget.currentText() == 'local':
            return self.get_local_handler(self.get_huggingface_token())
        else:
            return self.get_server_handler()

@dataclass
class SavedMaskState:
    mask: np.ndarray
    box: QRect
class PaintWidget(QWidget):

    def __init__(self, prompt_textarea_, modifiers_textarea_, stable_diffusion_manager_, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.np_image = None
        self.qt_image = None
        self.selection_rectangle = None
        self.selection_rectangle_size = (100, 100)
        self.is_dragging = False
        self.image_rect = None

        self.strength = 0.75
        self.steps = 50
        self.guidance_scale = 7.5
        self.seed = -1

        self.should_preview_scratchpad = False

        self.inpaint_method = inpaint_options[0]

        self.history = []
        self.future = []
        self.color = np.array([0, 0, 0])

        self.setAcceptDrops(True)
        self.scratchpad = None
        self.owner = None

        self.should_limit_box_size = True
        self.shoulds_swap_buttons = False

        self.saved_mask_state = None

        shortcuts = get_shortcut_dict()
        self.paste_shortcut = QShortcut(QKeySequence(shortcuts['paste_from_scratchpad']), self)
        self.paste_shortcut.activated.connect(self.handle_paste_scratchpad)

        self.decrease_size_shortcut = QShortcut(QKeySequence(shortcuts['decrease_size']), self)
        self.decrease_size_shortcut.activated.connect(self.handle_decrease_size_button)

        self.increase_size_shortcut = QShortcut(QKeySequence(shortcuts['increase_size']), self)
        self.increase_size_shortcut.activated.connect(self.handle_increase_size_button)

        self.export_shortcut = QShortcut(QKeySequence(shortcuts['export']), self)
        self.export_shortcut.activated.connect(self.handle_export_button)

        self.reimagine_shortcut = QShortcut(QKeySequence(shortcuts['reimagine']), self)
        self.reimagine_shortcut.activated.connect(self.handle_reimagine_button)

        self.inpaint_shortcut = QShortcut(QKeySequence(shortcuts['inpaint']), self)
        self.inpaint_shortcut.activated.connect(self.handle_inpaint_button)

        self.generate_shortcut = QShortcut(QKeySequence(shortcuts['generate']), self)
        self.generate_shortcut.activated.connect(self.handle_generate_button)

        self.select_color_shortcut = QShortcut(QKeySequence(shortcuts['select_color']), self)
        self.select_color_shortcut.activated.connect(self.handle_select_color_button)

        self.undo_shortcut = QShortcut(QKeySequence(shortcuts['undo']), self)
        self.redo_shortcut = QShortcut(QKeySequence(shortcuts['redo']), self)
        self.undo_shortcut.activated.connect(self.update_and(self.undo))
        self.redo_shortcut.activated.connect(self.update_and(self.redo))

        self.open_shortcut = QShortcut(QKeySequence(shortcuts['open']), self)
        self.open_shortcut.activated.connect(self.update_and(self.handle_load_image_button))

        self.toggle_scratchpad_shortcut = QShortcut(QKeySequence(shortcuts['toggle_scratchpad']), self)
        self.toggle_scratchpad_shortcut.activated.connect(self.update_and(self.handle_show_scratchpad))

        self.quicksave_shortcut = QShortcut(QKeySequence(shortcuts['quicksave']), self)
        self.quicksave_shortcut.activated.connect(self.update_and(self.handle_quicksave_button))

        self.quickload_shortcut = QShortcut(QKeySequence(shortcuts['quickload']), self)
        self.quickload_shortcut.activated.connect(self.update_and(self.handle_quickload_button))

        self.toggle_preview_shortcut = QShortcut(QKeySequence(shortcuts['toggle_preview']), self)
        self.toggle_preview_shortcut.activated.connect(self.update_and(self.toggle_should_preview_scratchpad))
        
        self.prompt_textarea = prompt_textarea_
        self.modifiers_textarea = modifiers_textarea_
        self.stable_diffusion_manager = stable_diffusion_manager_
        self.preview_image = None
    
    def set_preview_image(self, preview_image):
        self.preview_image = np.array(Image.fromarray(preview_image).resize((self.selection_rectangle.width(), self.selection_rectangle.height()), Image.NEAREST))

    def set_should_preview_scratchpad(self, val):
        self.should_preview_scratchpad = val

    def toggle_should_preview_scratchpad(self):
        self.set_should_preview_scratchpad(not self.should_preview_scratchpad)

    def set_should_swap_buttons(self, val):
        self.shoulds_swap_buttons = val

    def get_handler(self):
        return self.stable_diffusion_manager.get_handler()
    
    def reset_saved_mask(self):
        self.saved_mask_state = None

    def save_mask(self):
        mask = self.get_selection_np_image()[:, :, 3]
        box = self.selection_rectangle
        self.saved_mask_state = SavedMaskState(mask, box)

    def update_and(self, f):
        def update_and_f(*args, **kwargs):
            f(*args, **kwargs)
            self.update()
        return update_and_f

    def dragEnterEvent(self, e):
        if e.mimeData().hasImage:
            e.accept()
        else:
            e.ignore()
        
    def dropEvent(self, e):
        imdata = Image.open(e.mimeData().text()[8:])
        image_numpy = np.array(imdata)
        self.set_np_image(image_numpy)
        self.resize_to_image(only_if_smaller=True)
        self.update()
    
    def set_strength(self, new_strength):
        self.strength = new_strength

    def set_steps(self, new_steps):
        self.steps = new_steps

    def set_guidance_scale(self, new_guidance_scale):
        self.guidance_scale = new_guidance_scale
    
    def set_inpaint_method(self, method):
        self.inpaint_method = method
    
    def set_color(self, new_color):
        self.color = np.array([new_color.red(), new_color.green(), new_color.blue()])

    def undo(self):
        if len(self.history) > 0:
            self.future = [self.np_image.copy()] + self.future
            self.set_np_image(self.history[-1], add_to_history=False)
            self.history = self.history[:-1]

    def redo(self):
        if len(self.future) > 0:
            prev_image = self.np_image.copy()
            self.set_np_image(self.future[0], add_to_history=False)
            self.future = self.future[1:]
            self.history.append(prev_image)

    def set_np_image(self, arr, add_to_history=True):
        if arr.shape[-1] == 3:
            arr = np.concatenate([arr, np.ones(arr.shape[:2] + (1,)) * 255], axis=-1)

        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)

        if add_to_history == True:
            self.future = []

        if add_to_history and (not (self.np_image is None)):
            self.history.append(self.np_image.copy())

        self.np_image = arr
        self.qt_image = qimage_from_array(self.np_image)

    def mouseMoveEvent(self, e):
        right_button = self.get_mouse_button(Qt.RightButton)
        mid_button = self.get_mouse_button(Qt.MidButton)

        if self.is_dragging:
            self.selection_rectangle.moveCenter(e.pos())

            if e.buttons() & right_button:
                self.erase_selection(False)
            if e.buttons() & mid_button:
                self.paint_selection(False)

            self.update()
            if self.owner != None:
                self.owner.update()

    def mouseReleaseEvent(self, e):
        # if e.button() == Qt.LeftButton:
        self.is_dragging = False

    def update_selection_rectangle(self):
        if self.selection_rectangle != None:
            center = self.selection_rectangle.center()
            self.selection_rectangle = QRect(int(center.x() - self.selection_rectangle_size[0] / 2), int(center.y(
            ) - self.selection_rectangle_size[1] / 2), int(self.selection_rectangle_size[0]), int(self.selection_rectangle_size[1]))

    def wheelEvent(self, e):
        delta = 1
        if e.angleDelta().y() < 0:
            delta = -1
        delta *= max(1, self.selection_rectangle_size[0] / 10)

        self.selection_rectangle_size = [self.selection_rectangle_size[0] + delta, self.selection_rectangle_size[1] + delta]

        if self.selection_rectangle_size[0] <= 0:
            self.selection_rectangle_size[0] = 1
        if self.selection_rectangle_size[1] <= 0:
            self.selection_rectangle_size[1] = 1

        if self.should_limit_box_size:
            if self.selection_rectangle_size[0] > 512:
                self.selection_rectangle_size[0] = 512
            if self.selection_rectangle_size[1] > 512:
                self.selection_rectangle_size[1] = 512

        if self.selection_rectangle != None:
            self.update_selection_rectangle()
        self.update()
        
    def set_should_limit_box_size(self, val):
        self.should_limit_box_size = val

    def resize_to_image(self, only_if_smaller=False):
        if self.qt_image != None:
            if only_if_smaller:
                if self.qt_image.width() < self.width() and self.qt_image.height() < self.height():
                    return
            self.resize(self.qt_image.width(), self.qt_image.height())

    def map_widget_to_image(self, pos):
        w, h = self.qt_image.width(), self.qt_image.height()
        window_width = self.width()
        window_height = self.height()
        offset_x = (window_width - w) / 2
        offset_y = (window_height - h) / 2
        return QPoint(int(pos.x() - offset_x), int(pos.y() - offset_y))
    
    def map_widget_to_image_rect(self, widget_rect):
        image_rect = QRect()
        image_rect.setTopLeft(self.map_widget_to_image(widget_rect.topLeft()))
        image_rect.setBottomRight(self.map_widget_to_image(widget_rect.bottomRight()))
        return image_rect

    def crop_image_rect(self, image_rect):
        source_rect = QRect(0, 0, self.selection_rectangle.width(), self.selection_rectangle.height())

        if image_rect.left() < 0:
            source_rect.setLeft(-image_rect.left())
            image_rect.setLeft(0)
        if image_rect.right() >= self.qt_image.width():
            source_rect.setRight(self.selection_rectangle.width() -image_rect.right() + self.qt_image.width() - 1)
            image_rect.setRight(self.qt_image.width())
        if image_rect.top() < 0:
            source_rect.setTop(-image_rect.top())
            image_rect.setTop(0)
        if image_rect.bottom() >= self.qt_image.height():
            source_rect.setBottom(self.selection_rectangle.height() -image_rect.bottom() + self.qt_image.height() - 1)
            image_rect.setBottom(self.qt_image.height())
        return image_rect, source_rect

    def paint_selection(self, add_to_history=True):
        if self.selection_rectangle != None:
            image_rect = self.map_widget_to_image_rect(self.selection_rectangle)
            image_rect, source_rect = self.crop_image_rect(image_rect)
            new_image = self.np_image.copy()
            new_image[image_rect.top():image_rect.bottom(), image_rect.left():image_rect.right(), :3] = self.color
            new_image[image_rect.top():image_rect.bottom(), image_rect.left():image_rect.right(), 3] = 255
            self.set_np_image(new_image, add_to_history=add_to_history)

    def erase_selection(self, add_to_history=True):
        if self.selection_rectangle != None:
            image_rect = self.map_widget_to_image_rect(self.selection_rectangle)
            image_rect, source_rect = self.crop_image_rect(image_rect)
            new_image = self.np_image.copy()
            new_image[image_rect.top():image_rect.bottom(), image_rect.left():image_rect.right(), :] = 0
            self.set_np_image(new_image, add_to_history=add_to_history)
    
    def set_selection_image(self, patch_image):
        if self.selection_rectangle != None:
            image_rect = self.map_widget_to_image_rect(self.selection_rectangle)
            image_rect, source_rect = self.crop_image_rect(image_rect)
            new_image = self.np_image.copy()
            target_width = image_rect.width()
            target_height = image_rect.height()
            patch_np = np.array(patch_image)[source_rect.top():source_rect.bottom(), source_rect.left():source_rect.right(), :][:target_height, :target_width, :]
            if patch_np.shape[-1] == 4:
                patch_np, patch_alpha = patch_np[:, :, :3], patch_np[:, :, 3]
                patch_alpha = (patch_alpha > 128) * 255
            else:
                patch_alpha = np.ones((patch_np.shape[0], patch_np.shape[1])).astype(np.uint8) * 255

            new_image[image_rect.top():image_rect.top() + patch_np.shape[0], image_rect.left():image_rect.left()+patch_np.shape[1], :][patch_alpha > 128] = \
                np.concatenate(
                    [patch_np, patch_alpha[:, :, None]],
                axis=-1)[patch_alpha > 128]
            self.set_np_image(new_image)


    def get_selection_np_image(self):

        image_rect = self.map_widget_to_image_rect(self.selection_rectangle)
        image_rect, source_rect = self.crop_image_rect(image_rect)
        result = np.zeros((self.selection_rectangle.height(), self.selection_rectangle.width(), 4), dtype=np.uint8)

        if image_rect.width() != source_rect.width():
            source_rect.setRight(source_rect.right()-1)

        if image_rect.height() != source_rect.height():
            source_rect.setBottom(source_rect.bottom()-1)

        result[source_rect.top():source_rect.bottom(), source_rect.left():source_rect.right(), :] = \
            self.np_image[image_rect.top():image_rect.bottom(), image_rect.left():image_rect.right(), :]
        return result

    def increase_image_size(self):
        H = SIZE_INCREASE_INCREMENT // 2
        new_image = np.zeros((self.np_image.shape[0] + SIZE_INCREASE_INCREMENT, self.np_image.shape[1] + SIZE_INCREASE_INCREMENT, 4), dtype=np.uint8)
        new_image[H:-H, H:-H, :] = self.np_image
        self.set_np_image(new_image)

    def decrease_image_size(self):
        H = SIZE_INCREASE_INCREMENT // 2
        self.set_np_image(self.np_image[H:-H, H:-H, :])

    def get_mouse_button(self, button):
        if not self.shoulds_swap_buttons:
            return button
        else:
            if button == Qt.MidButton:
                return Qt.LeftButton
            if button == Qt.LeftButton:
                return Qt.MidButton
            return button

    def mousePressEvent(self, e):
        # return super().mousePressEvent(e)
        top_left = QPoint(int(e.pos().x() - self.selection_rectangle_size[0] / 2), int(e.pos().y() - self.selection_rectangle_size[1] / 2))
        self.selection_rectangle = QRect(top_left, QSize(int(self.selection_rectangle_size[0]), int(self.selection_rectangle_size[1])))

        button = self.get_mouse_button(e.button())

        if button == Qt.LeftButton:
            self.is_dragging = True

        if button == Qt.RightButton:
            self.erase_selection()
            self.is_dragging = True

        if button == Qt.MidButton:
            self.paint_selection()
            self.is_dragging = True
            # self.selection_rectangle_size = (256, 256)
            # self.update_selection_rectangle()

        self.update()

    def paintEvent(self, e):
        painter = QPainter(self)

        checkerboard_brush = QBrush()
        checkerboard_brush.setColor(QColor('gray'))
        checkerboard_brush.setStyle(Qt.Dense5Pattern)

        if self.qt_image != None:
            w, h = self.qt_image.width(), self.qt_image.height()
            window_width = self.width()
            window_height = self.height()
            offset_x = (window_width - w) / 2
            offset_y = (window_height - h) / 2
            self.image_rect = QRect(int(offset_x), int(offset_y), int(w), int(h))
            prev_brush = painter.brush()
            painter.fillRect(self.image_rect, checkerboard_brush)
            painter.setBrush(prev_brush)
            painter.drawImage(self.image_rect, self.qt_image)

        if self.saved_mask_state:
            painter.setPen(QPen(Qt.blue,  1, Qt.DashLine))
            painter.drawRect(self.saved_mask_state.box)

        if self.selection_rectangle != None:
            # painter.setBrush(redbrush)
            painter.setPen(QPen(Qt.red,  1, Qt.SolidLine))
            painter.drawRect(self.selection_rectangle)

        if not (self.preview_image is None):
            painter.drawImage(self.selection_rectangle, qimage_from_array(self.preview_image))

        if self.should_preview_scratchpad and (self.scratchpad != None) and (self.scratchpad.isVisible()):
            if (not (self.scratchpad.np_image is None)) and (not (self.scratchpad.selection_rectangle is None)) and (not (self.selection_rectangle is None)):
                try:
                    image = np.array(Image.fromarray(self.scratchpad.get_selection_np_image()).resize((self.selection_rectangle.width(), self.selection_rectangle.height()), Image.LANCZOS))
                    painter.drawImage(self.selection_rectangle, qimage_from_array(image))
                except Exception:
                    print(traceback.format_exc())



    def load_file(self, file_name):
        imdata = Image.open(file_name)
        image_numpy = np.array(imdata)
        self.set_np_image(image_numpy)
        self.resize_to_image(only_if_smaller=True)
        self.update()

    def handle_load_image_button(self):
        file_name = QFileDialog.getOpenFileName()

        if file_name[0]:
            self.load_file(file_name[0])

    def handle_erase_button(self):
        self.erase_selection()
        self.update()

    def handle_undo_button(self):
        self.undo()
        self.update()

    def handle_redo_button(self):
        self.redo()
        self.update()
    
    def get_callback(self):
        def callback(iternum, num_steps, latents, sd):

            done = iternum / num_steps
            # latents_ = 1 / 0.18215 * latents
            # image = ((latents_ / 2 + 0.5).clamp(0, 1).cpu().numpy().transpose(0, 2, 3, 1)[0].copy() * 255).astype(np.uint8)
            image = np.zeros((10, 10, 4), dtype=np.uint8)
            num_highlights = int(image.shape[1] * done)
            bottom = max(1, int(image.shape[0] * 0.1))
            image[:bottom, :num_highlights, :] = np.array([0, 255, 0, 255])
            self.set_preview_image(image)
            self.repaint()
        return callback

    def handle_generate_button(self):
        try:
            if self.selection_rectangle == None:
                QErrorMessage(self).showMessage("Select the target square first")
                return


            prompt = self.get_prompt()
            width = self.selection_rectangle.width()
            height = self.selection_rectangle.height()
            image = self.get_handler().generate(prompt,
                                                width=width,
                                                height=height,
                                                seed=self.seed,
                                                strength=self.strength,
                                                steps=self.steps,
                                                guidance_scale=self.guidance_scale,
                                                callback=self.get_callback())
            self.preview_image = None
            self.set_selection_image(image)
            self.update()
        except Exception:
            print(traceback.format_exc())

    def handle_inpaint_button(self):
        try:
            prompt = self.get_prompt()
            if self.saved_mask_state != None:
                self.selection_rectangle = self.saved_mask_state.box

            image_ = self.get_selection_np_image()
            image = image_[:, :, :3]
            if self.saved_mask_state == None:
                mask = 255 - image_[:, :, 3]
                image, _ = inpaint_functions[self.inpaint_method](image, 255 - mask)
            else:
                mask = 255 - self.saved_mask_state.mask


            inpainted_image = self.get_handler().inpaint(prompt,
                                                         image,
                                                         mask,
                                                         strength=self.strength,
                                                         steps=self.steps,
                                                         guidance_scale=self.guidance_scale,
                                                         seed=self.seed,
                                                         callback=self.get_callback())
            self.preview_image = None
                    

            self.set_selection_image(inpainted_image)
            self.update()
        except:
            print(traceback.format_exc())
            QErrorMessage(self).showMessage("Inpainting failed")

    def handle_quickload_button(self):
        path = get_most_recent_saved_file()
        self.load_file(path)

    def handle_quicksave_button(self):
        quicksave_image(self.np_image)

    def handle_export_button(self):
        path = QFileDialog.getSaveFileName()
        if path[0]:
            quicksave_image(self.np_image, file_path=path[0])

    def handle_select_color_button(self, select_color_button=None):
        color = QColorDialog.getColor()
        if color.isValid():
            self.set_color(color)

            if select_color_button != None:
                sheet = ('background-color: %s' % color.name()) + ';' + ('color: %s' % ('black' if color.lightness() > 128 else 'white')) + ';'
                select_color_button.setStyleSheet(sheet)



    def handle_paint_button(self):
        self.paint_selection()
        self.update()

    def handle_increase_size_button(self):
        self.increase_image_size()
        self.resize_to_image(only_if_smaller=True)
        self.update()

    def handle_decrease_size_button(self):
        self.decrease_image_size()
        self.update()

    def handle_show_scratchpad(self):
        if not (self.scratchpad is None):
            if scratchpad.isVisible():
                self.scratchpad.hide()
            else:
                self.scratchpad.show()

    def handle_paste_scratchpad(self):
        if not (self.scratchpad.np_image is None):
            resized = np.array(
                Image.fromarray(
                    self.scratchpad.get_selection_np_image()).resize(
                        (self.selection_rectangle.width(), self.selection_rectangle.height()), Image.LANCZOS))
            self.set_selection_image(resized)
            self.update()
    
    def handle_seed_change(self, new_seed):
        self.seed = new_seed

    def get_prompt(self):
        return self.prompt_textarea.text() + ", " + self.modifiers_textarea.text()

    def handle_reimagine_button(self):

        try:
            prompt = self.get_prompt()
            image_ = self.get_selection_np_image()
            image = image_[:, :, :3]
            reimagined_image = self.get_handler().reimagine(prompt,
                                                        image,
                                                        steps=self.steps,
                                                        strength=self.strength,
                                                        guidance_scale=self.guidance_scale,
                                                        seed=self.seed,
                                                        callback=self.get_callback())
            self.preview_image = None

            self.set_selection_image(reimagined_image)
            self.update()
        except:
            print(traceback.format_exc())
            QErrorMessage(self).showMessage("Reimagine failed")

def create_select_widget(name, options, select_callback=None):
    container_widget = QWidget()
    selector_widget = QComboBox()
    for option in options:
        selector_widget.addItem(option)
    selector_label = QLabel(name)

    layout = QHBoxLayout()
    layout.addWidget(selector_label)
    layout.addWidget(selector_widget)
    container_widget.setLayout(layout)

    if select_callback != None:
        selector_widget.activated.connect(select_callback)

    return container_widget, selector_widget

def create_slider_widget(name, minimum=0, maximum=1, default=0.5, dtype=float, value_changed_callback=None):
    strength_widget = QWidget()
    strength_slider = QSlider(Qt.Horizontal)
    strength_label = QLabel(name)
    value_text = QLineEdit()
    reset_button = QPushButton('↺')

    def slider_changed():
        value = dtype(strength_slider.value() / 100)
        value_text.setText(str(value))
        if value_changed_callback:
            value_changed_callback(value)

    def value_changed():
        try:
            value = dtype(float(value_text.text()) * 100)
            strength_slider.setValue(int(value))
            if value_changed_callback:
                value_changed_callback(dtype(value_text.text()))
        except Exception:
            print(traceback.format_exc())

    strength_slider.valueChanged.connect(slider_changed)
    value_text.textChanged.connect(value_changed)

    def reset():
        strength_slider.setValue(int(default * 100))
        value_text.setText(str(default))

    reset_button.clicked.connect(reset)

    strength_layout = QHBoxLayout()
    strength_layout.addWidget(strength_label)
    strength_layout.addWidget(strength_slider)
    strength_layout.addWidget(value_text)
    strength_layout.addWidget(reset_button)
    strength_widget.setLayout(strength_layout)

    strength_slider.setMinimum(minimum * 100)
    strength_slider.setMaximum(maximum * 100)
    strength_slider.setValue(int(default * 100))

    strength_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    value_text.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    reset_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    
    strength_slider.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    # set the default value


    return strength_widget, strength_slider, value_text

def handle_coffee_button():
    QDesktopServices.openUrl(QUrl('https://www.buymeacoffee.com/ahrm'))

def handle_twitter_button():
    QDesktopServices.openUrl(QUrl('https://twitter.com/Ali_Mostafavi_'))

def handle_github_button():
    QDesktopServices.openUrl(QUrl('https://github.com/ahrm/UnstableFusion'))

def handle_huggingface_button():
    QDesktopServices.openUrl(QUrl('https://huggingface.co/settings/tokens'))


if __name__ == '__main__':
    stbale_diffusion_manager = StableDiffusionManager()


    app = QApplication(sys.argv)
    icons_path = pathlib.Path(__file__).parent / 'icons'
    appicon = QIcon(str( icons_path / 'unstablefusion.png'))
    github_icon = QIcon(str( icons_path / 'github.png'))
    twitter_icon = QIcon(str( icons_path / 'twitter.png'))

    app.setWindowIcon(appicon)

    image_groupbox = QGroupBox('Edit Image')
    params_groupbox = QGroupBox('Stable Diffusion Parameters')
    run_groupbox = QGroupBox('Run Stable Diffusion')
    save_groupbox = QGroupBox('Save')

    image_groupbox_layout = QVBoxLayout()
    params_groupbox_layout = QVBoxLayout()
    run_groupbox_layout = QVBoxLayout()
    save_groupbox_layout = QVBoxLayout()

    huggingface_token_container = QWidget()
    huggingface_token_layout = QHBoxLayout()
    huggingface_token_label = QLabel('Huggingface Token')
    huggingface_token_text_field = QLineEdit()
    huggingface_token_open_button = QPushButton('Open Token Page')
    huggingface_token_layout.addWidget(huggingface_token_label)
    huggingface_token_layout.addWidget(huggingface_token_text_field)
    huggingface_token_layout.addWidget(huggingface_token_open_button)
    huggingface_token_container.setLayout(huggingface_token_layout)

    huggingface_token_text_field.setEchoMode(QLineEdit.Password)

    huggingface_token_open_button.clicked.connect(handle_huggingface_button)


    tools_widget = QWidget()
    tools_layout = QVBoxLayout()
    load_image_button = QPushButton('Load Image')
    erase_button = QPushButton('Erase')
    paint_widgets_container = QWidget()
    paint_widgets_layout = QHBoxLayout()
    paint_button = QPushButton('Paint')
    select_color_button = QPushButton('Select Color')
    paint_widgets_layout.addWidget(erase_button)
    paint_widgets_layout.addWidget(paint_button)
    paint_widgets_layout.addWidget(select_color_button)
    paint_widgets_container.setLayout(paint_widgets_layout)

    increase_size_container = QWidget()
    increase_size_layout = QHBoxLayout()
    increase_size_button = QPushButton('Increase Size')
    decrease_size_button = QPushButton('Decrease Size')
    increase_size_layout.addWidget(increase_size_button)
    increase_size_layout.addWidget(decrease_size_button)
    increase_size_container.setLayout(increase_size_layout)

    seed_container = QWidget()
    seed_layout = QHBoxLayout()
    seed_text = QLineEdit()
    seed_label = QLabel('Seed')
    seed_text.setText('-1')
    seed_random_button = QPushButton('🎲')
    seed_reset_button = QPushButton('↺')
    seed_layout.addWidget(seed_label)
    seed_layout.addWidget(seed_text)
    seed_layout.addWidget(seed_random_button)
    seed_layout.addWidget(seed_reset_button)
    seed_container.setLayout(seed_layout)

    def random_seed_buton_handler():
        seed_text.setText(str(random.randint(0, 1000000)))

    seed_random_button.clicked.connect(random_seed_buton_handler)

    undo_redo_container = QWidget()
    undo_redo_layout = QHBoxLayout()
    undo_button = QPushButton('Undo')
    redo_button = QPushButton('Redo')
    undo_redo_layout.addWidget(undo_button)
    undo_redo_layout.addWidget(redo_button)
    undo_redo_container.setLayout(undo_redo_layout)

    reimagine_button = QPushButton('Reimagine')
    inpaint_button = QPushButton('Inpaint')

    prompt_textarea = QLineEdit()
    prompt_textarea.setPlaceholderText('Prompt')

    modifiers_textarea = QLineEdit()
    modifiers_textarea.setPlaceholderText('Modifiers')
    modifiers_save_button = QPushButton('Save Modifiers')
    modifiers_load_button = QPushButton('Load Modifiers')
    modifiers_container = QWidget()
    modifiers_layout = QHBoxLayout()
    modifiers_layout.addWidget(modifiers_textarea)
    modifiers_layout.addWidget(modifiers_save_button)
    modifiers_layout.addWidget(modifiers_load_button)
    modifiers_container.setLayout(modifiers_layout)

    def handle_save_modifiers():
        mods = modifiers_textarea.text()
        save_modifiers(mods)

    def handle_load_modifiers():
        mods = load_modifiers()
        modifiers_textarea.setText(mods)

    modifiers_save_button.clicked.connect(handle_save_modifiers)
    modifiers_load_button.clicked.connect(handle_load_modifiers)

    generate_button = QPushButton('Generate')
    save_container = QWidget()
    save_layout = QHBoxLayout()
    quicksave_button = QPushButton('Quick Save')
    quickload_button = QPushButton('Quick Load')
    save_layout.addWidget(quicksave_button)
    save_layout.addWidget(quickload_button)
    save_container.setLayout(save_layout)

    generate_button.setStyleSheet('QPushButton {background: green; color: white;}')
    inpaint_button.setStyleSheet('QPushButton {background: green; color: white;}')
    reimagine_button.setStyleSheet('QPushButton {background: green; color: white;}')


    scratchpad_container = QWidget()
    scratchpad_layout = QHBoxLayout()
    show_scratchpad_button = QPushButton('Show Scratchpad')
    paste_scratchpad_button = QPushButton('Paste From Scratchpad')
    scratchpad_layout.addWidget(show_scratchpad_button)
    scratchpad_layout.addWidget(paste_scratchpad_button)
    scratchpad_container.setLayout(scratchpad_layout)
    export_button = QPushButton('Export')
    widget = PaintWidget(prompt_textarea, modifiers_textarea, stbale_diffusion_manager)
    scratchpad = PaintWidget(prompt_textarea, modifiers_textarea, stbale_diffusion_manager)

    widget.scratchpad = scratchpad
    widget.set_should_preview_scratchpad(True)
    scratchpad.owner = widget
    scratchpad.scratchpad = widget
    widget.owner = scratchpad



    def strength_change_callback(val):
        widget.set_strength(val)
        scratchpad.set_strength(val)

    def steps_change_callback(val):
        widget.set_steps(val)
        scratchpad.set_steps(val)

    def guidance_change_callback(val):
        widget.set_guidance_scale(val)
        scratchpad.set_guidance_scale(val)

    strength_widget, strength_slider, strength_text = create_slider_widget(
        "Strength",
        default=0.75,
        value_changed_callback=strength_change_callback)

    steps_widget, steps_slider, steps_text = create_slider_widget(
        "Steps",
         minimum=1,
         maximum=200,
         default=30,
         dtype=int,
         value_changed_callback=steps_change_callback)

    guidance_widget, guidance_slider, guidance_text = create_slider_widget(
        "Guidance",
         minimum=0,
         maximum=10,
         default=7.5,
         value_changed_callback=guidance_change_callback)
        
    def inpaint_change_callback(num):
        widget.set_inpaint_method(inpaint_options[num])

    inpaint_selector_container, inpaint_selector = create_select_widget(
        'Initializer',
        inpaint_options,
        select_callback=inpaint_change_callback)

    inpaint_container = QWidget()
    inpaint_layout = QHBoxLayout()
    inpaint_layout.addWidget(inpaint_selector_container)
    inpaint_layout.addWidget(inpaint_button)
    inpaint_container.setLayout(inpaint_layout)

    support_container = QWidget()
    support_layout = QHBoxLayout()
    coffee_button = QPushButton('Buy me a coffee')
    github_button = QPushButton()
    twitter_button = QPushButton()
    github_button.setIcon(github_icon)
    twitter_button.setIcon(twitter_icon)
    support_layout.addWidget(coffee_button)
    support_layout.addWidget(github_button)
    support_layout.addWidget(twitter_button)
    support_container.setLayout(support_layout)

    def runtime_change_callback(num):
        if runtime_options[num] == 'local':
            server_address_widget.setDisabled(True)
        else:
            server_address_widget.setEnabled(True)

    runtime_options = ['local', 'server']
    runtime_select_container, runtime_select_widget = create_select_widget('Runtime', runtime_options, select_callback=runtime_change_callback)
    server_address_widget = QLineEdit()

    stbale_diffusion_manager.mode_widget = runtime_select_widget
    stbale_diffusion_manager.huggingface_token_widget = huggingface_token_text_field
    stbale_diffusion_manager.server_address_widget = server_address_widget

    server_address_widget.setPlaceholderText('server address')
    if runtime_select_widget.currentText() == 'local':
        server_address_widget.setDisabled(True)
    server_address_widget.setText('http://127.0.0.1:5000')

    box_size_limit_container = QWidget()
    box_size_limit_label = QLabel('Should limit box size')
    box_size_limit_checkbox = QCheckBox()
    box_size_limit_checkbox.setChecked(True)
    box_size_limit_layout = QHBoxLayout()
    swap_buttons_label = QLabel('Paint using left click')
    swap_buttons_checkbox = QCheckBox()
    swap_buttons_checkbox.setChecked(False)
    box_size_limit_layout.addWidget(box_size_limit_label)
    box_size_limit_layout.addWidget(box_size_limit_checkbox)
    box_size_limit_layout.addWidget(swap_buttons_label)
    box_size_limit_layout.addWidget(swap_buttons_checkbox)

    box_size_limit_container.setLayout(box_size_limit_layout)

    def box_size_limit_callback(state):
        if state == Qt.Checked:
            widget.set_should_limit_box_size(True)
            scratchpad.set_should_limit_box_size(True)
        else:
            widget.set_should_limit_box_size(False)
            scratchpad.set_should_limit_box_size(False)

    def swap_buttons_callback(state):
        if state == Qt.Checked:
            widget.set_should_swap_buttons(True)
            scratchpad.set_should_swap_buttons(True)
        else:
            widget.set_should_swap_buttons(False)
            scratchpad.set_should_swap_buttons(False)

    box_size_limit_checkbox.stateChanged.connect(box_size_limit_callback)
    swap_buttons_checkbox.stateChanged.connect(swap_buttons_callback)

    disable_safety_button = QPushButton('Disable Safety Checker')
    
    def handle_autofill():
        image_ = widget.get_selection_np_image()
        image = image_[:, :, :3]
        mask = 255 - image_[:, :, 3]
        function = inpaint_functions[inpaint_selector.currentText()]
        image, _ = function(image, 255 - mask)
        widget.set_selection_image(image)
        widget.update()

    fill_button = QPushButton('Autofill')
    fill_button.clicked.connect(handle_autofill)

    def handle_save_mask_button():
        widget.save_mask()
        widget.update()

    def handle_forget_mask_button():
        widget.reset_saved_mask()
        widget.update()

    mask_control_container = QWidget()
    mask_control_layout = QHBoxLayout()

    mask_container_label = QLabel('Advanced Inpainting Mask')
    save_mask_button = QPushButton('Save Mask')
    save_mask_button.clicked.connect(handle_save_mask_button)
    forget_mask_button = QPushButton('Forget Mask')
    forget_mask_button.clicked.connect(handle_forget_mask_button)
    mask_control_layout.addWidget(mask_container_label)
    mask_control_layout.addWidget(save_mask_button)
    mask_control_layout.addWidget(forget_mask_button)
    mask_control_container.setLayout(mask_control_layout)

    scroll_area = QScrollArea()

    image_groupbox_layout.addWidget(load_image_button)
    image_groupbox_layout.addWidget(increase_size_container)
    image_groupbox_layout.addWidget(paint_widgets_container)
    image_groupbox_layout.addWidget(undo_redo_container)
    image_groupbox_layout.addWidget(box_size_limit_container)
    image_groupbox_layout.addWidget(fill_button)
    params_groupbox_layout.addWidget(prompt_textarea)
    params_groupbox_layout.addWidget(modifiers_container)
    params_groupbox_layout.addWidget(strength_widget)
    params_groupbox_layout.addWidget(steps_widget)
    params_groupbox_layout.addWidget(guidance_widget)
    params_groupbox_layout.addWidget(seed_container)
    run_groupbox_layout.addWidget(runtime_select_container)
    run_groupbox_layout.addWidget(server_address_widget)
    run_groupbox_layout.addWidget(generate_button)
    run_groupbox_layout.addWidget(inpaint_container)
    run_groupbox_layout.addWidget(mask_control_container)
    run_groupbox_layout.addWidget(reimagine_button)
    save_groupbox_layout.addWidget(save_container)
    save_groupbox_layout.addWidget(export_button)
    image_groupbox.setLayout(image_groupbox_layout)
    params_groupbox.setLayout(params_groupbox_layout)
    save_groupbox.setLayout(save_groupbox_layout)
    run_groupbox.setLayout(run_groupbox_layout)
    tools_layout.addWidget(huggingface_token_container)
    tools_layout.addWidget(image_groupbox)
    tools_layout.addWidget(params_groupbox)
    tools_layout.addWidget(run_groupbox)
    tools_layout.addWidget(save_groupbox)
    tools_layout.addWidget(scratchpad_container)
    # tools_layout.addWidget(disable_safety_button)
    tools_layout.addWidget(support_container)
    tools_widget.setLayout(tools_layout)

    scroll_area.setWidget(tools_widget)

    def handle_disable_safety():
        stbale_diffusion_manager.disable_safety()

    load_image_button.clicked.connect(lambda : widget.handle_load_image_button())
    erase_button.clicked.connect(lambda : widget.handle_erase_button())
    undo_button.clicked.connect(lambda : widget.handle_undo_button())
    redo_button.clicked.connect(lambda : widget.handle_redo_button())
    generate_button.clicked.connect(lambda : widget.handle_generate_button())
    inpaint_button.clicked.connect(lambda : widget.handle_inpaint_button())
    quicksave_button.clicked.connect(lambda : widget.handle_quicksave_button())
    quickload_button.clicked.connect(lambda : widget.handle_quickload_button())
    export_button.clicked.connect(lambda : widget.handle_export_button())
    select_color_button.clicked.connect(lambda : widget.handle_select_color_button( select_color_button))
    paint_button.clicked.connect(lambda : widget.handle_paint_button())
    increase_size_button.clicked.connect(lambda : widget.handle_increase_size_button())
    decrease_size_button.clicked.connect(lambda : widget.handle_decrease_size_button())
    show_scratchpad_button.clicked.connect(lambda : widget.handle_show_scratchpad())
    paste_scratchpad_button.clicked.connect(lambda : widget.handle_paste_scratchpad())
    reimagine_button.clicked.connect(lambda : widget.handle_reimagine_button())
    coffee_button.clicked.connect(lambda : handle_coffee_button())
    twitter_button.clicked.connect(lambda : handle_twitter_button())
    github_button.clicked.connect(lambda : handle_github_button())
    disable_safety_button.clicked.connect(lambda : handle_disable_safety())

    def seed_change_function(val):
        try:
            seed = int(val)
            widget.handle_seed_change(seed)
            scratchpad.handle_seed_change(seed)
        except Exception:
            print(traceback.format_exc())
    
    seed_text.textChanged.connect(seed_change_function)
    seed_reset_button.clicked.connect(lambda : seed_text.setText('-1'))

    widget.setWindowTitle('UnstableFusion')
    scratchpad.setWindowTitle('Scratchpad')
    tools_widget.setWindowTitle('Tools')
    widget.set_np_image(testtexture)
    scratchpad.set_np_image(testtexture)
    widget.resize_to_image()
    widget.show()
    # tools_widget.show()
    scroll_area.resize(tools_widget.sizeHint())
    scroll_area.show()
    app.exec()
