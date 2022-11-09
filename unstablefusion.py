from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys
import numpy as np
from PIL import Image
import traceback

try:
    from diffusionserver import StableDiffusionHandler
except ImportError:
    print(traceback.format_exc())
    print('Could not import StableDiffusionHandler, can not run locally')

import cv2
import pathlib
import time
import json
import os
import datetime
from base64 import decodebytes
import random
import time
import asyncio
import qasync
import httpx
from dataclasses import dataclass

client = httpx.AsyncClient(timeout=None)

SIZE_INCREASE_INCREMENT = 20
brush_options = ['square', 'circle']

inpaint_options = ['cv2_ns', 'cv2_telea',
         'gaussian']
        
def cv2_telea(img, mask):
    ret = cv2.inpaint(img, 255 - mask, 5, cv2.INPAINT_TELEA)
    return ret, mask


def cv2_ns(img, mask):
    ret = cv2.inpaint(img, 255 - mask, 5, cv2.INPAINT_NS)
    return ret, mask

def gaussian_noise(img, mask):
    noise = np.random.randn(mask.shape[0], mask.shape[1], 3)
    noise = (noise + 1) / 2 * 255
    noise = noise.astype(np.uint8)
    nmask = mask.copy()
    nmask[mask > 0] = 1
    img = nmask[:, :, np.newaxis] * img + (1 - nmask[:, :, np.newaxis]) * noise
    return img, mask

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
    'autofill_selection': 'F',
    "small_selection": "1",
    "medium_selection": "2",
    "large_selection": "3",
    "fit_image": "0",
    "save_mask": "Q",
    "forget_mask": "Shift+Q",
    "toggle_paint_using_left_click": "E",
    "pick_color": "Shift+C",
}

def smoothen_mask(original_mask):
    new_mask = (1 - original_mask).copy().astype(np.float32)
    for i in range(10):
        K = min(original_mask.shape[0] // 3, 30)
        new_mask = cv2.blur(new_mask, (K, K))
        new_mask[original_mask == 1] = 0
    new_mask = np.clip(new_mask * 2, 0, 1)
    return 1 - new_mask.astype(np.uint8)


def get_quicksave_path():
    parent_path = pathlib.Path(__file__).parents[0]
    folder_path = parent_path / 'quicksaves'
    folder_path.mkdir(exist_ok=True, parents=True)
    return folder_path

def get_modifiers_path():
    parent_path = pathlib.Path(__file__).parents[0]
    return str(parent_path / 'modifiers.txt')

def get_mod_list_path():
    parent_path = pathlib.Path(__file__).parents[0]
    return str(parent_path / 'mods.txt')

def save_modifiers(mods):
    with open(get_modifiers_path(), 'w') as outfile:
        outfile.write(mods)

def load_modifiers():
    try:
        with open(get_modifiers_path(), 'r') as infile:
            return infile.read()
    except FileNotFoundError:
        pass


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

def hbox(*args):

    container = QWidget()
    layout = QHBoxLayout()
    
    for arg in args:
        if type(arg) == tuple:
            label = QLabel(arg[0])
            layout.addWidget(label)
            layout.addWidget(arg[1])
        else:
            layout.addWidget(arg)
    
    container.setLayout(layout)
    return container

def qimage_from_array(arr):
    maximum = arr.max()
    if arr.shape[-1] != 4:
        arr = np.concatenate([arr, np.ones((arr.shape[0], arr.shape[1], 1)) * 255], axis=2)

    if maximum > 0 and maximum <= 1:
        return  QImage((arr.astype('uint8') * 255).data, arr.shape[1], arr.shape[0], QImage.Format_RGBA8888)
    else:
        return  QImage(arr.astype('uint8').data, arr.shape[1], arr.shape[0], QImage.Format_RGBA8888)

class DummyStableDiffusionHandler:

    def __init__(self):
        pass

    def inpaint(self, prompt, image, mask, strength=0.75, steps=50, guidance_scale=7.5, seed=-1, **kwargs):
        inpainted_image = np.zeros_like(image)
        for i in range(inpainted_image.shape[1]):
            for j in range(inpainted_image.shape[0]):
                inpainted_image[i, j, 0] = 255 * i // inpainted_image.shape[1]
                inpainted_image[i, j, 1] = 255 * j // inpainted_image.shape[1]

        new_image = image.copy()[:, :, :3]
        # new_image[mask > 0] = np.array([255, 0, 0])
        new_image[mask > 0] = inpainted_image[mask > 0]
        time.sleep(3)
        return Image.fromarray(new_image)

    def generate(self, prompt, width=512, height=512, strength=0.75, steps=50, guidance_scale=7.5, seed=-1, **kwargs):

        np_im = np.zeros((height, width, 3), dtype=np.uint8)
        np_im[:, :, 2] = 255
        for i in range(np_im.shape[0]):
            np_im[i, :, 1] = int((i / np_im.shape[0]) * 255)
        for j in range(np_im.shape[1]):
            np_im[:, j, 2] = int((j / np_im.shape[0]) * 255)
        time.sleep(3)
        return Image.fromarray(np_im)

    def reimagine(self, prompt, image, steps=50, guidance_scale=7.5, seed=-1, **kwargs):
        time.sleep(3)
        return image


class ServerStableDiffusionHandler:

    def __init__(self, server_address):
        self.addr = server_address
        if self.addr[-1] != '/':
            self.addr += '/'
    
    async def inpaint(self, prompt, image, mask, strength=0.75, steps=50, guidance_scale=7.5, seed=-1, callback=None, negative_prompt=None, use_gfp=False):
        request_data = {
            'prompt': prompt,
            'strength': strength,
            'steps': steps,
            'guidance_scale': guidance_scale,
            'seed': seed,
            'image': image.tolist(),
            'mask': mask.tolist(),
            'negative_prompt': negative_prompt,
            'use_gfp': use_gfp
        }
        url = self.addr + 'inpaint'

        resp = await client.post(url, json=request_data)

        resp_data = resp.json()
        size = resp_data['image_size']
        mode = resp_data['image_mode']
        image_data = decodebytes(bytes(resp_data['image_data'], encoding='ascii'))

        return Image.frombytes(mode, size, image_data)
    
    async def generate(self, prompt, width=512, height=512, strength=0.75, steps=50, guidance_scale=7.5,seed=-1, callback=None, negative_prompt=None, use_gfp=False):
            request_data = {
                'prompt': prompt,
                'strength': strength,
                'steps': steps,
                'guidance_scale': guidance_scale,
                'seed': seed,
                'width': width,
                'height': height,
                'negative_prompt': negative_prompt,
                'use_gfp': use_gfp
            }
            url = self.addr + 'generate'

            # resp = await requests.post(url, json=request_data)
            resp = await client.post(url, json=request_data)

            resp_data = resp.json()
            size = resp_data['image_size']
            mode = resp_data['image_mode']
            image_data = decodebytes(bytes(resp_data['image_data'], encoding='ascii'))

            return Image.frombytes(mode, size, image_data)
    
    async def reimagine(self, prompt, image, steps=50, guidance_scale=7.5, seed=-1, strength=7.5, callback=None, negative_prompt=None, use_gfp=False):
        request_data = {
            'prompt': prompt,
            'steps': steps,
            'guidance_scale': guidance_scale,
            'seed': seed,
            'strength': strength,
            'image': image.tolist(),
            'negative_prompt': negative_prompt,
            'use_gfp': use_gfp
        }
        url = self.addr + 'reimagine'

        resp = await client.post(url, json=request_data)

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

    def get_local_handler(self, token=True):
        if self.cached_local_handler == None:
            self.cached_local_handler = StableDiffusionHandler(token)
            # self.cached_local_handler = DummyStableDiffusionHandler()

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

    def __init__(self, prompt_textarea_, negative_prompt_textarea_, modifiers_textarea_, stable_diffusion_manager_, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.np_image = None
        self.qt_image = None
        self.selection_rectangle = None
        self.selection_rectangle_size = (100, 100)
        self.is_dragging = False
        self.image_rect = None
        self.window_scale = 1
        self.strength = 0.75
        self.steps = 50
        self.guidance_scale = 7.5
        self.seed = -1
        self.should_preview_scratchpad = False
        self.inpaint_method = inpaint_options[0]
        self.history = []
        self.future = []
        self.color = np.array([0, 0, 0])
        self.scratchpad = None
        self.owner = None
        self.should_limit_box_size = True
        self.shoulds_swap_buttons = False
        self.saved_mask_state = None
        self.brush = 'square'
        self.prompt_textarea = prompt_textarea_
        self.negative_prompt_textarea = negative_prompt_textarea_
        self.modifiers_textarea = modifiers_textarea_
        self.stable_diffusion_manager = stable_diffusion_manager_
        self.preview_image = None
        self.color_pushbutton = None
        self.paint_checkbox = None
        self.smooth_inpaint_checkbox = None
        self.gfpgan_checkbox = None
        self.pending_rect = None

        self.setAcceptDrops(True)
        self.add_shortcuts()

    
    def handle_pick_color(self):
        image = self.get_selection_np_image()
        mean_color_np = image.mean(axis=(0, 1))
        mean_color = QColor(mean_color_np[0], mean_color_np[1], mean_color_np[2])
        self.set_color(mean_color)
    
    def should_use_gfpgan(self):
        if self.gfpgan_checkbox == None:
            return False
        return self.gfpgan_checkbox.isChecked()

    def should_inpaint_smoothly(self):
        if self.smooth_inpaint_checkbox:
            return self.smooth_inpaint_checkbox.isChecked()
        else:
            return False

    def add_shortcuts(self):
        shortcuts = get_shortcut_dict()
        shortcut_function_map = {
            'paste_from_scratchpad': self.handle_paste_scratchpad,
            'decrease_size': self.handle_decrease_size_button,
            'increase_size': self.handle_increase_size_button,
            'export': self.handle_export_button,
            'reimagine': lambda : asyncio.create_task(self.handle_reimagine_button()),
            'inpaint': lambda : asyncio.create_task(self.handle_inpaint_button()),
            'generate': lambda: asyncio.create_task(self.handle_generate_button()),
            'select_color': self.handle_select_color_button,
            'undo': self.undo,
            'redo': self.redo,
            'open': self.handle_load_image_button,
            'toggle_scratchpad': self.handle_show_scratchpad,
            'quicksave': self.handle_quicksave_button,
            'quickload': self.handle_quickload_button,
            'toggle_preview': self.toggle_should_preview_scratchpad,
            'autofill_selection': self.handle_autofill,
            'small_selection': self.set_size_small,
            'medium_selection': self.set_size_medium,
            'large_selection': self.set_size_large,
            'fit_selection': self.set_size_fit_image,
            'save_mask': self.handle_save_mask,
            'forget_mask': self.handle_forget_mask,
            'toggle_paint_using_left_click': self.toggle_should_swap_buttons,
            'pick_color': self.handle_pick_color,
        }

        for name, function in shortcut_function_map.items():
            shortcut = QShortcut(QKeySequence(shortcuts[name]), self)
            shortcut.activated.connect(self.update_and(function))


    def set_pending_rect(self):
        self.pending_rect = self.clone_rect(self.selection_rectangle)
    
    def reset_pending_rect(self):
        self.pending_rect = None

    def inc_window_scale(self):
        self.window_scale *= 1.1

    def dec_window_scale(self):
        self.window_scale /= 1.1

    def set_preview_image(self, preview_image):
        self.preview_image = np.array(Image.fromarray(preview_image).resize((self.selection_rectangle.width(), self.selection_rectangle.height()), Image.NEAREST))

    def set_should_preview_scratchpad(self, val):
        self.should_preview_scratchpad = val

    def toggle_should_preview_scratchpad(self):
        self.set_should_preview_scratchpad(not self.should_preview_scratchpad)

    def toggle_should_swap_buttons(self):
        self.set_should_swap_buttons(not self.shoulds_swap_buttons)

    def set_should_swap_buttons(self, val):
        self.shoulds_swap_buttons = val

        if self.paint_checkbox:
            self.paint_checkbox.setChecked(val)

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
        if self.color_pushbutton:
            self.color_pushbutton.setStyleSheet("background-color: rgb(%d, %d, %d)" % (self.color[0], self.color[1], self.color[2]))

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
            self.selection_rectangle.moveCenter(self.window_to_image_point(e.pos()))

            try:
                if e.buttons() & right_button:
                    self.erase_selection(False)
                if e.buttons() & mid_button:
                    self.paint_selection(False)
            except ValueError:
                pass

            self.update()
            if self.owner != None:
                self.owner.update()

    def mouseReleaseEvent(self, e):
        # if e.button() == Qt.LeftButton:
        self.is_dragging = False

    def update_selection_rectangle(self):
        if self.selection_rectangle != None:
            center = self.selection_rectangle.center()
            center = QPoint(center.x()+1, center.y()+1)
            x_offset = self.selection_rectangle_size[0] - self.selection_rectangle_size[0] // 2
            y_offset = self.selection_rectangle_size[1] - self.selection_rectangle_size[1] // 2

            self.selection_rectangle = QRect(int(center.x() - x_offset), int(center.y(
            ) - y_offset), int(self.selection_rectangle_size[0]), int(self.selection_rectangle_size[1]))

    def wheelEvent(self, e):
        delta = 1
        if e.angleDelta().y() < 0:
            delta = -1
        delta *= max(1, int(self.selection_rectangle_size[0] / 10))

        if QApplication.keyboardModifiers() & Qt.ShiftModifier:
            return

        if QApplication.keyboardModifiers() & Qt.ControlModifier:
            if delta > 0:
                self.inc_window_scale()
            else:
                self.dec_window_scale()

            self.update()
            return

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
    
    def clone_rect(self, widget_rect: QRect):
        return QRect(widget_rect)

    def crop_image_rect(self, image_rect, rect=None):
        if rect == None:
            rect = self.selection_rectangle

        source_rect = QRect(0, 0, rect.width(), rect.height())

        if image_rect.left() < 0:
            source_rect.setLeft(-image_rect.left())
            image_rect.setLeft(0)
        if image_rect.right() >= self.qt_image.width():
            source_rect.setRight(rect.width() -image_rect.right() + self.qt_image.width() - 1)
            image_rect.setRight(self.qt_image.width())
        if image_rect.top() < 0:
            source_rect.setTop(-image_rect.top())
            image_rect.setTop(0)
        if image_rect.bottom() >= self.qt_image.height():
            source_rect.setBottom(rect.height() -image_rect.bottom() + self.qt_image.height() - 1)
            image_rect.setBottom(self.qt_image.height())
        return image_rect, source_rect

    def get_selection_index(self, rect=None):
        if rect == None:
            rect = self.selection_rectangle

        image_rect = self.clone_rect(rect)
        image_rect, source_rect = self.crop_image_rect(image_rect, rect)
        return (slice(image_rect.top(), image_rect.bottom()), slice(image_rect.left(), image_rect.right())),\
                (slice(source_rect.top(), source_rect.bottom()), slice(source_rect.left(), source_rect.right()))

    def paint_selection(self, add_to_history=True):
        if self.selection_rectangle != None:
            image_index, _ = self.get_selection_index()

            new_image = self.np_image.copy()
            new_image[(*image_index, slice(None, 3))] = self.color
            new_image[(*image_index, 3)] = 255
            brush = self.get_brush()

            if not brush is None:
                index = (*image_index, slice(0, 4))
                mask = np.stack([brush, brush, brush, brush], axis=2)
                new_image[index] = mask * self.np_image[index] + (1 - mask) * new_image[index]

            self.set_np_image(new_image, add_to_history=add_to_history)

    def get_brush(self):
        _, index = self.get_selection_index()

        width = self.selection_rectangle.width()-1
        height = self.selection_rectangle.height()-1

        if self.brush == 'circle':
            brush = np.ones((width, height))
            cv2.circle(brush, (width//2, height//2), width//2, 0, -1)
            return brush[index]
        if self.brush == 'square':
            return np.zeros((width, height))[index]


    def erase_selection(self, add_to_history=True):
        if self.selection_rectangle != None:
            image_index, _ = self.get_selection_index()
            index = (*image_index, slice(None, None))

            new_image = self.np_image.copy()
            brush = self.get_brush()

            mask = np.stack([brush, brush, brush, brush], axis=2)
            new_image[index] = (new_image[index] * mask).astype(np.uint8)
            self.set_np_image(new_image, add_to_history=add_to_history)
    

    def set_selection_image(self, patch_image, rect=None):
        if rect == None:
            rect = self.selection_rectangle

        if rect != None:
            image_index, source_index = self.get_selection_index(rect)
            new_image = self.np_image.copy()

            patch_np = np.array(patch_image)[(*source_index, slice(None, None))]

            if patch_np.shape[-1] == 4:
                patch_np, patch_alpha = patch_np[:, :, :3], patch_np[:, :, 3]
                patch_alpha = (patch_alpha > 128) * 255
            else:
                patch_alpha = np.ones((patch_np.shape[0], patch_np.shape[1])).astype(np.uint8) * 255

            index = (*image_index, slice(None, None))
            new_patch = np.concatenate([patch_np, patch_alpha[:, :, None]], axis=-1)
            new_image[index][patch_alpha > 128] = new_patch[patch_alpha > 128]
            self.set_np_image(new_image)

    def get_selection_np_image(self):
        result = np.zeros((self.selection_rectangle.height(), self.selection_rectangle.width(), 4), dtype=np.uint8)
        image_index, source_index = self.get_selection_index()
        result[(*source_index, slice(None, None))] = self.np_image[(*image_index, slice(None, None))]
        return result

    def set_size_small(self):
        self.selection_rectangle_size = (2, 2)
        self.update_selection_rectangle()

    def set_size_medium(self):
        self.selection_rectangle_size = (128, 128)
        self.update_selection_rectangle()
    
    def set_size_large(self):
        self.selection_rectangle_size = (512, 512)
        self.update_selection_rectangle()
    
    def set_size_fit_image(self):
        size = max(self.np_image.shape[0], self.np_image.shape[1])
        self.selection_rectangle_size = (size, size)
        self.selection_rectangle = QRect(0, 0, size, size)
        self.update_selection_rectangle()

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
        pos = self.window_to_image_point(e.pos())
        top_left = QPoint(int(pos.x() - self.selection_rectangle_size[0] / 2), int(pos.y() - self.selection_rectangle_size[1] / 2))
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

        self.update()

    def window_to_image_point(self, point: QPoint):
        new_x = (point.x() - self.width()/2 + self.np_image.shape[1] * self.window_scale / 2 ) / self.window_scale
        new_y = (point.y() - self.height()/2 + self.np_image.shape[0] * self.window_scale / 2) / self.window_scale
        return QPoint(int(new_x), int(new_y))

    def image_to_window_point(self, point: QPoint):
        new_x = point.x() * self.window_scale + (self.width() - self.np_image.shape[1] * self.window_scale) / 2
        new_y = point.y() * self.window_scale + (self.height() - self.np_image.shape[0] * self.window_scale) / 2
        return QPoint(int(new_x), int(new_y))

    def image_to_window_rect(self, rect):
        return QRect(self.image_to_window_point(rect.topLeft()), self.image_to_window_point(rect.bottomRight()))

    def draw_checkerboard_pattern(self, painter):
        image_rect = QRect(0, 0, self.qt_image.width(), self.qt_image.height())
        image_window_rect = self.image_to_window_rect(image_rect)

        w = image_window_rect.width()
        h = image_window_rect.height()
        # painter.fillRect(QRect(0, 0, w, h), QBrush(Qt.white))
        size = 16
        Nx = w // size
        Ny = h // size

        for i in range(Nx):
            for j in range(Ny):
                if (i+j) % 2 == 0:
                    rect = QRect(image_window_rect.bottomLeft().x() + (i * w) // Nx, image_window_rect.topLeft().y() + (j * h) // Ny, w // Nx, h // Ny)
                    painter.fillRect(rect, QBrush(QColor(220, 220, 220)))


    def paintEvent(self, e):
        painter = QPainter(self)

        self.draw_checkerboard_pattern(painter)

        if self.qt_image != None:
            w, h = self.qt_image.width(), self.qt_image.height()
            image_rect = QRect(0, 0, int(w), int(h))
            painter.drawImage(self.image_to_window_rect(image_rect), self.qt_image)


        if self.selection_rectangle != None:
            # painter.setBrush(redbrush)
            painter.setPen(QPen(Qt.red,  1, Qt.SolidLine))
            painter.drawRect(self.image_to_window_rect(self.selection_rectangle))
        if self.saved_mask_state:
            painter.setPen(QPen(Qt.blue,  1, Qt.DashLine))
            painter.drawRect(self.image_to_window_rect(self.saved_mask_state.box))

        if self.pending_rect:
            painter.setPen(QPen(Qt.gray,  1, Qt.DashLine))
            painter.drawRect(self.image_to_window_rect(self.pending_rect))

        if not (self.preview_image is None):
            painter.drawImage(self.image_to_window_rect(self.selection_rectangle), qimage_from_array(self.preview_image))

        if self.should_preview_scratchpad and (self.scratchpad != None) and (self.scratchpad.isVisible()):
            if (not (self.scratchpad.np_image is None)) and (not (self.scratchpad.selection_rectangle is None)) and (not (self.selection_rectangle is None)):
                try:
                    image = np.array(Image.fromarray(self.scratchpad.get_selection_np_image()).resize((self.selection_rectangle.width(), self.selection_rectangle.height()), Image.LANCZOS))
                    painter.drawImage(self.image_to_window_rect(self.selection_rectangle), qimage_from_array(image))
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
        def callback(iternum, num_steps, latents):

            done = (1000 - num_steps.cpu().item()) / 1000
            image = np.zeros((10, 10, 4), dtype=np.uint8)
            num_highlights = int(image.shape[1] * done)
            bottom = max(1, int(image.shape[0] * 0.1))
            image[:bottom, :num_highlights, :] = np.array([0, 255, 0, 255])
            self.set_preview_image(image)
            self.repaint()
        return callback

    async def handle_generate_button(self):
        try:
            if self.selection_rectangle == None:
                QErrorMessage(self).showMessage("Select the target square first")
                return


            prompt = self.get_prompt()
            negative_prompt = self.get_negative_prompt()
            width = self.selection_rectangle.width()
            height = self.selection_rectangle.height()
            rect = self.selection_rectangle
            if type(self.get_handler()) == ServerStableDiffusionHandler:
                self.set_pending_rect()
                image = await self.get_handler().generate(prompt,
                                                    width=width,
                                                    height=height,
                                                    seed=self.seed,
                                                    strength=self.strength,
                                                    steps=self.steps,
                                                    guidance_scale=self.guidance_scale,
                                                    callback=self.get_callback(),
                                                    negative_prompt=negative_prompt,
                                                    use_gfp=self.should_use_gfpgan())
                self.reset_pending_rect()
            else:

                image = self.get_handler().generate(prompt,
                                                    width=width,
                                                    height=height,
                                                    seed=self.seed,
                                                    strength=self.strength,
                                                    steps=self.steps,
                                                    guidance_scale=self.guidance_scale,
                                                    callback=self.get_callback(),
                                                    negative_prompt=negative_prompt,
                                                    use_gfp=self.should_use_gfpgan())
            self.preview_image = None
            self.set_selection_image(image, rect)
            self.update()
        except Exception:
            print(traceback.format_exc())


    async def handle_inpaint_button(self):
        try:
            prompt = self.get_prompt()
            negative_prompt = self.get_negative_prompt()
            if self.saved_mask_state != None:
                self.selection_rectangle = self.saved_mask_state.box

            image_ = self.get_selection_np_image()
            image = image_[:, :, :3]
            if self.saved_mask_state == None:
                mask = 255 - image_[:, :, 3]
                image, _ = inpaint_functions[self.inpaint_method](image, 255 - mask)
            else:
                mask_ = 255 - image_[:, :, 3]
                mask = 255 - self.saved_mask_state.mask
                image, _ = inpaint_functions[self.inpaint_method](image, 255 - mask_)

            rect = self.selection_rectangle
            if type(self.get_handler()) == ServerStableDiffusionHandler:
                self.set_pending_rect()
                inpainted_image = await self.get_handler().inpaint(prompt,
                                                                   image,
                                                                   mask,
                                                                   strength=self.strength,
                                                                   steps=self.steps,
                                                                   guidance_scale=self.guidance_scale,
                                                                   seed=self.seed,
                                                                   callback=self.get_callback(),
                                                                   negative_prompt=negative_prompt,
                                                                   use_gfp=self.should_use_gfpgan())
                self.reset_pending_rect()
            else:
                inpainted_image = self.get_handler().inpaint(prompt,
                                                            image,
                                                            mask,
                                                            strength=self.strength,
                                                            steps=self.steps,
                                                            guidance_scale=self.guidance_scale,
                                                            seed=self.seed,
                                                            callback=self.get_callback(),
                                                            negative_prompt=negative_prompt,
                                                            use_gfp=self.should_use_gfpgan())

            self.preview_image = None
                    

            # add mask as alpha channel
            # inpainted_image = np.concatenate([inpainted_image, mask[:, :, np.newaxis]], axis=2)

            if self.should_inpaint_smoothly():
                patch_alpha = mask.astype(np.float32) / 255
                patch_alpha[:, 0] = 0
                patch_alpha[:, -1] = 0
                patch_alpha[0, :] = 0
                patch_alpha[-1, :] = 0

                patch_alpha = smoothen_mask(patch_alpha)
                patch_alpha = np.stack([patch_alpha] * 3, axis=2)
                inpainted_image = (inpainted_image * patch_alpha + image * (1-patch_alpha)).astype(np.uint8)

            self.set_selection_image(inpainted_image, rect)
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


    def reset(self):
        self.set_np_image(get_texture())
        self.resize_to_image()
        self.update()

    def handle_autofill(self):
        image_ = self.get_selection_np_image()
        image = image_[:, :, :3]
        mask = 255 - image_[:, :, 3]
        function = inpaint_functions[self.inpaint_method]
        image, _ = function(image, 255 - mask)
        self.set_selection_image(image)
        self.update()

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
        if not (self.scratchpad.np_image is None) and not (self.scratchpad.selection_rectangle is None):
            resized = np.array(
                Image.fromarray(
                    self.scratchpad.get_selection_np_image()).resize(
                        (self.selection_rectangle.width(), self.selection_rectangle.height()), Image.LANCZOS))
            self.set_selection_image(resized)
            self.update()
    
    def handle_seed_change(self, new_seed):
        self.seed = new_seed

    def handle_save_mask(self):
        self.save_mask()
        self.update()

    def handle_forget_mask(self):
        self.reset_saved_mask()
        self.update()

    def get_prompt(self):
        return self.prompt_textarea.text() + ", " + self.modifiers_textarea.text()

    def get_negative_prompt(self):
        res = self.negative_prompt_textarea.text()
        return res
        # if len(res) == 0:
        #     return None
        # return [res]

    async def handle_reimagine_button(self):

        try:
            prompt = self.get_prompt()
            negative_prompt = self.get_negative_prompt()
            image_ = self.get_selection_np_image()
            image = image_[:, :, :3]
            rect = self.selection_rectangle
            if type(self.get_handler()) == ServerStableDiffusionHandler:
                self.set_pending_rect()
                reimagined_image = await self.get_handler().reimagine(prompt,
                                                            image,
                                                            steps=self.steps,
                                                            strength=self.strength,
                                                            guidance_scale=self.guidance_scale,
                                                            seed=self.seed,
                                                            callback=self.get_callback(),
                                                            negative_prompt=negative_prompt,
                                                            use_gfp=self.should_use_gfpgan())
                self.reset_pending_rect()
            else:
                reimagined_image = self.get_handler().reimagine(prompt,
                                                            image,
                                                            steps=self.steps,
                                                            strength=self.strength,
                                                            guidance_scale=self.guidance_scale,
                                                            seed=self.seed,
                                                            callback=self.get_callback(),
                                                            negative_prompt=negative_prompt,
                                                            use_gfp=self.should_use_gfpgan())

            self.preview_image = None

            self.set_selection_image(reimagined_image, rect)
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

def handle_colab_button():
    QDesktopServices.openUrl(QUrl('https://colab.research.google.com/github/ahrm/UnstableFusion/blob/main/UnstableFusionServer.ipynb'))

def handle_advanced_inpainting_doc_button():
    QDesktopServices.openUrl(QUrl('https://github.com/ahrm/UnstableFusion#how-to-use-advanced-inpainting'))

class PromptLineEdit(QLineEdit):

    def __init__(self, mods, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.suggestions = mods
        self.should_suggest = True

        def text_changed(text):
            if not self.should_suggest:
                return

            last_part = text.split(',')[-1].strip()
            if len(last_part) == 0:
                return
            suggestion = self.match_suggestion(last_part)
            if suggestion:
                rest= suggestion[len(last_part):]
                self.setText(text + rest)
                self.setSelection(len(text), len(text) + len(rest))

        self.textChanged.connect(text_changed)
    
    def keyPressEvent(self, e: QKeyEvent):
        disablers = [Qt.Key_Backspace, Qt.Key_Delete]
        completers = [Qt.Key_Return, Qt.Key_Tab]

        if e.key() in disablers:
            self.should_suggest = False
        else:
            self.should_suggest = True
        
        if e.key() in completers:
            self.setSelection(len(self.text()), len(self.text()))
            return

        return super().keyPressEvent(e)

    def match_suggestion(self, text):
        for sug in self.suggestions:
            if sug.startswith(text):
                return sug
        return None
    
    
class CustomScroll(QScrollArea):

    def __init__(self):
        QScrollArea.__init__(self)

    def wheelEvent(self, ev):
        if ev.type() == QEvent.Wheel:
            if QApplication.keyboardModifiers() & Qt.ShiftModifier:
                if QApplication.keyboardModifiers() & Qt.ControlModifier:
                    self.verticalScrollBar().setValue(self.verticalScrollBar().value() - ev.angleDelta().y())
                else:
                    self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - ev.angleDelta().y())
                ev.accept()
            else:
                ev.ignore()

if __name__ == '__main__':
    stbale_diffusion_manager = StableDiffusionManager()

    with open(get_mod_list_path(), 'r', encoding='utf8') as infile:
        mods = infile.read().split('\n')

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

    tools_widget = QWidget()
    tools_layout = QVBoxLayout()

    huggingface_token_text_field = QLineEdit()
    huggingface_token_text_field.setEchoMode(QLineEdit.Password)
    huggingface_token_open_button = QPushButton('Open Token Page')
    huggingface_token_container = hbox(('Huggingface Token', huggingface_token_text_field), huggingface_token_open_button)

    load_image_button = QPushButton('Load Image')

    increase_size_button = QPushButton('Increase Size')
    decrease_size_button = QPushButton('Decrease Size')
    increase_size_container = hbox(increase_size_button, decrease_size_button)

    erase_button = QPushButton('Erase')
    paint_button = QPushButton('Paint')
    select_color_button = QPushButton('Select Color')
    pick_color_button = QPushButton('Pick Color')
    paint_widgets_container = hbox(erase_button, paint_button, pick_color_button, select_color_button)

    undo_button = QPushButton('Undo')
    redo_button = QPushButton('Redo')
    undo_redo_container = hbox(undo_button, redo_button)
    reset_button = QPushButton('Reset')

    box_size_limit_checkbox = QCheckBox()
    box_size_limit_checkbox.setChecked(True)
    swap_buttons_checkbox = QCheckBox()
    swap_buttons_checkbox.setChecked(False)
    brush_select_widget, brush_selector  = create_select_widget('Brush', brush_options)
    box_size_limit_container = hbox(
        ('Should limit box size', box_size_limit_checkbox),
        ('Paint using left click', swap_buttons_checkbox),
        brush_select_widget)

    fill_button = QPushButton('Autofill')

    prompt_textarea = QLineEdit()
    prompt_textarea.setPlaceholderText('Prompt')

    negative_prompt_textarea = QLineEdit()
    negative_prompt_textarea.setPlaceholderText('Negative Prompt')

    modifiers_textarea = PromptLineEdit(mods)
    modifiers_textarea.setPlaceholderText('Modifiers')
    modifiers_save_button = QPushButton('Save Modifiers')
    modifiers_load_button = QPushButton('Load Modifiers')
    modifiers_container = hbox(modifiers_textarea, modifiers_save_button, modifiers_load_button)

    seed_text = QLineEdit()
    seed_text.setText('-1')
    seed_random_button = QPushButton('🎲')
    seed_reset_button = QPushButton('↺')
    seed_container = hbox(('Seed', seed_text), seed_random_button, seed_reset_button)

    def random_seed_buton_handler():
        seed_text.setText(str(random.randint(0, 1000000)))


    generate_button = QPushButton('Generate')
    reimagine_button = QPushButton('Reimagine')
    inpaint_button = QPushButton('Inpaint')
    generate_button.setStyleSheet('QPushButton {background: green; color: white;}')
    inpaint_button.setStyleSheet('QPushButton {background: green; color: white;}')
    reimagine_button.setStyleSheet('QPushButton {background: green; color: white;}')


    def handle_save_modifiers():
        mods = modifiers_textarea.text()
        save_modifiers(mods)

    def handle_load_modifiers():
        mods = load_modifiers()
        if mods:
            modifiers_textarea.setText(mods)


    quicksave_button = QPushButton('Quick Save')
    quickload_button = QPushButton('Quick Load')
    save_container = hbox(quicksave_button, quickload_button)

    show_scratchpad_button = QPushButton('Show Scratchpad')
    paste_scratchpad_button = QPushButton('Paste From Scratchpad')
    scratchpad_container = hbox(show_scratchpad_button, paste_scratchpad_button)

    export_button = QPushButton('Export')

    widget = PaintWidget(prompt_textarea, negative_prompt_textarea, modifiers_textarea, stbale_diffusion_manager)
    scratchpad = PaintWidget(prompt_textarea, negative_prompt_textarea, modifiers_textarea, stbale_diffusion_manager)

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

    smooth_inpaint_checkbox = QCheckBox('Smooth Inpaint')
    inpaint_container = hbox(inpaint_selector_container, smooth_inpaint_checkbox, inpaint_button)

    gfpgan_checkbox = QCheckBox('Use GFPGAN if available')

    coffee_button = QPushButton('Buy me a coffee')
    github_button = QPushButton()
    twitter_button = QPushButton()
    github_button.setIcon(github_icon)
    twitter_button.setIcon(twitter_icon)

    support_container = hbox(coffee_button, github_button, twitter_button)

    def runtime_change_callback(num):
        if runtime_options[num] == 'local':
            server_container.setDisabled(True)
        else:
            server_container.setEnabled(True)

    runtime_options = ['local', 'server']
    runtime_select_container, runtime_select_widget = create_select_widget('Runtime', runtime_options, select_callback=runtime_change_callback)
    
    server_address_widget = QLineEdit()
    open_colab_widget = QPushButton('Open Colab Notebook')
    server_container = hbox(server_address_widget, open_colab_widget)


    stbale_diffusion_manager.mode_widget = runtime_select_widget
    stbale_diffusion_manager.huggingface_token_widget = huggingface_token_text_field
    stbale_diffusion_manager.server_address_widget = server_address_widget

    server_address_widget.setPlaceholderText('server address')
    if runtime_select_widget.currentText() == 'local':
        server_container.setDisabled(True)

    server_address_widget.setText('http://127.0.0.1:5000')

    def brush_select_callback(num):
        option = brush_options[num]
        widget.brush = option
        scratchpad.brush = option

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

    
    def handle_autofill():
        widget.handle_autofill()


    save_mask_button = QPushButton('Save Mask')
    forget_mask_button = QPushButton('Forget Mask')
    advanced_inpainting_doc_button = QPushButton('?')
    mask_control_container = hbox(('Advanced Inpainting Mask', save_mask_button), forget_mask_button, advanced_inpainting_doc_button)

    scroll_area = QScrollArea()

    image_groupbox_layout.addWidget(load_image_button)
    image_groupbox_layout.addWidget(increase_size_container)
    image_groupbox_layout.addWidget(paint_widgets_container)
    image_groupbox_layout.addWidget(undo_redo_container)
    image_groupbox_layout.addWidget(reset_button)
    image_groupbox_layout.addWidget(box_size_limit_container)
    image_groupbox_layout.addWidget(fill_button)
    params_groupbox_layout.addWidget(prompt_textarea)
    params_groupbox_layout.addWidget(negative_prompt_textarea)
    params_groupbox_layout.addWidget(modifiers_container)
    params_groupbox_layout.addWidget(strength_widget)
    params_groupbox_layout.addWidget(steps_widget)
    params_groupbox_layout.addWidget(guidance_widget)
    params_groupbox_layout.addWidget(seed_container)
    run_groupbox_layout.addWidget(runtime_select_container)
    run_groupbox_layout.addWidget(server_container)
    run_groupbox_layout.addWidget(generate_button)
    run_groupbox_layout.addWidget(inpaint_container)
    run_groupbox_layout.addWidget(mask_control_container)
    run_groupbox_layout.addWidget(reimagine_button)
    run_groupbox_layout.addWidget(gfpgan_checkbox)
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
    tools_layout.addWidget(support_container)
    tools_widget.setLayout(tools_layout)

    scroll_area.setWidget(tools_widget)


    load_image_button.clicked.connect(lambda : widget.handle_load_image_button())
    erase_button.clicked.connect(lambda : widget.handle_erase_button())
    undo_button.clicked.connect(lambda : widget.handle_undo_button())
    redo_button.clicked.connect(lambda : widget.handle_redo_button())
    generate_button.clicked.connect(lambda : asyncio.create_task(widget.handle_generate_button()))
    inpaint_button.clicked.connect(lambda : asyncio.create_task(widget.handle_inpaint_button()))
    reimagine_button.clicked.connect(lambda : asyncio.create_task(widget.handle_reimagine_button()))
    quicksave_button.clicked.connect(lambda : widget.handle_quicksave_button())
    quickload_button.clicked.connect(lambda : widget.handle_quickload_button())
    export_button.clicked.connect(lambda : widget.handle_export_button())
    select_color_button.clicked.connect(lambda : widget.handle_select_color_button( select_color_button))
    pick_color_button.clicked.connect(lambda : widget.handle_pick_color())
    paint_button.clicked.connect(lambda : widget.handle_paint_button())
    increase_size_button.clicked.connect(lambda : widget.handle_increase_size_button())
    decrease_size_button.clicked.connect(lambda : widget.handle_decrease_size_button())
    show_scratchpad_button.clicked.connect(lambda : widget.handle_show_scratchpad())
    paste_scratchpad_button.clicked.connect(lambda : widget.handle_paste_scratchpad())
    coffee_button.clicked.connect(lambda : handle_coffee_button())
    twitter_button.clicked.connect(lambda : handle_twitter_button())
    github_button.clicked.connect(lambda : handle_github_button())
    save_mask_button.clicked.connect(lambda : widget.handle_save_mask())
    forget_mask_button.clicked.connect(lambda : widget.handle_forget_mask())
    huggingface_token_open_button.clicked.connect(handle_huggingface_button)
    brush_selector.activated.connect(brush_select_callback)
    fill_button.clicked.connect(handle_autofill)
    modifiers_save_button.clicked.connect(handle_save_modifiers)
    modifiers_load_button.clicked.connect(handle_load_modifiers)
    seed_random_button.clicked.connect(random_seed_buton_handler)
    open_colab_widget.clicked.connect(handle_colab_button)
    advanced_inpainting_doc_button.clicked.connect(handle_advanced_inpainting_doc_button)
    reset_button.clicked.connect(lambda : widget.reset())

    widget.color_pushbutton = select_color_button
    widget.paint_checkbox = swap_buttons_checkbox
    widget.smooth_inpaint_checkbox = smooth_inpaint_checkbox
    widget.gfpgan_checkbox = gfpgan_checkbox

    def seed_change_function(val):
        try:
            seed = int(val)
            widget.handle_seed_change(seed)
            scratchpad.handle_seed_change(seed)
        except Exception:
            print(traceback.format_exc())
    
    seed_text.textChanged.connect(seed_change_function)
    seed_reset_button.clicked.connect(lambda : seed_text.setText('-1'))

    initial_texture = get_texture()

    widget_container = CustomScroll()
    widget_container.setWidget(widget)
    widget_container.resize(initial_texture.shape[0] + 10, initial_texture.shape[1] + 10)
    widget_container.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

    widget.setWindowTitle('UnstableFusion')
    scratchpad.setWindowTitle('Scratchpad')
    tools_widget.setWindowTitle('Tools')
    widget.set_np_image(initial_texture)
    scratchpad.set_np_image(initial_texture)
    widget.resize_to_image()
    # widget.show()
    widget_container.show()
    # tools_widget.show()
    scroll_area.resize(tools_widget.sizeHint())
    scroll_area.show()

    # async def main():
    #     app.exec()
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)
    with loop:
        loop.run_forever()


    # asyncio.run(main())
    # app.exec()
