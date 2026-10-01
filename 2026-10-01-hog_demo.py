"""HoG "I spy" demo for Tutorial 4 (2026-10-01, Object Detection).

Find every digit 3 in a cluttered scene of ~100 MNIST digits, first by hand
using only the histogram of oriented gradients (HoG) under the magnifier, then
with your sliding-window detector from hog_detector.py.

Image pane (left): the scene, hidden until you reveal it (V). The blind test
scene is never revealed. The magnifier is the HoG window.
Sidebar (right): the live HoG under the magnifier, and at the bottom the
reference HoG of the train 3s with how well the two match (or the SVM).
Mode bar (bottom): Manual / Scan / Train, the scene, the score, and the state
of your TODOs E1-E6 in hog_detector.py.

Keys
    Tab             next mode (or click the mode buttons)
    click / Space   Manual: "there is a 3 here"; Scan: run the scan; Train: train the SVM
    wheel, Q / E    probe size, probe rotation (-/+ 5 degrees); 0 resets both
    1-5             HoG presets; B / C / U / O: bins, cells, signed, align
    A               reference from upright 3s or 3s as found
    G               bars or star glyphs
    M               Scan: score with the template or the SVM
    [ / ]           Scan: lower / raise the detection threshold
    H               Scan: score heatmap
    V               reveal the scene (not the blind test)
    T / N           next scene / new random test scene
    R               reset the found 3s and the scan of this scene
    P               print the live descriptor
    Esc             quit

hog_detector.py and answers/ are reloaded automatically when they change.

Run:  python 2026-10-01-hog_demo.py            (--no-answers ignores released answers)
"""
import argparse
import importlib
import math
import os
import queue
import random
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
import pygame  # noqa: E402

import hog_lib as L  # noqa: E402
import hog_scenes as hs  # noqa: E402

PAD = 16
TOP = 52
IMG_W, IMG_H = hs.SIZE
SIDE_W = 340
BAR_H = 66
WIN_W = PAD + IMG_W + PAD + SIDE_W + PAD
WIN_H = TOP + IMG_H + 12 + BAR_H + 8
IMG_X, IMG_Y = PAD, TOP
SIDE_X = IMG_X + IMG_W + PAD
BAR_Y = TOP + IMG_H + 12

BG = (248, 250, 252)          # #f8fafc
FG = (15, 23, 42)             # #0f172a
MUTED = (100, 116, 139)       # #64748b
TRACK = (203, 213, 225)       # #cbd5e1
ACCENT = (37, 99, 235)        # #2563eb
PANE_BG = (226, 232, 240)     # #e2e8f0
GOOD = (22, 163, 74)
WARN = (220, 38, 38)
AMBER = (217, 119, 6)
GLYPH_BG = (15, 23, 42)
MONO = 'menlo,consolas,dejavusansmono,couriernew,monospace'

MODES = ['manual', 'scan', 'train']
MODE_NAMES = {'manual': 'Manual', 'scan': 'Scan', 'train': 'Train'}
HINTS = {
    'manual': 'click: a 3 is here  V: reveal  wheel Q E: probe size/angle  1-5 B C U O: HoG  '
              'A: reference  G: glyphs  R: reset  Tab: mode',
    'scan': 'Space: scan  M: template/SVM  [ ]: threshold  H: heatmap  V: reveal  T N: scene  '
            '1-5: HoG  R: reset  Tab: mode',
    'train': 'Space: train the SVM on the train scene (DetectConfig.svm_angles)  1-5: HoG  Tab: mode',
}
STATUS_STYLE = {'pass': ('ok', GOOD), 'answer': ('answer', ACCENT), 'todo': ('TODO', MUTED),
                'needs': ('needs', AMBER), 'fail': ('FAIL', WARN)}
LENS_CENTER, LENS_RADIUS = 175.5, 144.0     # in assets/magnifier.png (512 x 512)


def bin_color(cfg, b):
    """Colour of orientation bin b: hue follows the angle around the period."""
    period = 360.0 if cfg.signed else 180.0
    c = pygame.Color(0)
    c.hsva = (float(L.bin_centers(cfg)[b] / period * 360) % 360, 65, 92, 100)
    return c


def gray_surface(img):
    a = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    return pygame.surfarray.make_surface(np.repeat(a.T[..., None], 3, axis=2))


class SceneState:
    def __init__(self):
        self.found, self.misses = set(), []
        self.revealed = False
        self.scan = None       # result dict of the last scan
        self.heat = None       # pygame surface


class App:
    def __init__(self, args):
        self.args = args
        if args.no_answers:
            os.environ['HOG_NO_ANSWERS'] = '1'
        pygame.init()
        pygame.display.set_caption('HoG I spy - Tutorial 4 - 2026-10-01')
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        self.font = pygame.font.SysFont(MONO, 15)
        self.bold = pygame.font.SysFont(MONO, 15, bold=True)
        self.small = pygame.font.SysFont(MONO, 12)
        self.clock = pygame.time.Clock()
        icon = pygame.image.load(str(hs.ASSETS / 'magnifier.png')).convert_alpha()
        self.icon_small = pygame.transform.smoothscale(icon, (26, 26))
        self.lens = icon.copy()                # see-through lens for the cursor
        alpha = pygame.surfarray.pixels_alpha(self.lens)
        yy, xx = np.mgrid[0:512, 0:512]
        inside = np.hypot(xx - LENS_CENTER, yy - LENS_CENTER) < LENS_RADIUS - 2
        alpha[inside.T] = np.minimum(alpha[inside.T], 40)
        del alpha
        self.lens_cache = {}

        import hog_detector
        self.hd = hog_detector
        self.scenes = [hs.load_train(), hs.load_blind_test()]
        self.states = [SceneState(), SceneState()]
        self.scene_i = 0
        self.mode = 'manual'
        self.preset_i, self.cfg = 0, L.PRESETS[0]
        self.upright, self.view = True, 'bars'
        self.probe_size, self.probe_angle = float(L.WINDOW), 0.0
        self.scorer = 'template'
        self.thresholds = {}
        self.svm = None
        self.msg, self.msg_color = 'Find every 3. The image is hidden: use the HoG.', FG
        self.jobs, self.busy = queue.Queue(), None
        self.live = self.live_err = None
        self.live_key = None
        self.watch = self.code_mtimes()
        self.last_watch = time.time()
        self.code_changed(initial=True)

    # ------------------------------------------------------------ state helpers

    @property
    def scene(self):
        return self.scenes[self.scene_i]

    @property
    def st(self):
        return self.states[self.scene_i]

    def say(self, text, color=FG):
        self.msg, self.msg_color = text, color

    def code_mtimes(self):
        files = [HERE / 'hog_detector.py'] + sorted((HERE / 'answers').glob('*.py'))
        return {str(f): f.stat().st_mtime for f in files if f.exists()}

    def check_reload(self):
        if time.time() - self.last_watch < 0.5:
            return
        self.last_watch = time.time()
        now = self.code_mtimes()
        if now == self.watch:
            return
        self.watch = now
        for name in [m for m in sys.modules if m.startswith('answers.')]:
            del sys.modules[name]
        importlib.invalidate_caches()
        try:
            importlib.reload(self.hd)
        except Exception as e:
            self.say(f'hog_detector.py did not load: {type(e).__name__}: {e}', WARN)
            return
        self.code_changed()

    def code_changed(self, initial=False):
        self.status = L.self_test(self.hd)
        self.thresholds = {}
        self.live_key = None
        self.update_reference()
        fails = [(t, n, d) for t, n, s, d in self.status if s == 'fail']
        if fails:
            self.say('{} {}(): {}'.format(*fails[0]), WARN)
        elif not initial:
            todo = [t for t, _, s, _ in self.status if s in ('todo', 'needs')]
            self.say('Reloaded hog_detector.py' + (f'; still to do: {" ".join(todo)}' if todo else
                                                     '; all self-tests pass'), ACCENT)

    def update_reference(self):
        try:
            self.ref = L.Template(self.scenes[0], self.cfg, self.hd, self.upright)
            self.ref_err = None
        except Exception as e:
            self.ref, self.ref_err = None, self.error_text(e)
        self.live_key = None

    def error_text(self, e):
        if isinstance(e, NotImplementedError):
            return f'{e} (hog_detector.py)'
        return f'{type(e).__name__}: {e}'

    def set_cfg(self, cfg, preset_i=None):
        self.cfg = cfg
        self.preset_i = preset_i
        self.update_reference()
        for s in self.states:
            s.heat = None

    def detect_config(self):
        try:
            return self.hd.DetectConfig()
        except Exception as e:
            self.say(f'DetectConfig: {self.error_text(e)}', WARN)
            return None

    def threshold(self, scorer):
        if scorer not in self.thresholds:
            dc = self.detect_config()
            self.thresholds[scorer] = (dc.template_threshold if scorer == 'template'
                                       else dc.svm_threshold) if dc else 0.0
        return self.thresholds[scorer]

    def svm_ok(self):
        return self.svm is not None and self.svm['cfg'] == self.cfg

    def score_fn(self, scorer):
        if scorer == 'svm':
            return self.svm['model'].decision_function
        return self.ref.score

    # ------------------------------------------------------------ background jobs

    def run_job(self, label, fn, done):
        if self.busy:
            self.say(f'Busy: {self.busy}', AMBER)
            return

        def work():
            try:
                result = fn()
                self.jobs.put((done, result, None))
            except Exception as e:
                if not isinstance(e, NotImplementedError):
                    traceback.print_exc()
                self.jobs.put((done, None, e))

        self.busy = label
        self.say(f'{label} ...', ACCENT)
        threading.Thread(target=work, daemon=True).start()

    def poll_jobs(self):
        while not self.jobs.empty():
            done, result, err = self.jobs.get()
            self.busy = None
            if err is not None:
                self.say(self.error_text(err), WARN)
            else:
                done(result)

    def start_scan(self):
        if self.scorer == 'template' and self.ref is None:
            self.say(f'No reference HoG: {self.ref_err}', WARN)
            return
        if self.scorer == 'svm' and not self.svm_ok():
            self.say('Train the SVM for this HoG first (Train mode, Space)', AMBER)
            return
        dc = self.detect_config()
        if dc is None:
            return
        scene, st, cfg, scorer, hd = self.scene, self.st, self.cfg, self.scorer, self.hd
        thr, score_fn = self.threshold(scorer), self.score_fn(scorer)

        def fn():
            t = time.time()
            boxes, scores = hd.detect(scene.image, lambda im, stride: L.dense_windows(im, cfg, hd, stride),
                                      score_fn, thr, dc)
            boxes = np.asarray(boxes, float).reshape(-1, 4)
            scores = np.asarray(scores, float).reshape(-1)
            ev = L.evaluate(boxes, scores, scene.three_boxes(), hd.iou)
            return dict(boxes=boxes, scores=scores, ev=ev, thr=thr, scorer=scorer, cfg=cfg,
                        secs=time.time() - t, levels=len(dc.scales) * len(dc.angles))

        def done(r):
            st.scan = r
            e = r['ev']
            hint = ''
            if (cfg.layout == 'rings' or cfg.align) and len(dc.angles) > 1:
                hint = '. This HoG ignores rotation: try DetectConfig.angles = (0,)'
            self.say(f'{len(r["boxes"])} detections over {r["levels"]} pyramid levels in '
                     f'{r["secs"]:.1f} s{hint}', GOOD if e['fn'] == 0 else FG)
        self.run_job('Scanning', fn, done)

    def start_heat(self):
        if self.scorer == 'template' and self.ref is None or self.scorer == 'svm' and not self.svm_ok():
            self.say('Nothing to score with yet', AMBER)
            return
        scene, st, cfg, hd, scorer = self.scene, self.st, self.cfg, self.hd, self.scorer
        score_fn, thr = self.score_fn(scorer), self.threshold(scorer)
        span = 0.3 if scorer == 'template' else 2.5       # fade in below the threshold

        def fn():
            centers, feats = L.dense_windows(scene.image, cfg, hd, 4)
            s = np.asarray(score_fn(feats))
            v = np.clip((s - thr + span) / span, 0, 1)
            heat = np.zeros((IMG_H, IMG_W))
            step = 8 if cfg.layout == 'rings' else 4
            for (x, y), val in zip(centers, v):
                x0, y0 = int(x - step / 2 + 0.5), int(y - step / 2 + 0.5)
                heat[y0:y0 + step, x0:x0 + step] = val
            return heat

        def done(heat):
            surf = pygame.Surface((IMG_W, IMG_H), pygame.SRCALPHA)
            surf.fill((*AMBER, 0))
            alpha = pygame.surfarray.pixels_alpha(surf)
            alpha[:] = (heat.T * 210).astype(np.uint8)
            del alpha
            st.heat = surf
            self.say('Heatmap: score of the upright, scale-1 window at each pixel (solid = above threshold)', FG)
        self.run_job('Computing heatmap', fn, done)

    def start_train(self):
        dc = self.detect_config()
        if dc is None:
            return
        train, cfg, hd = self.scenes[0], self.cfg, self.hd

        def fn():
            t = time.time()
            X, y, wins = L.sample_windows(train, cfg, hd, dc.svm_angles)
            model = L.LinearSVM().fit(X, y)
            acc = float(np.mean(np.sign(model.decision_function(X)) == y))
            return dict(model=model, cfg=cfg, wins=wins, n_pos=int(np.sum(y > 0)),
                        n_neg=int(np.sum(y < 0)), acc=acc, secs=time.time() - t,
                        angles=tuple(dc.svm_angles))

        def done(r):
            self.svm = r
            self.scorer = 'svm'
            self.say(f'SVM trained in {r["secs"]:.1f} s. Scan mode now scores with it (M toggles).', GOOD)
        self.run_job('Training the SVM', fn, done)

    def new_scene(self):
        seed = random.randrange(1, 1_000_000)

        def fn():
            return hs.make_scene(seed, 'test')

        def done(scene):
            self.scenes.append(scene)
            self.states.append(SceneState())
            self.scene_i = len(self.scenes) - 1
            self.say(f'New test scene #{seed}: {len(scene.threes)} threes. V reveals it.', GOOD)
        self.run_job('Generating a test scene (downloads MNIST the first time)', fn, done)

    # ------------------------------------------------------------ input

    def mouse_image_pos(self):
        mx, my = pygame.mouse.get_pos()
        if IMG_X <= mx < IMG_X + IMG_W and IMG_Y <= my < IMG_Y + IMG_H:
            return mx - IMG_X, my - IMG_Y
        return None

    def claim(self, pos):
        x, y = pos
        gt = self.scene.three_boxes()
        inside = [i for i, b in enumerate(gt) if b[0] <= x <= b[2] and b[1] <= y <= b[3]]
        new = [i for i in inside if i not in self.st.found]
        if new:
            i = min(new, key=lambda i: math.hypot(*(gt[i, :2] + gt[i, 2:]) / 2 - (x, y)))
            self.st.found.add(i)
            left = len(gt) - len(self.st.found)
            self.say('Found a 3!' + (f' {left} left.' if left else ' That was the last one!'), GOOD)
        elif inside:
            self.say('You already found that 3.', AMBER)
        else:
            self.st.misses.append((x, y))
            self.say('No 3 there.', WARN)

    def handle_key(self, key):
        k = pygame.key.name(key)
        if key == pygame.K_TAB:
            self.mode = MODES[(MODES.index(self.mode) + 1) % len(MODES)]
            self.say(HINTS[self.mode].split('  ')[0], FG)
        elif k in '12345' and len(k) == 1:
            i = int(k) - 1
            self.set_cfg(L.PRESETS[i], i)
            self.say(f'HoG preset {k}: {self.cfg.name} ({self.cfg.summary()})', ACCENT)
        elif k in ('b', 'c', 'u', 'o'):
            c = self.cfg
            if k == 'b':
                c = L.HogConfig('Custom', c.layout, c.n_cells, L.BIN_CHOICES[
                    (L.BIN_CHOICES.index(c.n_bins) + 1) % len(L.BIN_CHOICES)
                    if c.n_bins in L.BIN_CHOICES else 0], c.signed, c.align)
            elif k == 'c':
                if c.layout == 'rings':
                    cells = c.n_cells % 4 + 1
                else:
                    cells = L.CELL_CHOICES[(L.CELL_CHOICES.index(c.n_cells) + 1) % len(L.CELL_CHOICES)
                                           if c.n_cells in L.CELL_CHOICES else 0]
                c = L.HogConfig('Custom', c.layout, cells, c.n_bins, c.signed, c.align)
            elif k == 'u':
                c = L.HogConfig('Custom', c.layout, c.n_cells, c.n_bins, not c.signed, c.align)
            else:
                c = L.HogConfig('Custom', c.layout, c.n_cells, c.n_bins, c.signed, not c.align)
            self.set_cfg(c)
            self.say(f'HoG: {c.summary()}', ACCENT)
        elif k == 'a':
            self.upright = not self.upright
            self.update_reference()
            self.say('Reference: mean of the train 3s ' + ('de-rotated to upright' if self.upright
                     else 'as found, at their random angles'), ACCENT)
        elif k == 'g':
            self.view = 'glyphs' if self.view == 'bars' else 'bars'
        elif k == 'v':
            if self.scene.revealable:
                self.st.revealed = not self.st.revealed
            else:
                self.say('The blind test stays hidden. N makes a test scene you can reveal.', AMBER)
        elif k == 't':
            self.scene_i = (self.scene_i + 1) % len(self.scenes)
            self.say(f'Scene: {self.scene.name}', FG)
        elif k == 'n':
            self.new_scene()
        elif k == 'r':
            self.states[self.scene_i] = SceneState()
            self.say(f'Reset {self.scene.name}', FG)
        elif k == 'q':
            self.probe_angle -= 5
        elif k == 'e':
            self.probe_angle += 5
        elif k == '0':
            self.probe_size, self.probe_angle = float(L.WINDOW), 0.0
        elif k == 'm':
            if self.scorer == 'template' and not self.svm_ok():
                self.say('Train the SVM for this HoG first (Train mode, Space)', AMBER)
            else:
                self.scorer = 'svm' if self.scorer == 'template' else 'template'
                self.say(f'Scoring with the {self.scorer}', ACCENT)
                self.live_key = None
        elif k in ('[', ']'):
            step = 0.025 if self.scorer == 'template' else 0.25
            self.thresholds[self.scorer] = self.threshold(self.scorer) + (step if k == ']' else -step)
            self.say(f'{self.scorer} threshold {self.thresholds[self.scorer]:.3f}', ACCENT)
            if self.mode == 'scan' and self.st.scan is not None:
                self.start_scan()
        elif k == 'h':
            if self.st.heat is not None:
                self.st.heat = None
            else:
                self.start_heat()
        elif k == 'p':
            if self.live is not None:
                np.set_printoptions(precision=3, suppress=True, linewidth=120)
                print(f'{self.cfg.summary()}\n{self.live.reshape(self.cfg.n_regions, -1)}')
        elif key == pygame.K_SPACE:
            self.action(self.mouse_image_pos())

    def action(self, pos):
        if self.mode == 'manual':
            if pos is not None:
                self.claim(pos)
        elif self.mode == 'scan':
            self.start_scan()
        else:
            self.start_train()

    # ------------------------------------------------------------ live descriptor

    def update_live(self):
        pos = self.mouse_image_pos()
        key = (pos, self.probe_size, self.probe_angle, self.cfg, self.scene_i, id(self.hd))
        if pos is None or key == self.live_key:
            return
        self.live_key = key
        self.patch = L.extract_patch(self.scene.image, pos, L.WINDOW + 2, self.probe_angle,
                                     self.probe_size / L.WINDOW)
        try:
            self.live = np.asarray(L.patch_descriptor(self.patch, self.cfg, self.hd), float)
            self.live_err = None
        except Exception as e:
            self.live, self.live_err = None, self.error_text(e)

    # ------------------------------------------------------------ drawing: HoG

    def text(self, s, pos, font=None, color=FG, max_w=None):
        font = font or self.font
        if max_w is not None:
            while s and font.size(s)[0] > max_w:
                s = s[:-2] + '…'
        surf = font.render(s, True, color)
        self.screen.blit(surf, pos)
        return surf.get_width()

    def wrap(self, s, rect, font=None, color=MUTED):
        font = font or self.small
        words, line, y = s.split(' '), '', rect.y
        for w in words:
            if font.size(line + w)[0] > rect.w and line:
                self.text(line, (rect.x, y), font, color)
                y += font.get_linesize()
                line = ''
            line += w + ' '
        self.text(line, (rect.x, y), font, color)

    def draw_hog(self, rect, desc, cfg, view=None):
        view = view or self.view
        B = cfg.n_bins
        d = np.asarray(desc, float).reshape(cfg.n_regions, B)
        vmax = max(np.abs(d).max(), 1e-9)
        colors = [bin_color(cfg, b) for b in range(B)]
        scr = self.screen
        if cfg.layout == 'rings':
            names = ['inner', 'middle', 'outer'] if cfg.n_cells == 3 else [f'ring {i + 1}' for i in range(cfg.n_cells)]
            cw = rect.w / cfg.n_cells
            for r in range(cfg.n_cells):
                cell = pygame.Rect(rect.x + r * cw + 2, rect.y, cw - 4, rect.h - 16)
                self.draw_bars(cell, d[r], vmax, colors)
                self.text(names[r], (cell.x, cell.bottom + 2), self.small, MUTED)
            return
        n = cfg.n_cells
        if view == 'glyphs':
            side = min(rect.w, rect.h)
            box = pygame.Rect(rect.x + (rect.w - side) // 2, rect.y + (rect.h - side) // 2, side, side)
            pygame.draw.rect(scr, GLYPH_BG, box)
            c = side / n
            ang = np.radians(L.bin_centers(cfg))
            for i in range(n):
                for j in range(n):
                    cx, cy = box.x + (j + 0.5) * c, box.y + (i + 0.5) * c
                    for b in range(B):
                        v = d[i * n + j, b] / vmax
                        if v <= 0.02:
                            continue
                        ln = v * c * 0.47
                        dx, dy = math.cos(ang[b]) * ln, math.sin(ang[b]) * ln
                        start = (cx - dx, cy - dy) if not cfg.signed else (cx, cy)
                        pygame.draw.line(scr, colors[b], start, (cx + dx, cy + dy), 2)
            for i in range(1, n):
                pygame.draw.line(scr, (51, 65, 85), (box.x + i * c, box.y), (box.x + i * c, box.bottom))
                pygame.draw.line(scr, (51, 65, 85), (box.x, box.y + i * c), (box.right, box.y + i * c))
            return
        cw, ch = rect.w / n, rect.h / n
        for i in range(n):
            for j in range(n):
                cell = pygame.Rect(rect.x + j * cw + 1, rect.y + i * ch + 1, cw - 2, ch - 2)
                self.draw_bars(cell, d[i * n + j], vmax, colors)

    def draw_bars(self, cell, values, vmax, colors):
        scr = self.screen
        pygame.draw.rect(scr, (255, 255, 255), cell)
        pygame.draw.rect(scr, TRACK, cell, 1)
        B = len(values)
        bw = (cell.w - 2) / B
        for b, v in enumerate(values):
            h = abs(v) / vmax * (cell.h - 3)
            if h >= 0.5:
                pygame.draw.rect(scr, colors[b], pygame.Rect(cell.x + 1 + b * bw, cell.bottom - 1 - h,
                                                             max(1, bw - 1), h))

    def draw_wheel(self, center, radius, cfg):
        pygame.draw.circle(self.screen, TRACK, center, radius, 1)
        for b, a in enumerate(np.radians(L.bin_centers(cfg))):
            dx, dy = math.cos(a) * radius, math.sin(a) * radius
            start = center if cfg.signed else (center[0] - dx, center[1] - dy)
            pygame.draw.line(self.screen, bin_color(cfg, b), start, (center[0] + dx, center[1] + dy), 3)

    def thumbs(self, x, y, size):
        p = self.patch[1:-1, 1:-1]
        self.screen.blit(pygame.transform.scale(gray_surface(p), (size, size)), (x, y))
        try:
            mag, ang = self.hd.image_gradients(self.patch)
            mag, ang = np.asarray(mag)[1:-1, 1:-1], np.asarray(ang)[1:-1, 1:-1]
        except Exception:
            self.text('needs E1', (x, y + size + 10), self.small, MUTED)
            return
        m = mag / max(mag.max(), 1e-9)
        self.screen.blit(pygame.transform.scale(gray_surface(1 - m), (size, size)), (x, y + size + 4))
        period = 360.0 if self.cfg.signed else 180.0
        hue = (ang % period) / period
        rgb = np.zeros(p.shape + (3,))
        for k, off in enumerate((0, 1 / 3, 2 / 3)):          # cheap hue wheel
            rgb[..., k] = 0.5 + 0.5 * np.cos(2 * np.pi * (hue - off))
        rgb = (rgb * m[..., None] * 255).astype(np.uint8)
        surf = pygame.surfarray.make_surface(np.ascontiguousarray(rgb.swapaxes(0, 1)))
        self.screen.blit(pygame.transform.scale(surf, (size, size)), (x, y + 2 * (size + 4)))

    # ------------------------------------------------------------ drawing: panes

    def draw_sidebar(self):
        x, y, w = SIDE_X, IMG_Y, SIDE_W
        scr = self.screen
        scr.blit(self.icon_small, (x, y - 4))
        self.text('Live HoG', (x + 32, y), self.bold)
        self.draw_wheel((x + w - 20, y + 16), 17, self.cfg)
        self.text(f'probe {self.probe_size:.0f} px, {self.probe_angle:+.0f} deg', (x + 32, y + 20),
                  self.small, MUTED)
        name = f'{self.preset_i + 1} {self.cfg.name}' if self.preset_i is not None else 'Custom'
        self.text(f'{name}: {self.cfg.summary()}', (x, y + 40), self.small, FG, max_w=w)

        live = pygame.Rect(x, y + 60, w, 190)
        if self.st.revealed and getattr(self, 'patch', None) is not None:
            self.thumbs(x, live.y, 58)                 # slide 48: patch, |gradient|, angle
            live = pygame.Rect(x + 66, live.y, w - 66, live.h)
        if self.mouse_image_pos() is None and self.live is None and not self.live_err:
            self.wrap('Move the magnifier over the image.', live)
        elif self.live_err:
            self.wrap(self.live_err, live, color=WARN)
        elif self.live is not None:
            self.draw_hog(live, self.live, self.cfg)

        y2 = y + 262
        pygame.draw.line(scr, TRACK, (x, y2 - 6), (x + w, y2 - 6))
        if self.scorer == 'svm' and self.svm_ok():
            self.text('SVM weights:  w > 0 (3)  |  w < 0 (not 3)', (x, y2), self.small, FG)
            wt = self.svm['model'].w
            half = (w - 8) // 2
            self.draw_hog(pygame.Rect(x, y2 + 18, half, 140), np.clip(wt, 0, None), self.cfg)
            self.draw_hog(pygame.Rect(x + half + 8, y2 + 18, half, 140), np.clip(-wt, 0, None), self.cfg)
            if self.live is not None:
                s = float(self.svm['model'].decision_function(self.live[None])[0])
                thr = self.threshold('svm')
                self.meter(y2 + 170, f'SVM score {s:+.2f}', (s + 3) / 6, s > thr, s > thr - 1)
        else:
            n3 = len(self.scenes[0].threes)
            how = 'upright' if self.upright else 'as found'
            self.text(f'Reference: mean of {n3} train 3s ({how}, A)', (x, y2), self.small, FG)
            ref = pygame.Rect(x, y2 + 18, w, 140)
            if self.ref is None:
                self.wrap(self.ref_err or '', ref, color=WARN)
            else:
                self.draw_hog(ref, self.ref.ref, self.cfg)
                if self.live is not None:
                    s, thr = float(self.ref.score(self.live)), self.threshold('template')
                    self.meter(y2 + 170, f'Match {s:+.2f}  (beyond the average digit)', (s + 0.2) / 0.8,
                               s > thr, s > thr - 0.15)

    def meter(self, y, label, frac, hit, close):
        x, w = SIDE_X, SIDE_W
        color = GOOD if hit else (AMBER if close else WARN)
        self.text(label, (x, y), self.bold, color)
        bar = pygame.Rect(x, y + 22, w, 12)
        pygame.draw.rect(self.screen, TRACK, bar, border_radius=6)
        fill = bar.copy()
        fill.w = int(w * min(max(frac, 0), 1))
        pygame.draw.rect(self.screen, color, fill, border_radius=6)

    def draw_image(self):
        scr, st, scene = self.screen, self.st, self.scene
        pane = pygame.Rect(IMG_X, IMG_Y, IMG_W, IMG_H)
        shown = st.revealed or (self.mode == 'train' and self.scene_i == 0 and self.svm is not None)
        if shown:
            if getattr(scene, '_surf', None) is None:
                scene._surf = gray_surface(scene.image)
            scr.blit(scene._surf, pane)
        else:
            pygame.draw.rect(scr, PANE_BG, pane)
            for gx in range(L.WINDOW, IMG_W, L.WINDOW):
                pygame.draw.line(scr, TRACK, (IMG_X + gx, IMG_Y), (IMG_X + gx, IMG_Y + IMG_H))
            for gy in range(L.WINDOW, IMG_H, L.WINDOW):
                pygame.draw.line(scr, TRACK, (IMG_X, IMG_Y + gy), (IMG_X + IMG_W, IMG_Y + gy))
            self.text('hidden' + ('' if scene.revealable else ' (blind test)'),
                      (IMG_X + 8, IMG_Y + IMG_H - 20), self.small, MUTED)
        scr.set_clip(pane)
        if st.heat is not None and self.mode == 'scan':
            scr.blit(st.heat, pane)
        gt = scene.three_boxes()
        if self.mode == 'manual':
            for i in st.found:
                self.rect(gt[i], GOOD, 2)
            for mx, my in st.misses:
                px, py = IMG_X + mx, IMG_Y + my
                pygame.draw.line(scr, WARN, (px - 6, py - 6), (px + 6, py + 6), 3)
                pygame.draw.line(scr, WARN, (px - 6, py + 6), (px + 6, py - 6), 3)
        elif self.mode == 'scan' and st.scan is not None:
            r = st.scan
            for b, s, tp in zip(r['boxes'], r['scores'], r['ev']['is_tp']):
                color = GOOD if tp else WARN
                self.rect(b, color, 2)
                self.text(f'{s:.2f}', (IMG_X + b[0], IMG_Y + b[1] - 13), self.small, color)
        elif self.mode == 'train' and self.svm is not None and self.scene_i == 0:
            for cx, cy, size, angle, label in self.svm['wins'][::3]:
                pygame.draw.circle(scr, GOOD if label > 0 else WARN, (IMG_X + cx, IMG_Y + cy), 2)
        scr.set_clip(None)
        pygame.draw.rect(scr, TRACK, pane, 1)

        pos = self.mouse_image_pos()
        pygame.mouse.set_visible(pos is None)
        if pos is not None:
            c = np.array([IMG_X + pos[0], IMG_Y + pos[1]])
            half = self.probe_size / 2
            corners = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
            pts = c + corners @ L.rot(self.probe_angle).T
            pygame.draw.polygon(scr, ACCENT, pts.tolist(), 2)
            r_needed = half * math.sqrt(2) + 4
            size = int(512 * r_needed / LENS_RADIUS)
            if size not in self.lens_cache:
                self.lens_cache[size] = pygame.transform.smoothscale(self.lens, (size, size))
            off = LENS_CENTER * size / 512
            scr.blit(self.lens_cache[size], (c[0] - off, c[1] - off))

    def rect(self, b, color, width):
        pygame.draw.rect(self.screen, color, pygame.Rect(IMG_X + b[0], IMG_Y + b[1],
                                                         b[2] - b[0], b[3] - b[1]), width)

    def draw_bar(self):
        scr = self.screen
        y = BAR_Y
        pygame.draw.line(scr, TRACK, (PAD, y - 6), (WIN_W - PAD, y - 6))
        x = PAD
        self.mode_buttons = []
        for m in MODES:
            label = MODE_NAMES[m]
            r = pygame.Rect(x, y, self.bold.size(label)[0] + 20, 26)
            active = m == self.mode
            pygame.draw.rect(scr, ACCENT if active else PANE_BG, r, border_radius=6)
            self.text(label, (r.x + 10, r.y + 4), self.bold, (255, 255, 255) if active else FG)
            self.mode_buttons.append((r, m))
            x = r.right + 6
        x += 10
        x += self.text(f'Scene: {self.scene.name}', (x, y + 5), self.bold) + 18
        st, n3 = self.st, len(self.scene.threes)
        if self.mode == 'manual':
            info = f'3s left {n3 - len(st.found)} of {n3}    misses {len(st.misses)}'
        elif self.mode == 'scan':
            thr = self.threshold(self.scorer)
            info = f'{self.scorer} (M), threshold {thr:.3f} ([ ])'
            if st.scan is not None:
                e = st.scan['ev']
                info = (f'{st.scan["scorer"]} @ {st.scan["thr"]:.3f}: TP {e["tp"]} FP {e["fp"]} FN {e["fn"]}'
                        f'  P {e["precision"]:.2f} R {e["recall"]:.2f} AP {e["ap"]:.2f}')
        else:
            s = self.svm
            info = ('not trained yet' if s is None else
                    f'{s["n_pos"]} pos / {s["n_neg"]} neg windows, train accuracy {100 * s["acc"]:.1f}%'
                    + ('' if self.svm_ok() else '  (other HoG: retrain)'))
        self.text(info, (x, y + 6), self.font, FG, max_w=WIN_W - PAD - x)

        y += 34
        x = PAD
        for tag, name, status, detail in self.status:
            word, color = STATUS_STYLE[status]
            label = f'{tag} {word}' + (f' {detail}' if status == 'needs' else '')
            r = pygame.Rect(x, y, self.small.size(label)[0] + 12, 20)
            pygame.draw.rect(scr, color, r, 1, border_radius=5)
            self.text(label, (r.x + 6, r.y + 3), self.small, color)
            x = r.right + 5
        msg = self.msg if not self.busy else f'{self.busy} ...'
        self.text(msg, (x + 8, y + 3), self.small, self.msg_color if not self.busy else ACCENT,
                  max_w=WIN_W - PAD - x - 8)

    def draw(self):
        self.screen.fill(BG)
        self.text('HoG I spy: find every 3', (PAD, 8), self.bold)
        self.text(HINTS[self.mode], (PAD, 30), self.small, MUTED, max_w=WIN_W - 2 * PAD)
        self.draw_image()
        self.draw_sidebar()
        self.draw_bar()
        pygame.display.flip()

    # ------------------------------------------------------------ main loop

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    return
                if event.type == pygame.KEYDOWN:
                    self.handle_key(event.key)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    for r, m in getattr(self, 'mode_buttons', []):
                        if r.collidepoint(event.pos):
                            self.mode = m
                            break
                    else:
                        pos = self.mouse_image_pos()
                        if pos is not None and self.mode == 'manual':
                            self.claim(pos)
                elif event.type == pygame.MOUSEWHEEL:
                    self.probe_size = float(np.clip(self.probe_size + 2 * event.y, 24, 96))
            self.check_reload()
            self.poll_jobs()
            self.update_live()
            self.draw()
            self.clock.tick(60)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--no-answers', action='store_true', help='ignore released answers in answers/')
    App(ap.parse_args()).run()
    pygame.quit()


if __name__ == '__main__':
    main()
