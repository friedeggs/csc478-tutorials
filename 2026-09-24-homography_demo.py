"""Interactive homography demo for Tutorial 3 (2026-09-24, Image Formation).

The homography is built from its parameters, following the 2D transformation
hierarchy of Lecture 3 (affine + perspective skew), in coordinates centred
on the image centre c:

        [ A   0 ]                    [ cos t  -sin t ] [ 1  k ]
    G = [       ]      A = s R K = s [              ] [      ]
        [ p   1 ]                    [ sin t   cos t ] [ 0  1 ]

    H = T(c + (tx, ty)) @ G @ T(-c)

where T(v) is the 3x3 translation by v, p = (p1, p2), and H is scaled so
H[2, 2] = 1. s is the scale factor, t the rotation angle and k the shear
(x shifts by k * y, applied before rotating). Scale, rotation, shear and the
perspective skew all act about c, so the image centre stays put while
(tx, ty) = 0. The price: with skew, the printed H mixes the
parameters, e.g. its bottom row is (p1, p2) / (1 - p1 cx - p2 cy).

Left pane: the source image with a draggable source quadrilateral (starts on
the poster corners). Right pane: the source image warped live by H, with the
target quadrilateral H @ source (computed, not draggable). Corners correspond
by colour and label (TL, TR, BR, BL) across the two panes.

Controls (middle column):
    tx, ty, s, k, p1, p2 sliders, and a knob for the angle t.
    Drag to set, scroll wheel for fine steps, right-click to reset one control.

Keys:
    R        reset the controls and the source corners
    M        toggle masking the warped image to the target quadrilateral
    P        print the corners and H to the terminal
    Esc / Q  quit

Run:  python 2026-09-24-homography_demo.py    (needs pygame: pip install pygame)
"""
import math
from pathlib import Path

import cv2
import numpy as np
import pygame

HERE = Path(__file__).resolve().parent
IMG_FILENAME = HERE / 'assets' / 'CS478_Poster_Warped.png'
DATA_FILENAME = HERE / 'assets' / 'CS478_Poster_Warped_corners.txt'

PAD = 50           # minimum margin around each image, so corners can sit off it
GAP = 16           # space between panes
TOP = 44           # title row height
CENTER_W = 370     # width of the middle column (H and its controls)
TEXT_H = 96        # coordinate readout under each pane
HANDLE_R = 8       # corner handle radius (pixels)
DECIMALS = 4       # H is floored to this many decimals for display
POSTER_ASPECT = 1545 / 2000  # width / height of the true, unwarped poster

BG = (248, 250, 252)          # #f8fafc
FG = (15, 23, 42)             # #0f172a
MUTED = (100, 116, 139)       # #64748b
TRACK = (203, 213, 225)       # #cbd5e1
ACCENT = (37, 99, 235)        # #2563eb
PANE_BG = (226, 232, 240)     # #e2e8f0
WARN = (220, 38, 38)
CORNER_NAMES = ['TL', 'TR', 'BR', 'BL']
CORNER_COLORS = [(220, 38, 38), (22, 163, 74), (37, 99, 235), (217, 119, 6)]
MONO = 'menlo,consolas,dejavusansmono,couriernew,monospace'


def build_homography(tx, ty, scale, theta_deg, shear, p1, p2, center):
    """Shear, rotation by theta, scaling and perspective skew (p1, p2), all
    about center, then translation by (tx, ty). Scaled so H[2, 2] = 1."""
    c, s = math.cos(math.radians(theta_deg)), math.sin(math.radians(theta_deg))
    R = np.array([[c, -s], [s, c]])
    K = np.array([[1.0, shear], [0.0, 1.0]])
    G = np.eye(3)
    G[:2, :2] = scale * R @ K
    G[2, :2] = p1, p2
    cx, cy = center
    to_centre = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float64)
    from_centre = np.array([[1, 0, cx + tx], [0, 1, cy + ty], [0, 0, 1]],
                           dtype=np.float64)
    H = from_centre @ G @ to_centre
    return H / H[2, 2]   # H[2, 2] = 1 - p1 cx - p2 cy > 0 for the slider range


def apply_homography(H, pts):
    """(N, 2) points -> (N, 2) transformed points and their (N,) w values."""
    ph = np.column_stack([pts, np.ones(len(pts))]) @ H.T
    return ph[:, :2] / ph[:, 2:], ph[:, 2]


def floor_decimals(M, decimals=DECIMALS):
    """Round down (towards -inf) to a fixed number of decimals; + 0.0 turns
    -0.0 into 0.0 so it prints without a stray minus sign."""
    scale = 10 ** decimals
    return np.floor(M * scale) / scale + 0.0


def format_matrix(H, decimals=DECIMALS):
    """Fixed-point rows (never scientific notation), floored."""
    Hf = floor_decimals(H, decimals)
    width = max(len(f'{v:.{decimals}f}') for v in Hf.ravel())
    return ['[ ' + '  '.join(f'{v:>{width}.{decimals}f}' for v in row) + ' ]'
            for row in Hf]


def to_surface(bgr):
    """OpenCV BGR uint8 image -> pygame Surface."""
    rgb = bgr[..., ::-1]
    return pygame.surfarray.make_surface(np.ascontiguousarray(rgb.swapaxes(0, 1)))


class Slider:
    """Horizontal slider: label and value on one line, track below."""
    HEIGHT = 46

    def __init__(self, label, lo, hi, default, step, fmt):
        self.label, self.lo, self.hi, self.step, self.fmt = label, lo, hi, step, fmt
        self.default = self.value = default
        self.track = pygame.Rect(0, 0, 0, 0)

    def place(self, x, y, w):
        self.track = pygame.Rect(x, y + 28, w, 6)

    def set(self, v):
        v = round(v / self.step) * self.step
        self.value = min(max(v, self.lo), self.hi) + 0.0

    def reset(self):
        self.value = self.default

    def hit(self, pos):
        return self.track.inflate(16, 24).collidepoint(pos)

    def drag(self, pos):
        frac = (pos[0] - self.track.x) / max(self.track.w, 1)
        self.set(self.lo + frac * (self.hi - self.lo))

    def nudge(self, n):
        self.set(self.value + n * self.step)

    def draw(self, screen, fonts, active):
        font, small = fonts
        screen.blit(font.render(self.label, True, FG),
                    (self.track.x, self.track.y - 28))
        value = font.render(format(self.value, self.fmt), True, FG)
        screen.blit(value, (self.track.right - value.get_width(), self.track.y - 28))
        pygame.draw.rect(screen, TRACK, self.track, border_radius=3)
        # Filled from the default value to the current one.
        x0 = self.track.x + (self.default - self.lo) / (self.hi - self.lo) * self.track.w
        x1 = self.track.x + (self.value - self.lo) / (self.hi - self.lo) * self.track.w
        fill = pygame.Rect(int(min(x0, x1)), self.track.y, int(abs(x1 - x0)), self.track.h)
        pygame.draw.rect(screen, ACCENT, fill, border_radius=3)
        pygame.draw.circle(screen, ACCENT, (x1, self.track.centery), 10 if active else 8)
        pygame.draw.circle(screen, BG, (x1, self.track.centery), 10 if active else 8, 2)


class Knob:
    """Rotary knob for an angle in degrees. The pointer shows the direction
    the image x-axis is rotated to (screen y points down, like the image)."""
    RADIUS = 38
    HEIGHT = 2 * RADIUS + 40

    def __init__(self, label, default=0.0, step=0.5):
        self.label, self.step = label, step
        self.default = self.value = default
        self.center = (0, 0)

    def place(self, x, y, w):
        self.center = (x + w // 2, y + self.RADIUS + 6)

    def set(self, v):
        v = round(v / self.step) * self.step
        self.value = (v + 180) % 360 - 180 + 0.0   # wrap to [-180, 180)

    def reset(self):
        self.value = self.default

    def hit(self, pos):
        dx, dy = pos[0] - self.center[0], pos[1] - self.center[1]
        return dx * dx + dy * dy <= (self.RADIUS + 10) ** 2

    def drag(self, pos):
        dx, dy = pos[0] - self.center[0], pos[1] - self.center[1]
        if dx or dy:
            self.set(math.degrees(math.atan2(dy, dx)))

    def nudge(self, n):
        self.set(self.value + n * 1.0)

    def draw(self, screen, fonts, active):
        font, small = fonts
        cx, cy = self.center
        r = self.RADIUS
        pygame.draw.circle(screen, TRACK, self.center, r)
        pygame.draw.circle(screen, ACCENT if active else MUTED, self.center, r, 2)
        for k in range(8):  # ticks every 45 degrees
            a = math.radians(45 * k)
            inner = r - (8 if k % 2 == 0 else 4)
            pygame.draw.line(screen, MUTED,
                             (cx + inner * math.cos(a), cy + inner * math.sin(a)),
                             (cx + r * math.cos(a), cy + r * math.sin(a)), 2)
        a = math.radians(self.value)
        tip = (cx + (r - 6) * math.cos(a), cy + (r - 6) * math.sin(a))
        pygame.draw.line(screen, ACCENT, self.center, tip, 3)
        pygame.draw.circle(screen, ACCENT, tip, 5)
        pygame.draw.circle(screen, FG, self.center, 3)
        text = font.render(f'{self.label}  {self.value:7.1f}°', True, FG)
        screen.blit(text, (cx - text.get_width() // 2, cy + r + 8))


class Pane:
    """A pane of screen with an image drawn at origin (pane pixels) and four
    corners stored in that image's pixel coordinates."""

    def __init__(self, x, y, w, h, origin, title, corners, draggable):
        self.rect = pygame.Rect(x, y, w, h)
        self.origin = origin
        self.title = title
        self.draggable = draggable
        self.initial = np.asarray(corners, dtype=np.float64)
        self.corners = self.initial.copy()

    def reset(self):
        self.corners = self.initial.copy()

    def to_screen(self, p):
        ox, oy = self.origin
        return (self.rect.x + ox + p[0], self.rect.y + oy + p[1])

    def from_screen(self, pos):
        ox, oy = self.origin
        x = pos[0] - self.rect.x - ox
        y = pos[1] - self.rect.y - oy
        # Keep handles inside the pane.
        x = min(max(x, -ox), self.rect.w - ox)
        y = min(max(y, -oy), self.rect.h - oy)
        return np.array([x, y], dtype=np.float64)

    def hit(self, pos):
        """Index of the corner handle under pos, or None."""
        if not self.draggable:
            return None
        best, best_d = None, (2 * HANDLE_R) ** 2
        for i, p in enumerate(self.corners):
            sx, sy = self.to_screen(p)
            d = (sx - pos[0]) ** 2 + (sy - pos[1]) ** 2
            if d <= best_d:
                best, best_d = i, d
        return best

    def draw(self, screen, image_surface, fonts, active=None, show_quad=True):
        font, small = fonts
        screen.fill(PANE_BG, self.rect)
        screen.set_clip(self.rect)
        if image_surface is not None:
            screen.blit(image_surface, (self.rect.x, self.rect.y))
        if show_quad:
            pts = [self.to_screen(p) for p in self.corners]
            pygame.draw.lines(screen, FG, True, pts, 2)
            for i, (sx, sy) in enumerate(pts):
                r = HANDLE_R + 2 if i == active else HANDLE_R
                if self.draggable:
                    pygame.draw.circle(screen, CORNER_COLORS[i], (sx, sy), r)
                    pygame.draw.circle(screen, BG, (sx, sy), r, 2)
                else:  # computed corners: hollow markers
                    pygame.draw.circle(screen, BG, (sx, sy), r)
                    pygame.draw.circle(screen, CORNER_COLORS[i], (sx, sy), r, 3)
                label = small.render(CORNER_NAMES[i], True, CORNER_COLORS[i])
                screen.blit(label, (sx + r + 3, sy - r - label.get_height() + 4))
        screen.set_clip(None)
        pygame.draw.rect(screen, MUTED, self.rect, 1)

        title = font.render(self.title, True, FG)
        screen.blit(title, (self.rect.x, self.rect.y - title.get_height() - 8))

        # Corner coordinates (image pixels) beneath the pane, two per row.
        y = self.rect.bottom + 12
        for row in ((0, 1), (3, 2)):
            x = self.rect.x
            for i in row:
                px, py = self.corners[i]
                name = font.render(f'{CORNER_NAMES[i]} ', True, CORNER_COLORS[i])
                screen.blit(name, (x, y))
                text = font.render(f'({px:7.1f}, {py:7.1f})', True, FG)
                screen.blit(text, (x + name.get_width(), y))
                x += self.rect.w // 2
            y += font.get_linesize() + 4


def main():
    source_image = cv2.imread(str(IMG_FILENAME), cv2.IMREAD_COLOR)
    if source_image is None:
        raise SystemExit(f'Could not read {IMG_FILENAME}')
    src_h, src_w = source_image.shape[:2]
    source_corners = np.loadtxt(DATA_FILENAME)

    # Panes have the aspect ratio of the true, unwarped poster, and are just
    # big enough to hold the source image with at least PAD on every side.
    aspect = POSTER_ASPECT
    pane_h = src_h + 2 * PAD
    pane_w = round(pane_h * aspect)
    if pane_w < src_w + 2 * PAD:
        pane_w = src_w + 2 * PAD
        pane_h = round(pane_w / aspect)
    origin = ((pane_w - src_w) // 2, (pane_h - src_h) // 2)  # image centred
    win_w = 2 * pane_w + CENTER_W + 4 * GAP
    win_h = TOP + pane_h + TEXT_H

    pygame.init()
    pygame.display.set_caption('Homography demo - 2026-09-24')
    screen = pygame.display.set_mode((win_w, win_h))
    font = pygame.font.SysFont(MONO, 16)
    small = pygame.font.SysFont(MONO, 13, bold=True)
    fonts = (font, small)
    clock = pygame.time.Clock()

    left = Pane(GAP, TOP, pane_w, pane_h, origin, 'Source (drag corners)',
                source_corners, draggable=True)
    right_x = GAP + pane_w + GAP + CENTER_W + GAP
    right = Pane(right_x, TOP, pane_w, pane_h, origin, 'Target = H @ source',
                 source_corners, draggable=False)

    # Perspective skew, centred on the image: keep it small enough that
    # w = p1 (x - cx) + p2 (y - cy) + 1 stays positive over the image and its
    # margin: p_max * (pane_w / 2 + pane_h / 2) = 0.0015 * 532 < 1.
    p_max = 0.0015
    tx = Slider('tx', -300, 300, 0.0, 1.0, '8.1f')
    ty = Slider('ty', -300, 300, 0.0, 1.0, '8.1f')
    scale = Slider('s (scale)', 0.25, 2.0, 1.0, 0.01, '8.2f')
    theta = Knob('θ')
    shear = Slider('k (shear)', -1.0, 1.0, 0.0, 0.01, '8.2f')
    p1 = Slider('p1 (skew x)', -p_max, p_max, 0.0, 0.00001, '.5f')
    p2 = Slider('p2 (skew y)', -p_max, p_max, 0.0, 0.00001, '.5f')
    controls = [tx, ty, scale, theta, shear, p1, p2]

    # Middle column layout: H on top, then the controls, then key help.
    cx = GAP + pane_w + GAP
    inner_x, inner_w = cx + 24, CENTER_W - 48
    matrix_y = TOP + 8
    y = matrix_y + 5 * (font.get_linesize() + 4) + 8
    for control in controls:
        control.place(inner_x, y, inner_w)
        y += control.HEIGHT
    help_y = y + 4

    # The source image never moves: draw it once onto a pane-sized canvas.
    canvas = np.empty((pane_h, pane_w, 3), dtype=np.uint8)
    canvas[:] = PANE_BG[::-1]
    ox, oy = origin
    canvas[oy:oy + src_h, ox:ox + src_w] = source_image
    left_surface = to_surface(canvas)

    # Image pixel coordinates -> right-pane canvas coordinates.
    T = np.array([[1, 0, ox], [0, 1, oy], [0, 0, 1]], dtype=np.float64)

    dragging = None    # corner index in the left pane
    active = None      # control being dragged
    mask_to_quad = False
    dirty = True
    H, right_surface, behind = None, None, False

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    left.reset()
                    for control in controls:
                        control.reset()
                    dirty = True
                elif event.key == pygame.K_m:
                    mask_to_quad = not mask_to_quad
                    dirty = True
                elif event.key == pygame.K_p:
                    print(f'tx = {tx.value}, ty = {ty.value}, s = {scale.value:.2f}, '
                          f'theta = {theta.value} deg, k = {shear.value:.2f}, '
                          f'p1 = {p1.value:.5f}, p2 = {p2.value:.5f}')
                    print('Source corners (TL, TR, BR, BL):')
                    print(np.array2string(left.corners, suppress_small=True))
                    print('Target corners (TL, TR, BR, BL):')
                    print(np.array2string(right.corners, suppress_small=True))
                    print('H (floored):')
                    print('\n'.join(format_matrix(H)))
                    print()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button in (1, 3):
                hit = next((c for c in controls if c.hit(event.pos)), None)
                if event.button == 3:
                    if hit is not None:
                        hit.reset()
                        dirty = True
                elif hit is not None:
                    active = hit
                    active.drag(event.pos)
                    dirty = True
                else:
                    dragging = left.hit(event.pos)
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                dragging = active = None
            elif event.type == pygame.MOUSEMOTION:
                if active is not None:
                    active.drag(event.pos)
                    dirty = True
                elif dragging is not None:
                    left.corners[dragging] = left.from_screen(event.pos)
                    dirty = True
            elif event.type == pygame.MOUSEWHEEL:
                pos = pygame.mouse.get_pos()
                hit = next((c for c in controls if c.hit(pos)), None)
                if hit is not None:
                    hit.nudge(event.y)
                    dirty = True

        if dirty:
            H = build_homography(tx.value, ty.value, scale.value, theta.value,
                                 shear.value, p1.value, p2.value,
                                 center=(src_w / 2, src_h / 2))
            right.corners, w = apply_homography(H, left.corners)
            # A corner with w <= 0 is mapped through the line at infinity,
            # so the target quadrilateral is not a proper polygon.
            behind = bool(np.any(w <= 0))
            warped = cv2.warpPerspective(
                source_image, T @ H, (pane_w, pane_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=PANE_BG[::-1])
            if mask_to_quad and not behind:
                mask = np.zeros((pane_h, pane_w), dtype=np.uint8)
                quad = np.round(right.corners + origin).astype(np.int32)
                cv2.fillPoly(mask, [quad], 1)
                warped[mask == 0] = PANE_BG[::-1]
            right_surface = to_surface(warped)
            dirty = False

        screen.fill(BG)
        left.draw(screen, left_surface, fonts, active=dragging)
        right.draw(screen, right_surface, fonts, show_quad=not behind)

        # Middle column: H, then its controls.
        lines = [(font, 'Homography H', FG),
                 (small, f'floored to {DECIMALS} decimals', MUTED)]
        lines += [(font, row, FG) for row in format_matrix(H)]
        y = matrix_y
        for f, text, color in lines:
            surf = f.render(text, True, color)
            screen.blit(surf, (cx + (CENTER_W - surf.get_width()) // 2, y))
            y += f.get_linesize() + 4
        for control in controls:
            control.draw(screen, fonts, control is active)

        help_lines = [(small, 'drag / scroll; right-click resets one', MUTED),
                      (small, f'M  mask to target quad [{"on" if mask_to_quad else "off"}]',
                       MUTED),
                      (small, 'R  reset   P  print   Q  quit', MUTED)]
        if behind:
            help_lines.insert(0, (small, 'a corner has w <= 0 (past the horizon)', WARN))
        y = help_y
        for f, text, color in help_lines:
            surf = f.render(text, True, color)
            screen.blit(surf, (cx + (CENTER_W - surf.get_width()) // 2, y))
            y += f.get_linesize() + 2

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == '__main__':
    main()
